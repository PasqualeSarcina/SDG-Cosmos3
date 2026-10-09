import base64
import json
import logging
import sys
import time

from PIL import Image

from pathlib import Path
from typing import Annotated

import asyncio
import openai
import pydantic
import tyro

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    # Direct execution puts scripts/ on sys.path, not the project root.
    sys.path.insert(0, str(_PACKAGE_DIR))

from src.caption.structured_caption import extract_xml_tag, parse_structured_caption, media_fields_from_metadata, \
    assemble_caption_json

from src.request.client import chat_completion, create_client, resolve_model, run_batch, with_retries
from src.request.response_errors import response_content

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
log = logging.getLogger(__name__)

class Args(pydantic.BaseModel):
    input_files: Annotated[list[Path] | None, tyro.conf.arg(aliases=("-if",))] = None
    """Path to input manifest files (JSON/JSONL).
    Each entry needs a 'vision_path' (a local path or an http(s)/data URL) and may
    include 'name' and a 'media' dict (resolution/aspect_ratio) — the
    latter is used as the caption's media fields when the image is a remote URL that
    cannot be read locally. Mutually exclusive with --image."""

    image: Annotated[Path | None, tyro.conf.arg(aliases=("-i",))] = None
    """Path to a single image file or a directory of images.
    Only files directly in the directory are processed. Mutually exclusive with --input_files."""

    mask: Path | None = None
    """Optional mask file or directory, paired with --image by filename stem.
    Extensions may differ. Each mask is sent as the second image in its request.
    Not supported with --input-files."""

    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Output directory; image.jpg becomes image.json, with no per-image subdirectories."""

    server: str = "http://localhost:8000/v1"
    """The URL of the OpenAI-compatible VLM API server."""
    model: str | None = None
    """The model to use. If not provided, the first model served will be used."""

    max_workers: int = pydantic.Field(default=16, ge=1)
    """Maximum number of concurrent requests to the API."""
    max_retries: int = pydantic.Field(default=3, ge=1)
    """Maximum number of attempts for each image, including the first request."""
    max_tokens: int = pydantic.Field(default=4096, ge=1)
    """Maximum output tokens per request; increase if the response is truncated."""
    timeout: float = pydantic.Field(default=1200.0, gt=0)
    """Per-request client timeout in seconds; a hung request fails after this and is retried."""

    prompt_template_path: Path | None = None
    """Path to a custom prompt template. Defaults to the built-in image_captioner.txt."""

    debug: bool = False
    """Save complete API responses to output_dir/debug/<name>.attempt-<attempt>.json."""

    raw_content: bool = False
    """Save message.content verbatim to <name>.json, bypassing structured caption parsing."""


def _is_remote_ref(ref: str) -> bool:
    """True if ``ref`` is something the server fetches itself (URL / data URI)."""
    return "://" in ref or ref.startswith("data:")


def _image_url(image_ref: str) -> str:
    """Map a local path or remote ref to the ``image_url`` string the server receives.

    Remote refs (``http(s)://`` or ``data:``) are passed through untouched.
    Local images are embedded as base64 data URLs so a remote server can read
    them without access to the client's filesystem.
    """
    if _is_remote_ref(image_ref):
        return image_ref
    path = Path(image_ref)
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_vlm_messages(
    image_ref: str,
    prompt_template: str,
    mask_ref: str | None = None,
) -> list[dict]:
    """Build one multimodal request, optionally including an original/mask pair."""
    content = [{"type": "image_url", "image_url": {"url": _image_url(image_ref)}}]
    if mask_ref is not None:
        content.append({"type": "image_url", "image_url": {"url": _image_url(mask_ref)}})
    content.append({"type": "text", "text": prompt_template})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


async def _process_single(
    args: Args,
    client: openai.AsyncOpenAI,
    name: str,
    image_ref: str,
    media_override: dict | None,
    prompt_template: str,
    mask_ref: str | None = None,
) -> bool:
    assert args.model

    output_dir = args.output_dir
    output_path = output_dir / f"{name}.json"
    try:
        messages = _build_vlm_messages(
            image_ref, prompt_template,
            mask_ref=mask_ref if mask_ref is not None else (
                str(args.mask) if args.mask is not None and not args.mask.is_dir() else None
            ),
        )
    except (OSError, ValueError) as e:
        log.error("Cannot read image %s (%s): %s", name, image_ref, e)
        return False

    async def _attempt(i_retry: int) -> bool:
        started = time.monotonic()
        log.info("[%s/%s] Sending image %s", i_retry + 1, args.max_retries, name)
        response_path = output_dir / "debug" / f"{name}.attempt-{i_retry}.json" if args.debug else None
        response = await chat_completion(
            client, response_path=response_path,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=0.7,  #nvfp4 0.7
            top_p=0.8,      #nvfp4 0.8
            #extra_body={"top_k": 20, "min_p": 0.0},
            extra_body={
                "top_k": 20,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )
        text = response_content(response, max_tokens=args.max_tokens, response_path=response_path)
        comprehensive_t2i_caption = None
        scene_draft = None
        if not args.raw_content:
            comprehensive_t2i_caption = extract_xml_tag(text, "comprehensive_t2i_caption")
            scene_draft = parse_structured_caption(text)
            if not comprehensive_t2i_caption:
                raise ValueError("Missing or empty <comprehensive_t2i_caption> tag")
            if scene_draft is None:
                raise ValueError("Invalid scene_draft JSON")

        if args.raw_content:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            log.info("Caption saved for %s after %.1fs: %s", name, time.monotonic() - started, output_dir)
            return True

        # Media fields: prefer a manifest-provided override; else ffprobe a local
        # file; else leave empty (e.g. a remote URL ffprobe cannot read).
        if media_override is not None:
            media = media_override
        elif not _is_remote_ref(image_ref):
            try:
                with Image.open(image_ref) as image:
                    media = media_fields_from_metadata(image.size)
            except Exception as e:  # noqa: BLE001 - degrade gracefully, keep the caption
                log.warning(f"Cannot read image dimensions for {name}: {e}; writing caption_json without media fields")
                media = {}
        else:
            media = {}

        caption_json = assemble_caption_json(scene_draft, comprehensive_t2i_caption, media)

        output_dir.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(caption_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )

        # Advisory: the SFT loader truncates very long prompts (see _MAX_CAPTION_TOKENS
        # in sft_dataset.py). ~4 chars/token is a rough guide; warn if the serialized
        # JSON looks large so it can be checked against the recipe's max_caption_tokens.
        log.info("Caption saved for %s after %.1fs: %s", name, time.monotonic() - started, output_dir)
        return True

    try:
        return await with_retries(
            f"Caption {name}", _attempt, max_attempts=args.max_retries, delay_s=1.0, server=args.server,
        )
    except RuntimeError:
        log.warning("Failed to get caption for %s", name)
        return False


def _read_manifest_entries(input_files: list[Path]) -> list[tuple[str, str, dict | None]]:
    """Parse ``-i`` JSON/JSONL manifests into ``(name, image_ref, media)`` tuples.

    Each entry must have a ``vision_path`` (a local path or an ``http(s)``/``data``
    URL) and may carry an optional ``name`` and an optional ``media`` dict (the
    structured caption's media fields: resolution/aspect_ratio).  The
    ``media`` override lets remote-URL images — which ffprobe cannot read — still
    get accurate media fields.
    """
    items: list[tuple[str, str, dict | None]] = []
    for path in input_files:
        text = path.read_text()
        if path.suffix == ".jsonl":
            entries = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            entries = data if isinstance(data, list) else [data]
        for e in entries:
            vp = e.get("vision_path")
            name = e.get("name")
            if not vp:
                log.warning(f"Skipping entry with no vision_path: {name or '?'}")
                continue
            if Path(vp).suffix.lower() not in IMAGE_EXTENSIONS:
                log.warning(f"Skipping '{name or vp}': vision_path is not an image ({Path(vp).suffix})")
                continue
            items.append((name or Path(vp).stem, vp, e.get("media")))
    return items


def _collect_image_items(args: Args) -> list[tuple[str, str, dict | None]]:
    """Return ``(name, image_ref, media_override)`` items from the CLI arguments.

    ``image_ref`` is a local filesystem path or a remote URL (``http(s)``/``data``).
    """
    items: list[tuple[str, str, dict | None]] = []

    if args.input_files:
        items = _read_manifest_entries(args.input_files)
    elif args.image:
        if args.image.is_dir():
            for vp in sorted(args.image.iterdir(), key=lambda item: (item.name.casefold(), item.name)):
                if vp.is_file() and vp.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((vp.stem, str(vp), None))
        elif args.image.is_file():
            if args.image.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {args.image.suffix}")
            items.append((args.image.stem, str(args.image), None))
        else:
            raise FileNotFoundError(f"Image path does not exist: {args.image}")

    if not items:
        raise ValueError("No image inputs found. Provide --input_files (-if) or --image (-i).")
    names: set[str] = set()
    for name, _, _ in items:
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Image output name must be a filename, received {name!r}.")
        key = name.casefold()
        if key in names:
            raise ValueError(f"Ambiguous original image name: {name!r}; output filenames must be unique.")
        names.add(key)
    return items


def _pair_mask_items(args: Args, items: list[tuple[str, str, dict | None]]) -> list[str | None]:
    """Resolve all pairs before starting requests; reject missing or ambiguous names."""
    if args.mask is None:
        return [None] * len(items)
    if args.input_files or args.image is None:
        raise ValueError("--mask requires --image and cannot be used with --input-files.")
    if args.mask.is_dir():
        masks = sorted(
            path for path in args.mask.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif args.mask.is_file():
        if args.mask.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported mask format: {args.mask.suffix}")
        masks = [args.mask]
    else:
        raise FileNotFoundError(f"Mask path does not exist: {args.mask}")

    by_stem: dict[str, Path] = {}
    for mask in masks:
        key = mask.stem.casefold()
        if key in by_stem:
            raise ValueError(f"Ambiguous mask name '{mask.stem}': {by_stem[key]} and {mask}")
        by_stem[key] = mask

    refs: list[str | None] = []
    seen: set[str] = set()
    for _, image_ref, _ in items:
        stem = Path(image_ref).stem
        key = stem.casefold()
        if key in seen:
            raise ValueError(f"Ambiguous original image name: '{stem}'")
        seen.add(key)
        if key not in by_stem:
            raise ValueError(f"No matching mask for image '{image_ref}' (expected name '{stem}')")
        refs.append(str(by_stem[key]))
    return refs


async def caption_from_image(args: Args):
    if args.input_files and args.image:
        raise ValueError("Provide either --input_files or --image, not both.")
    if args.mask is not None and (args.input_files or args.image is None):
        raise ValueError("--mask requires --image and cannot be used with --input-files.")

    if args.prompt_template_path:
        prompt_template = args.prompt_template_path.read_text(encoding="utf-8")
    else:
        prompt_template = (_PACKAGE_DIR / "src/caption/image_captioner.txt").read_text(encoding="utf-8")

    items = _collect_image_items(args)
    mask_refs = _pair_mask_items(args, items)

    async with create_client(
        args.server,
        api_token="a84be36b570d14855696b9298eeba9b37a1e6bb0977be381ab284e3207ff648d",
        timeout_s=args.timeout,
    ) as client:
        if not args.model:
            args.model = await resolve_model(client, args.model)
            log.info(f"Using model: {args.model}")

        log.info(
            "Captioning %s images at %s with model=%s, max_workers=%s, timeout=%ss, max_tokens=%s",
            len(items), args.server, args.model, args.max_workers, args.timeout, args.max_tokens,
        )
        async def process(item: tuple[tuple[str, str, dict | None], str | None]) -> bool:
            (name, image_ref, media), mask_ref = item
            return await _process_single(
                args, client, name, image_ref, media, prompt_template, mask_ref=mask_ref,
            )

        results = await run_batch(
            list(zip(items, mask_refs)), process, max_workers=args.max_workers,
            desc="Captioning", unit="image",
        )
        log.info("%d/%d images were successfully captioned", sum(results), len(items))


def main():
    args = tyro.cli(Args, description=__doc__)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(caption_from_image(args))


if __name__ == "__main__":
    main()
