import argparse
import base64
import json
import logging
import sys
import time

from PIL import Image

from pathlib import Path

import asyncio
import openai

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

def _image_url(image_ref: str) -> str:
    """Embed a local image as a base64 data URL for the server."""
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
) -> list[dict]:
    """Build one multimodal request with an image and a text prompt."""
    content = [{"type": "image_url", "image_url": {"url": _image_url(image_ref)}}]
    content.append({"type": "text", "text": prompt_template})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


async def _process_single(
    args: argparse.Namespace,
    client: openai.AsyncOpenAI,
    name: str,
    image_ref: str,
    prompt_template: str,
) -> bool:
    assert args.model

    output_dir = args.output_dir
    output_path = output_dir / f"{name}.json"
    try:
        messages = _build_vlm_messages(image_ref, prompt_template)
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

        try:
            with Image.open(image_ref) as image:
                media = media_fields_from_metadata(image.size)
        except Exception as e:  # noqa: BLE001 - degrade gracefully, keep the caption
            log.warning(f"Cannot read image dimensions for {name}: {e}; writing caption_json without media fields")
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


def _collect_image_items(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return ``(name, image_path)`` items from a local image or directory."""
    items: list[tuple[str, str]] = []

    if args.image.is_dir():
        for vp in sorted(args.image.iterdir(), key=lambda item: (item.name.casefold(), item.name)):
            if vp.is_file() and vp.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((vp.stem, str(vp)))
    elif args.image.is_file():
        if args.image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {args.image.suffix}")
        items.append((args.image.stem, str(args.image)))
    else:
        raise FileNotFoundError(f"Image path does not exist: {args.image}")

    if not items:
        raise ValueError(f"No image inputs found in {args.image}.")
    names: set[str] = set()
    for name, _ in items:
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Image output name must be a filename, received {name!r}.")
        key = name.casefold()
        if key in names:
            raise ValueError(f"Ambiguous original image name: {name!r}; output filenames must be unique.")
        names.add(key)
    return items


async def caption_from_image(args: argparse.Namespace):
    if args.prompt_template_path:
        prompt_template = args.prompt_template_path.read_text(encoding="utf-8")
    else:
        prompt_template = (_PACKAGE_DIR / "src/caption/image_captioner.txt").read_text(encoding="utf-8")

    items = _collect_image_items(args)

    async with create_client(
        args.server,
        api_token=args.api_token,
        timeout_s=args.timeout,
    ) as client:
        if not args.model:
            args.model = await resolve_model(client, args.model)
            log.info(f"Using model: {args.model}")

        log.info(
            "Captioning %s images at %s with model=%s, max_workers=%s, timeout=%ss, max_tokens=%s",
            len(items), args.server, args.model, args.max_workers, args.timeout, args.max_tokens,
        )
        async def process(item: tuple[str, str]) -> bool:
            name, image_ref = item
            return await _process_single(
                args, client, name, image_ref, prompt_template,
            )

        results = await run_batch(
            items, process, max_workers=args.max_workers,
            desc="Captioning", unit="image",
        )
        log.info("%d/%d images were successfully captioned", sum(results), len(items))


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the standalone CLI parser for image captioning."""
    parser = argparse.ArgumentParser(description="Generate captions from a local image or directory of images.")
    parser.add_argument(
        "--image", "-i", type=Path, required=True,
        help="Single image file or directory of images; subdirectories are ignored.",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", "-o", type=Path, required=True,
        help="Output directory; image.jpg becomes image.json, with no per-image subdirectories.",
    )
    parser.add_argument(
        "--server", required=True, type=str,
        help="OpenAI-compatible VLM endpoint URL.",
    )
    parser.add_argument("--api-token", "--api_token", help="Optional API token for authentication with the server.")
    parser.add_argument("--model", help="Model name. If omitted, use the first model returned by the endpoint.")
    parser.add_argument(
        "--max-workers", "--max_workers", type=int, default=16,
        help="Maximum concurrent requests (default: 16).",
    )
    parser.add_argument(
        "--max-retries", "--max_retries", type=int, default=3,
        help="Maximum attempts per image, including the first request (default: 3).",
    )
    parser.add_argument(
        "--max-tokens", "--max_tokens", type=int, default=4096,
        help="Maximum output tokens per request (default: 4096).",
    )
    parser.add_argument(
        "--timeout", type=float, default=1200.0,
        help="Per-request timeout in seconds (default: 1200).",
    )
    parser.add_argument(
        "--prompt-template-path", "--prompt_template_path", type=Path,
        help="UTF-8 prompt template file. Defaults to src/caption/image_captioner.txt.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging and save complete API responses to <output-dir>/debug/.",
    )
    parser.add_argument(
        "--raw-content", "--raw_content", action="store_true",
        help="Save message.content verbatim to <name>.json, bypassing structured caption parsing.",
    )
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point used by ``python scripts/caption_from_image.py``."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    for option in ("max_workers", "max_retries", "max_tokens"):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be >= 1.")
    if not args.timeout > 0:
        parser.error("--timeout must be > 0.")
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(caption_from_image(args))
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
