import base64
import json
import logging
import sys
import time

from PIL import Image

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import asyncio
import openai
import tyro

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    # Direct execution puts scripts/ on sys.path, not the project root.
    sys.path.insert(0, str(_PACKAGE_DIR))

from src.caption.structured_caption import extract_xml_tag, parse_structured_caption, media_fields_from_metadata, \
    assemble_caption_json

from src.request.client import chat_completion, create_client, resolve_model, run_batch, with_retries
from src.request.response_errors import response_content
from src.request.response_parsing import parse_json_response
from src.validation.t2i_schema import t2i_template_to_json_schema

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
log = logging.getLogger(__name__)


@dataclass
class Args:
    """Generate captions from a local image or directory of images."""

    image: Annotated[Path, tyro.conf.arg(aliases=["-i"])]
    """Single image file or directory of images; subdirectories are ignored."""
    output_dir: Annotated[Path, tyro.conf.arg(aliases=["-o", "--output_dir"])]
    """Output directory; image.jpg becomes image.json."""
    server: str
    """OpenAI-compatible VLM endpoint URL."""
    api_token: Annotated[str | None, tyro.conf.arg(aliases=["--api_token"])] = None
    """Optional API token for authentication with the server."""
    model: str | None = None
    """Model name. If omitted, use the first model returned by the endpoint."""
    max_workers: Annotated[int, tyro.conf.arg(aliases=["--max_workers"])] = 16
    """Maximum concurrent requests."""
    max_retries: Annotated[int, tyro.conf.arg(aliases=["--max_retries"])] = 3
    """Maximum attempts per image, including the first request."""
    max_tokens: Annotated[int, tyro.conf.arg(aliases=["--max_tokens"])] = 8192
    """Maximum output tokens per request."""
    timeout: float = 1200.0
    """Per-request timeout in seconds."""
    prompt_template_path: Annotated[Path | None, tyro.conf.arg(aliases=["--prompt_template_path"])] = None
    """UTF-8 prompt template file. Defaults to src/caption/image_captioner.txt."""
    debug: bool = False
    """Enable debug logging and save complete API responses to <output-dir>/debug/."""
    enforce_cosmos_format: Annotated[bool, tyro.conf.arg(aliases=["--enforce_cosmos_format"])] = True
    """Validate the Cosmos caption format; disable to require description_json_schema JSON."""
    image_items: list[tuple[str, str]] = field(init=False, repr=False)
    prompt_template: str = field(init=False, repr=False)
    response_format: dict | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for option in ("max_workers", "max_retries", "max_tokens"):
            if getattr(self, option) < 1:
                raise ValueError(f"--{option.replace('_', '-')} must be >= 1.")
        if not self.timeout > 0:
            raise ValueError("--timeout must be > 0.")
        if not self.server.strip():
            raise ValueError("--server cannot be empty.")
        self.image = Path(self.image)
        self.output_dir = Path(self.output_dir)
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"Output path must be a directory: {self.output_dir}")

        items: list[tuple[str, str]] = []
        if self.image.is_dir():
            for path in sorted(self.image.iterdir(), key=lambda item: (item.name.casefold(), item.name)):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((path.stem, str(path)))
        elif self.image.is_file():
            if self.image.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {self.image.suffix}")
            items.append((self.image.stem, str(self.image)))
        else:
            raise FileNotFoundError(f"Image path does not exist: {self.image}")
        if not items:
            raise ValueError(f"No image inputs found in {self.image}.")
        names: set[str] = set()
        for name, _ in items:
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError(f"Image output name must be a filename, received {name!r}.")
            key = name.casefold()
            if key in names:
                raise ValueError(f"Ambiguous original image name: {name!r}; output filenames must be unique.")
            names.add(key)
        self.image_items = items

        template_path = Path(self.prompt_template_path or _PACKAGE_DIR / "src/caption/image_captioner.txt")
        self.prompt_template = template_path.read_text(encoding="utf-8")
        if not self.prompt_template.strip():
            raise ValueError(f"Prompt template file is empty: {template_path}")
        if not self.enforce_cosmos_format:
            schema_path = _PACKAGE_DIR / "src/schema/description_json_schema.json"
            template = json.loads(schema_path.read_text(encoding="utf-8-sig"))
            self.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "image_description",
                    "strict": True,
                    "schema": t2i_template_to_json_schema(template),
                },
            }


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
    args: Args,
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
        log.debug("[%s/%s] Sending image %s", i_retry + 1, args.max_retries, name)
        response_path = output_dir / "debug" / f"{name}.attempt-{i_retry}.json" if args.debug else None
        response = await chat_completion(
            client, response_path=response_path,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            **({"response_format": args.response_format} if args.response_format is not None else {}),
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )
        text = response_content(response, max_tokens=args.max_tokens, response_path=response_path)
        comprehensive_t2i_caption = None
        scene_draft = None
        if args.enforce_cosmos_format:
            comprehensive_t2i_caption = extract_xml_tag(text, "comprehensive_t2i_caption")
            scene_draft = parse_structured_caption(text)
            if comprehensive_t2i_caption is None and scene_draft is not None:
                if extract_xml_tag(text, "scene_draft") is None:
                    comprehensive_t2i_caption = scene_draft.get("comprehensive_t2i_caption")
            if not isinstance(comprehensive_t2i_caption, str) or not comprehensive_t2i_caption.strip():
                raise ValueError("Missing or empty <comprehensive_t2i_caption> tag or JSON field")
            if scene_draft is None:
                raise ValueError("Invalid scene_draft JSON")

        if not args.enforce_cosmos_format:
            data = parse_json_response(text)
            schema = args.response_format["json_schema"]["schema"]
            if not isinstance(data, dict) or set(data) != set(schema["required"]):
                raise ValueError("Caption response must contain exactly the description schema fields.")
            if any(not isinstance(value, str) for value in data.values()):
                raise ValueError("Description schema fields must be strings.")
            output_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text, encoding="utf-8")
            log.debug("Caption saved for %s after %.1fs: %s", name, time.monotonic() - started, output_dir)
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
        log.debug("Caption saved for %s after %.1fs: %s", name, time.monotonic() - started, output_dir)
        return True

    try:
        return await with_retries(
            f"Caption {name}", _attempt, max_attempts=args.max_retries, delay_s=1.0, server=args.server,
        )
    except RuntimeError:
        log.warning("Failed to get caption for %s", name)
        return False


async def caption_from_image(args: Args):
    prompt_template = args.prompt_template

    items = args.image_items

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


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point used by ``python scripts/caption_from_image.py``."""
    args = tyro.cli(Args, args=argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for logger_name in ("httpx", "httpx2"):
        logging.getLogger(logger_name).setLevel(logging.DEBUG if args.debug else logging.WARNING)
    asyncio.run(caption_from_image(args))
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
