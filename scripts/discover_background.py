"""Discover background place categories from Qdrant and save only the final list."""

import asyncio
import json
import logging
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import tyro
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from src.request.client import chat_completion, create_client, resolve_model, run_batch, with_retries
from src.request.response_errors import EmptyResponseError, response_content
from src.request.response_parsing import parse_json_response

log = logging.getLogger(__name__)
OUTPUT_NAME = "background_categories.json"
CONSOLIDATION_BATCH_SIZE = 128
CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "background_categories": {"type": "array", "items": {"type": "string", "minLength": 1}},
    },
    "required": ["background_categories"],
    "additionalProperties": False,
}


@dataclass
class Args:
    """Discover background place categories from Qdrant and save only the final list."""

    output: Path
    """Directory for the final background_categories.json."""
    server: str
    """OpenAI-compatible LLM endpoint; the first served model is selected automatically."""
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    """Qdrant URL; defaults to QDRANT_URL or http://localhost:6333."""
    collection: str = "description"
    """Qdrant collection containing background payloads."""
    api_token: Annotated[str | None, tyro.conf.arg(aliases=["--api_token"])] = None
    """Optional API token for the LLM endpoint."""
    batch_size: int = 32
    """Maximum backgrounds per Qdrant page and extraction request."""
    max_workers: Annotated[int, tyro.conf.arg(aliases=["--max_workers"])] = 16
    """Maximum concurrent extraction or consolidation requests."""
    max_tokens: int = 65536
    """Output budget per request, including thinking."""
    max_retries: int = 3
    """Maximum attempts per request, including the first."""
    timeout: float = 3600.0
    """LLM request timeout in seconds."""
    reasoning_effort: Literal["low", "medium", "xhigh"] = "medium"
    """Thinking depth used for extraction and consolidation."""
    consolidation_reasoning_effort: Literal["disabled", "low", "medium", "xhigh"] | None = None
    """Override thinking for consolidation only; disabled requests a direct JSON answer."""
    debug: bool = False
    """Save raw LLM responses under output/debug/ for diagnosis."""

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_workers", "max_tokens", "max_retries"):
            if getattr(self, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 1.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("--timeout must be positive and finite.")
        for name in ("server", "url", "collection"):
            if not getattr(self, name).strip():
                raise ValueError(f"--{name} cannot be empty.")
        if self.reasoning_effort not in ("low", "medium", "xhigh"):
            raise ValueError("--reasoning-effort must be low, medium or xhigh.")
        if self.consolidation_reasoning_effort not in (None, "disabled", "low", "medium", "xhigh"):
            raise ValueError("--consolidation-reasoning-effort must be disabled, low, medium or xhigh.")
        self.output = Path(self.output).resolve()
        if self.output.exists() and not self.output.is_dir():
            raise ValueError("--output must be a directory.")


def iter_background_batches(client, collection, batch_size=32) -> Iterator[list[str]]:
    """Read and validate one Qdrant page at a time, without requesting vectors."""
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    found = False
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection, limit=batch_size, offset=offset,
            with_payload=["background"], with_vectors=False,
        )
        backgrounds = []
        for point in points:
            value = (point.payload or {}).get("background")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Point {point.id}: background must be a nonempty string.")
            backgrounds.append(value)
        if next_offset is not None and next_offset == offset:
            raise ValueError("Qdrant scroll did not advance to the next page.")
        if backgrounds:
            found = True
            yield backgrounds
        if next_offset is None:
            break
        offset = next_offset
    if not found:
        raise ValueError(f"Collection {collection!r} has no backgrounds to analyze.")


def load_backgrounds(client, collection, batch_size=256) -> list[str]:
    """Materialize backgrounds for callers that explicitly need the full list."""
    return [text for batch in iter_background_batches(client, collection, batch_size) for text in batch]


def unique_categories(values):
    result, seen = [], set()
    for value in values:
        name = " ".join(value.split())
        if name.casefold() not in seen:
            seen.add(name.casefold())
            result.append(name)
    return result


def validate_categories(data, *, require_nonempty=False):
    if not isinstance(data, dict) or set(data) != {"background_categories"}:
        raise ValueError("Expected exactly one field: background_categories.")
    values = data["background_categories"]
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("background_categories must be an array of nonempty strings.")
    if require_nonempty and not values:
        raise ValueError("Consolidation cannot discard every supplied category.")
    return unique_categories(values)


async def request_categories(args, client, model, prompt, input_data, *, operation, consolidation=False):
    """Retry validated requests in memory, saving raw responses only with debug."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))},
    ]
    schema = CATEGORY_SCHEMA
    if consolidation:
        schema = {**CATEGORY_SCHEMA, "properties": {
            "background_categories": {**CATEGORY_SCHEMA["properties"]["background_categories"], "minItems": 1},
        }}
    effort = (args.consolidation_reasoning_effort if consolidation else None) or args.reasoning_effort
    enable_thinking = effort != "disabled"
    parameters = dict(
        model=model, messages=messages, max_tokens=args.max_tokens,
        temperature=1.0 if enable_thinking else 0.7,
        top_p=0.95 if enable_thinking else 0.8,
        presence_penalty=0.0 if enable_thinking else 1.5,
        response_format={"type": "json_schema", "json_schema": {
            "name": "background_categories", "strict": True, "schema": schema,
        }},
        extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    if enable_thinking:
        parameters["reasoning_effort"] = effort

    async def attempt(index):
        response_path = None
        if args.debug:
            stem = re.sub(r"[^a-z0-9]+", "-", operation.casefold()).strip("-")
            response_path = args.output / "debug" / f"{stem}.attempt-{index + 1}.json"
        response = await chat_completion(
            client, response_path=response_path, **parameters,
        )
        try:
            text = response_content(response, max_tokens=args.max_tokens, response_path=response_path)
            text = re.sub(r"\A\s*<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
            if not text.strip():
                raise EmptyResponseError(
                    f"Chat completion contains only thinking, with no final answer "
                    f"(max_tokens={args.max_tokens}, usage={response.usage!r}, "
                    f"raw response={response_path or 'not saved (enable --debug)'})."
                )
        except EmptyResponseError as exc:
            hint = "Inspect the server reasoning parser/chat template and the raw response (--debug)."
            if consolidation and enable_thinking:
                hint += " To request consolidation without thinking, use --consolidation-reasoning-effort disabled."
            raise EmptyResponseError(f"{exc} {hint}") from exc
        return validate_categories(parse_json_response(text), require_nonempty=consolidation)

    return await with_retries(operation, attempt, max_attempts=args.max_retries,
                              delay_s=1.0, server=args.server)


async def consolidate_categories(args, client, model, prompt, categories):
    """Consolidate small vocabularies directly, larger ones with an in-memory tree."""
    categories = unique_categories(categories)
    if not categories:
        return []
    chunks = [categories[start:start + CONSOLIDATION_BATCH_SIZE]
              for start in range(0, len(categories), CONSOLIDATION_BATCH_SIZE)]

    async def normalize(index):
        return await request_categories(
            args, client, model, prompt, {"background_categories": chunks[index]},
            operation=f"Normalize background categories {index + 1}/{len(chunks)}", consolidation=True,
        )

    nodes = await run_batch(list(range(len(chunks))), normalize, max_workers=args.max_workers,
                            desc="Consolidating backgrounds", unit="batch")
    level = 1
    while len(nodes) > 1:
        current = nodes

        async def merge(start):
            values = unique_categories(current[start] + current[start + 1])
            return await request_categories(
                args, client, model, prompt, {"background_categories": values},
                operation=f"Merge background categories round {level} pair {start // 2 + 1}",
                consolidation=True,
            )

        nodes = await run_batch(list(range(0, len(current) - 1, 2)), merge,
                                max_workers=args.max_workers,
                                desc=f"Consolidating backgrounds round {level}", unit="pair")
        if len(current) % 2:
            nodes.append(current[-1])
        level += 1
    return nodes[0]


def save_categories(path, categories):
    """Publish once, after every request succeeds, preserving any previous result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".backgrounds-", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(json.dumps(categories, ensure_ascii=False, indent=2) + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def discover_background(args: Args, system_prompt, backgrounds: list[str]):
    """Analyze an existing list using the same bounded pipeline as the CLI."""
    if not backgrounds:
        raise ValueError("No backgrounds to analyze.")
    if any(not isinstance(background, str) or not background.strip() for background in backgrounds):
        raise ValueError("Every background must be a nonempty string.")
    batches = (backgrounds[start:start + args.batch_size]
               for start in range(0, len(backgrounds), args.batch_size))
    categories, _ = await discover_background_batches(args, system_prompt, batches)
    return categories


async def discover_background_batches(args: Args, system_prompt, batches: Iterator[list[str]]):
    """Read pages on demand with at most max_workers extraction requests in flight."""
    consolidation_prompt = (ROOT / "src/templates/consolidate_background.txt").read_text(encoding="utf-8")
    batch = await asyncio.to_thread(next, batches, None)
    if batch is None:
        raise ValueError("No backgrounds to analyze.")
    count = 0
    async with create_client(args.server, api_token=args.api_token, timeout_s=args.timeout) as client:
        model = await resolve_model(client, None)
        pending = {}
        results = {}
        index = 0
        with tqdm(desc="Discovering backgrounds", unit="batch") as progress:
            async def collect():
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    results[pending[task]] = task.result()
                    del pending[task]
                    progress.set_postfix(backgrounds=count, pending=len(pending), refresh=False)
                    progress.update(1)

            try:
                while batch is not None:
                    count += len(batch)
                    pending[asyncio.create_task(request_categories(
                        args, client, model, system_prompt, batch,
                        operation=f"Discover background batch {index + 1}",
                    ))] = index
                    index += 1
                    progress.set_postfix(backgrounds=count, pending=len(pending), refresh=False)
                    if len(pending) >= args.max_workers:
                        await collect()
                    # The synchronous Qdrant client must not block LLM requests.
                    batch = await asyncio.to_thread(next, batches, None)
                progress.total = index
                while pending:
                    await collect()
            finally:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        categories = unique_categories(name for batch_index in range(index) for name in results[batch_index])
        categories = await consolidate_categories(args, client, model, consolidation_prompt, categories)
    categories = sorted(categories, key=str.casefold)
    output_path = args.output / OUTPUT_NAME
    save_categories(output_path, categories)
    return categories, count


def cli_main(argv: list[str] | None = None) -> int:
    args = tyro.cli(Args, args=argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    for name in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from qdrant_client import QdrantClient

        prompt = (ROOT / "src/templates/discover_background.txt").read_text(encoding="utf-8")
        with logging_redirect_tqdm(), closing(
            QdrantClient(url=args.url, api_key=os.getenv("QDRANT_API_KEY"), timeout=120)
        ) as client:
            batches = iter_background_batches(client, args.collection, batch_size=args.batch_size)
            categories, count = asyncio.run(discover_background_batches(args, prompt, batches))
    except Exception as exc:
        log.error("Background discovery aborted: %s", exc)
        return 1
    print(json.dumps({"backgrounds": count, "categories": len(categories),
                      "output": str(args.output / OUTPUT_NAME)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
