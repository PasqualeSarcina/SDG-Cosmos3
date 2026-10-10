"""Shared asynchronous requests for OpenAI-compatible servers."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from openai import APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm

from src.request.response_errors import log_response_error

log = logging.getLogger(__name__)
T = TypeVar("T")
R = TypeVar("R")


def normalize_base_url(url: str) -> str:
    """Accept a host, a /v1 base URL, or a full chat-completions URL."""
    normalized = url.strip().rstrip("/")
    if not normalized:
        raise ValueError("server cannot be empty.")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    if normalized.endswith("/chat/completions"):
        normalized = normalized[:-len("/chat/completions")]
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def create_client(server: str, *, api_token: str | None, timeout_s: float) -> AsyncOpenAI:
    """Disable SDK retries: callers retry the complete request and validation."""
    return AsyncOpenAI(
        base_url=normalize_base_url(server),
        api_key=api_token or "EMPTY",
        timeout=timeout_s,
        max_retries=0,
    )


async def list_models(client: AsyncOpenAI) -> list[str]:
    response = await client.models.list()
    models = [item.id for item in response.data if isinstance(item.id, str) and item.id]
    if not models:
        raise ValueError("Model list response did not include any model ids.")
    return models


async def resolve_model(client: AsyncOpenAI, model: str | None) -> str:
    return model or (await list_models(client))[0]


def _save_response(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log.info("Raw API response saved: %s", path)


async def chat_completion(
    client: AsyncOpenAI, *, response_path: Path | None = None, **parameters: Any,
) -> ChatCompletion:
    """Save the exact HTTP body before parsing, including HTTP error bodies."""
    try:
        if response_path is None:
            response = await client.chat.completions.create(**parameters)
        else:
            async with client.chat.completions.with_streaming_response.create(**parameters) as raw:
                _save_response(response_path, await raw.text())
                response = await raw.parse()
    except APIStatusError as exc:
        if response_path is not None:
            _save_response(response_path, exc.response.text)
        raise
    return response


async def with_retries(
    operation: str,
    call: Callable[[int], Awaitable[R]],
    *,
    max_attempts: int,
    delay_s: float,
    backoff: float = 1.0,
    server: str | None = None,
) -> R:
    """Retry a complete operation, including its output validation."""
    if max_attempts < 1:
        raise ValueError("max_retries must be >= 1.")
    for attempt in range(max_attempts):
        started = time.monotonic()
        try:
            return await call(attempt)
        except Exception as exc:
            log_response_error(
                operation, exc, attempt=attempt, max_attempts=max_attempts,
                elapsed_s=time.monotonic() - started, server=server,
            )
            if attempt + 1 == max_attempts:
                raise RuntimeError(
                    f"{operation} failed after {max_attempts} attempts: {exc}"
                ) from exc
            await asyncio.sleep(delay_s * backoff ** attempt)
    raise AssertionError("Unreachable")


async def run_batch(
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    max_workers: int,
    desc: str,
    unit: str,
) -> list[R]:
    """Run workers under a semaphore; report progress and retain input order."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1.")
    semaphore = asyncio.Semaphore(max_workers)

    async def run(index: int, item: T) -> tuple[int, R]:
        async with semaphore:
            return index, await worker(item)

    tasks = [asyncio.create_task(run(index, item)) for index, item in enumerate(items)]
    results: dict[int, R] = {}
    try:
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc, unit=unit):
            index, result = await task
            results[index] = result
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return [results[index] for index in range(len(items))]

