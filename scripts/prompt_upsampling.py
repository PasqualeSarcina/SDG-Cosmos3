# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prompt upsampling client for Cosmos3 generation/evaluation scripts.

This module has two layers:

1. Prompt builders that turn a terse user prompt into a model-specific chat
   request for text-to-image upsampling.
2. An asynchronous OpenAI-compatible client and CLI that send those requests to an
   external upsampler endpoint and write normalized JSON prompts to disk.

The code is intentionally endpoint-light: all prompt shape decisions live here,
while authentication, model choice, and endpoint URL are supplied by callers or
CLI arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(_PACKAGE_DIR))

from src.request.client import chat_completion, create_client, list_models, run_batch, with_retries
from src.request.response_errors import response_content
from src.validation.t2i_schema import t2i_template_to_json_schema

# In Cosmos3 training, we used `json.dumps(...)` for converting dict-structured caption objects to string.
# This JSON-formatted string was then used as text caption input to the tokenizer.
# This has the side effect of converting non-ASCII characters to their ASCII equivalents.
# Although this is not ideal for languages like Chinese, it's how the model was trained.
# For the prompt upsampling client, we therefore ensure that JSON output is ASCII-only by default.
# If this JSON_ENSURE_ASCII environment variable is set to 0, we use `json.dumps(..., ensure_ascii=False)`
# instead and characters like `中文` will be preserved and get tokenized instead of `\\u4e2d\\u6587`.
JSON_ENSURE_ASCII = bool(int(os.environ.get("JSON_ENSURE_ASCII", "1")))

log = logging.getLogger(__name__)

RESOLUTION_RATIO_DICT: dict[str, dict[str, dict[str, int]]] = {
    # "256": {
    #     "1,1": {"W": 256, "H": 256},
    #     "4,3": {"W": 320, "H": 256},
    #     "3,4": {"W": 256, "H": 320},
    #     "16,9": {"W": 320, "H": 192},
    #     "9,16": {"W": 192, "H": 320},
    # },
    # "480": {
    #     "1,1": {"W": 640, "H": 640},
    #     "4,3": {"W": 736, "H": 544},
    #     "3,4": {"W": 544, "H": 736},
    #     "16,9": {"W": 832, "H": 480},
    #     "9,16": {"W": 480, "H": 832},
    # },
    "720": {
        "1,1": {"W": 960, "H": 960},
        "4,3": {"W": 1104, "H": 832},
        "3,4": {"W": 832, "H": 1104},
        "16,9": {"W": 1280, "H": 720},
        "9,16": {"W": 720, "H": 1280},
    },
    # "768": {
    #     "1,1": {"W": 1024, "H": 1024},
    #     "4,3": {"W": 1184, "H": 880},
    #     "3,4": {"W": 880, "H": 1184},
    #     "16,9": {"W": 1360, "H": 768},
    #     "9,16": {"W": 768, "H": 1360},
    # },
}

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
PROMPTING_TEMPLATES_DIR = SRC_DIR / "upsampling"


def build_openai_message(
        prompt: str,
        *,
        additional_instruction: str,
        system_prompt: str,
        json_schema: dict[str, Any],
        user_prompt: str | None = None,
) -> dict[str, Any]:
    """Pack the chat messages and strict response schema into one request."""
    message_text = user_prompt or (
        f"INPUT SCENE DESCRIPTION:\n{prompt}\n\n"
        f"ADDITIONAL USER INSTRUCTIONS:\n{additional_instruction}"
    )
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": message_text}],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "t2i_caption", "strict": True, "schema": json_schema},
        },
    }


def _apply_t2i_output_parameters(data: dict[str, Any], *, resolution: str, aspect_ratio: str) -> dict[str, Any]:
    """Force canonical image metadata into an upsampled T2I JSON object."""
    if resolution not in RESOLUTION_RATIO_DICT:
        raise ValueError(f"Unsupported upsampler resolution {resolution!r}.")
    if aspect_ratio not in RESOLUTION_RATIO_DICT[resolution]:
        raise ValueError(f"Unsupported upsampler aspect_ratio {aspect_ratio!r} for resolution {resolution!r}.")
    resolution_pair = RESOLUTION_RATIO_DICT[resolution][aspect_ratio]
    data["resolution"] = {"H": resolution_pair["H"], "W": resolution_pair["W"]}
    data["aspect_ratio"] = aspect_ratio
    return data


@dataclass(slots=True)
class PromptUpsamplerConfig:
    """Connection and sampling settings for ``PromptUpsamplerClient``.

    ``endpoint_url`` may be a bare host, a ``/v1`` base URL, or a full
    ``/chat/completions`` URL; it is normalized by the client. Sampling
    parameters set to ``None`` are omitted from the request payload so the same
    client can work across gateways with different accepted knobs.
    """

    endpoint_url: str
    model: str | None
    api_token: str | None
    timeout_s: float
    max_tokens: int
    max_retries: int
    temperature: float | None
    top_p: float | None
    system_prompt_path: str | Path | None = None


class PromptUpsamplerClient:
    """Async prompt client with structured decoding and request retries."""

    def __init__(self, config: PromptUpsamplerConfig, *, client: AsyncOpenAI | None = None) -> None:
        self.config = config
        system_path = Path(config.system_prompt_path or PROMPTING_TEMPLATES_DIR / "t2i_system_upsampler_prompt.txt")
        json_schema_path = SRC_DIR / "schema" / "t2i_json_schema.json"
        self.system_prompt = system_path.read_text(encoding="utf-8-sig").strip()
        if not self.system_prompt:
            raise ValueError(f"System prompt file is empty: {system_path}")
        self.json_schema = t2i_template_to_json_schema(json.loads(json_schema_path.read_text(encoding="utf-8-sig")))
        self._client = client or create_client(
            config.endpoint_url, api_token=config.api_token, timeout_s=config.timeout_s,
        )
        self._owns_client = client is None
        self._model_lock = asyncio.Lock()

    async def __aenter__(self) -> PromptUpsamplerClient:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._owns_client:
            await self._client.close()

    async def list_models(self) -> list[str]:
        async def call(attempt: int) -> list[str]:
            return await list_models(self._client)

        return await with_retries(
            "List models", call, max_attempts=self.config.max_retries,
            delay_s=1.0, backoff=2.0, server=self.config.endpoint_url,
        )

    async def upsample(
            self,
            prompt: str,
            *,
            additional_instruction:str,
            resolution: str,
            aspect_ratio: str,
            user_prompt: str | None = None,
            debug_dir: str | Path | None = None,
            debug_name: str = "response",
    ) -> dict[str, Any]:
        """Upsample a text-to-image prompt using structured decoding."""
        request = build_openai_message(
            prompt,
            additional_instruction=additional_instruction,
            system_prompt=self.system_prompt,
            json_schema=self.json_schema,
            user_prompt=user_prompt,
        )
        debug_dir = Path(debug_dir) if debug_dir is not None else None

        async def _call(attempt: int) -> dict[str, Any]:
            response_path = debug_dir / f"{debug_name}.attempt-{attempt}.json" if debug_dir is not None else None
            data = await self._chat_completion(request, response_path=response_path)
            return _apply_t2i_output_parameters(data, resolution=resolution, aspect_ratio=aspect_ratio)

        return await with_retries(
            "Prompt upsampling", _call, max_attempts=self.config.max_retries,
            delay_s=1.0, backoff=2.0, server=self.config.endpoint_url,
        )

    async def _get_model(self) -> str:
        """Resolve the model name from config or endpoint discovery."""
        async with self._model_lock:
            if self.config.model:
                return self.config.model
            self.config.model = (await self.list_models())[0]
            return self.config.model

    async def _chat_completion(self, request: dict[str, Any], *, response_path: Path | None = None) -> dict[str, Any]:
        """Send messages and extract a JSON object without serializing it again."""
        model = await self._get_model()
        parameters: dict[str, Any] = {
            "model": model,
            **request,
            "max_completion_tokens": self.config.max_tokens,
        }
        for name in ("temperature", "top_p"):
            value = getattr(self.config, name)
            if value is not None:
                parameters[name] = value
        response = await chat_completion(self._client, response_path=response_path, **parameters)
        content = response_content(response, max_tokens=self.config.max_tokens, response_path=response_path)
        data = json.loads(content)
        subject_details: dict[str, str] = {}
        for entry in data["subject_details"]:
            key = entry["key"]
            if key in subject_details:
                raise ValueError(f"Duplicate subject_details key: {key!r}")
            subject_details[key] = entry["value"]
        data["subject_details"] = subject_details
        return data


def _read_prompts(path: str | Path) -> list[tuple[Path, str]]:
    """Read one JSON object per .json file, in filename order.

    Directory inputs include only immediate .json files. Validate every input
    before making requests.
    """
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json file: {source}")
        files = [source]
    elif source.is_dir():
        files = sorted(
            (item for item in source.iterdir() if item.is_file() and item.suffix.lower() == ".json"),
            key=lambda item: (item.name.casefold(), item.name),
        )
        if not files:
            raise ValueError(f"No .json files found in {source}.")
    else:
        raise ValueError(f"Input path does not exist: {source}")
    prompts = []
    for file in files:
        prompt = file.read_text(encoding="utf-8-sig")
        if not prompt.strip():
            raise ValueError(f"Prompt file is empty: {file}")
        try:
            data = json.loads(prompt)
        except ValueError as exc:
            raise ValueError(f"Invalid JSON in {file}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Input JSON must be an object: {file}")
        prompts.append((file, prompt))
    return prompts


def _optional_float(value: str) -> float | None:
    """Parse a float CLI argument that can also be disabled with ``none``."""
    if value.strip().lower() in {"none", "null"}:
        return None
    return float(value)


async def upsample_prompt_file(
        client: PromptUpsamplerClient,
        *,
        additional_instruction:str,
        input_path: str | Path,
        output_path: str | Path,
        resolution: str,
        aspect_ratio: str,
        max_workers: int,
        debug: bool,
) -> list[dict[str, Any]]:
    """Upsample a .json file or directory into structured .json captions.

    Only immediate .json files are read from directory inputs. Outputs retain
    the exact input filenames and contain the structured caption object directly.
    Raw debug responses share the output/debug directory, with one file per attempt.
    """
    prompts = _read_prompts(input_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    debug_dir = output / "debug" if debug else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    failures: list[Path] = []
    async def process(item: tuple[Path, str]) -> dict[str, Any] | None:
        source, prompt = item
        try:
            result = await client.upsample(
                prompt, additional_instruction=additional_instruction,
                resolution=resolution, aspect_ratio=aspect_ratio,
                **({"debug_dir": debug_dir, "debug_name": source.stem} if debug else {}),
            )
            (output / source.name).write_text(
                json.dumps(result, ensure_ascii=JSON_ENSURE_ASCII, indent=2) + "\n", encoding="utf-8",
            )
            log.debug("Saved %s", source.name)
            return result
        except Exception:
            failures.append(source)
            log.exception("Failed to upsample %s", source)
            return None

    results = await run_batch(
        prompts, process, max_workers=max_workers, desc="Upsampling", unit="prompt",
    )
    if failures:
        raise RuntimeError(
            f"Failed to upsample {len(failures)}/{len(prompts)} files: {', '.join(str(p) for p in failures)}")
    return [result for result in results if result is not None]


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the standalone CLI parser for batch prompt upsampling."""
    parser = argparse.ArgumentParser(description="Upsample .json files into Cosmos3 text-to-image JSON prompts.")
    parser.add_argument("--input", required=True,
                        help="Single .json file or directory containing .json files; subdirectories are ignored. Each file must contain one JSON object.")
    parser.add_argument("--output", required=True,
                        help="Output directory; each JSON keeps the exact input filename.")
    parser.add_argument(
        "--server",
        required=True,
        help="OpenAI-compatible endpoint URL.",
    )
    parser.add_argument(
        "--model",
        help="Model name. If omitted, use the first model returned by the endpoint.",
    )
    parser.add_argument(
        "--api-token",
        help="API token.",
    )
    parser.add_argument(
        "--system-prompt",
        help="UTF-8 text file for the system message. Defaults to src/upsampling/t2i_system_upsampler_prompt.txt.",
    )
    parser.add_argument(
        "--additional-instruction",
        type=str,
        default=""
    )
    parser.add_argument("--resolution", default="720", choices=sorted(RESOLUTION_RATIO_DICT), help="Resolution tier.")
    parser.add_argument("--aspect-ratio", default="16,9", help="Aspect ratio key, e.g. 16,9, 9,16, or 1,1.")
    parser.add_argument("--timeout-s", type=float, default=300.0, help="Request timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum response tokens.")
    parser.add_argument("--max-retries", type=int, default=10, help="Prompt upsampling retries per prompt.")
    parser.add_argument("--max-workers", "--max_workers", type=int, default=16,
                        help="Maximum parallel requests")
    parser.add_argument(
        "--temperature", type=_optional_float, default=None, help="Optional sampling temperature, or 'none'."
    )
    parser.add_argument("--top-p", type=_optional_float, default=None, help="Optional sampling top-p, or 'none'.")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging and save raw API responses to <output>/debug/<input-stem>.attempt-<attempt>.json (attempts start at 0).",
    )
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point used by ``python scripts/prompt_upsampling.py``."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1.")

    # Build the client after CLI validation so endpoint/model/token mistakes
    # fail before any input file is processed.
    config = PromptUpsamplerConfig(
        endpoint_url=args.server,
        model=args.model,
        api_token=args.api_token,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt_path=args.system_prompt,
    )
    async def run() -> list[dict[str, Any]]:
        async with PromptUpsamplerClient(config) as client:
            return await upsample_prompt_file(
                client,
                additional_instruction=args.additional_instruction,
                input_path=args.input,
                output_path=args.output,
                resolution=args.resolution,
                aspect_ratio=args.aspect_ratio,
                max_workers=args.max_workers,
                debug=args.debug,
            )

    results = asyncio.run(run())
    log.info("Wrote %d upsampled prompts to %s", len(results), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
