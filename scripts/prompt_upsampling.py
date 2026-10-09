# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prompt upsampling client for Cosmos3 generation/evaluation scripts.

This module has two layers:

1. Prompt builders that turn a terse user prompt into a model-specific chat
   request for text-to-image upsampling.
2. A small OpenAI-compatible HTTP client and CLI that send those requests to an
   external upsampler endpoint and write normalized JSON prompts to disk.

The code is intentionally endpoint-light: all prompt shape decisions live here,
while authentication, model choice, and endpoint URL are supplied by callers or
environment variables.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from threading import Lock, local
from typing import Any

import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# In Cosmos3 training, we used `json.dumps(...)` for converting dict-structured caption objects to string.
# This JSON-formatted string was then used as text caption input to the tokenizer.
# This has the side effect of converting non-ASCII characters to their ASCII equivalents.
# Although this is not ideal for languages like Chinese, it's how the model was trained.
# For the prompt upsampling client, we therefore ensure that JSON output is ASCII-only by default.
# If this JSON_ENSURE_ASCII environment variable is set to 0, we use `json.dumps(..., ensure_ascii=False)`
# instead and characters like `中文` will be preserved and get tokenized instead of `\\u4e2d\\u6587`.
JSON_ENSURE_ASCII = bool(int(os.environ.get("JSON_ENSURE_ASCII", "1")))

SYSTEM_MESSAGE: dict[str, Any] = {
    "role": "system",
    "content": [{"type": "text", "text": "You are a helpful assistant."}],
}
DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
log = logging.getLogger(__name__)

# Prompt templates are stored outside this Python file so teams can iterate on
# prompt wording/schema contracts without editing the request/CLI plumbing.
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


@lru_cache(maxsize=None)
def _load_prompting_template(filename: str) -> str:
    """Load a built-in template once and strip the trailing newline.

    ``Template.substitute`` is used later, so template files should use
    ``$json_template``, ``$nl_description``, and ``$resolution_ratio_dict``
    placeholders rather than Python ``str.format`` braces.
    """
    return (PROMPTING_TEMPLATES_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


T2I_JSON_TEMPLATE = (SRC_DIR / "schema" / "t2i_json_schema.json").read_text(encoding="utf-8").rstrip("\n")
T2I_PROMPT_TEMPLATE = Template(_load_prompting_template("t2i_prompt.txt"))


def _resolution_ratio_dict_text() -> str:
    """Return the valid output resolution table rendered for prompt injection."""
    resolution_ratio_dict = {
        resolution: {aspect_ratio: {"H": size["H"], "W": size["W"]} for aspect_ratio, size in aspect_ratio_dict.items()}
        for resolution, aspect_ratio_dict in RESOLUTION_RATIO_DICT.items()
    }
    return json.dumps(resolution_ratio_dict, indent=2)


def build_nl_description(
        prompt: str,
        *,
        resolution: str,
        aspect_ratio: str,
) -> str:
    """Append literal output parameters to the user prompt.

    The external upsampler receives one text field, so we make generation
    metadata explicit in prose. The templates instruct the model to copy these
    values back into the JSON output.
    """
    params = [f"resolution {resolution}", f"aspect_ratio {aspect_ratio}"]
    return f"{prompt.strip()}\n\nOutput parameters: {', '.join(params)}."


def build_t2i_prompt_text(prompt: str, *, additional_instruction: str, resolution: str, aspect_ratio: str) -> str:
    """Render the complete text-to-image upsampler prompt."""
    nl_description = build_nl_description(prompt, resolution=resolution, aspect_ratio=aspect_ratio)
    return T2I_PROMPT_TEMPLATE.substitute(
        json_template=T2I_JSON_TEMPLATE,
        nl_description=nl_description,
        additional_instruction=additional_instruction.strip(),
        resolution_ratio_dict=_resolution_ratio_dict_text(),
    )


def build_t2i_messages(
        prompt: str,
        *,
        additional_instruction: str,
        resolution: str,
        aspect_ratio: str,
        user_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible chat messages for text-to-image upsampling."""
    message_text = user_prompt or build_t2i_prompt_text(prompt, additional_instruction=additional_instruction,
                                                        resolution=resolution, aspect_ratio=aspect_ratio)
    message_text += (
        "\n\nOutput format for this request: return only a bare JSON object matching the schema. "
        "Do not wrap it in Markdown code fences or add any commentary."
    )
    return [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [{"type": "text", "text": message_text}],
        },
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from a raw model response.

    Upsamplers may return either a bare JSON object or a fenced `````json```
    block. This helper normalizes both forms and rejects non-object payloads.
    """
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Upsampler response JSON must be an object.")
    return parsed


def extract_json_object_text(text: str) -> str:
    """Extract and normalize a JSON object from a raw model response."""
    parsed = extract_json_object(text)
    return json.dumps(parsed, ensure_ascii=JSON_ENSURE_ASCII)


def _compact_json_object(data: dict[str, Any]) -> str:
    """Serialize a JSON object in the compact format expected by inference."""
    return json.dumps(data, ensure_ascii=JSON_ENSURE_ASCII)


def validate_t2i_caption(data: dict[str, Any]) -> dict[str, Any]:
    """Require the keys and value types described by the active JSON template.

    The template is an example object with prose field descriptions, not a
    formal JSON Schema. subject_details intentionally allows arbitrary keys.
    """

    def check(value: Any, template: Any, path: str) -> None:
        if isinstance(template, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object.")
            if path == "caption.subject_details":
                return
            missing = template.keys() - value.keys()
            extra = value.keys() - template.keys()
            if missing or extra:
                raise ValueError(f"{path}: missing keys={sorted(missing)}, unexpected keys={sorted(extra)}")
            for key, field_template in template.items():
                check(value[key], field_template, f"{path}.{key}")
        elif isinstance(template, list):
            if not isinstance(value, list):
                raise ValueError(f"{path} must be an array.")
            for index, item in enumerate(value):
                if template:
                    check(item, template[0], f"{path}[{index}]")
        elif path in ("caption.resolution.H", "caption.resolution.W") or (
                isinstance(template, str) and template.startswith("int;")
        ):
            minimum = 1 if path.startswith("caption.resolution.") else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{path} must be an integer >= {minimum}.")
        elif not isinstance(value, str):
            raise ValueError(f"{path} must be a string.")

    check(data, json.loads(T2I_JSON_TEMPLATE), "caption")
    if not data.get("comprehensive_t2i_caption", "").strip():
        raise ValueError("comprehensive_t2i_caption must not be empty.")
    return data


def build_t2i_json_schema() -> dict[str, Any]:
    """Convert the active caption example into a server-side JSON Schema.

    Match the local validator, including arbitrary subject_details keys.
    This requires a serving backend that supports this JSON Schema subset.
    """

    def build(template: Any, path: str) -> dict[str, Any]:
        if isinstance(template, dict):
            if path == "caption.subject_details":
                return {"type": "object", "additionalProperties": True}
            return {
                "type": "object",
                "properties": {key: build(value, f"{path}.{key}") for key, value in template.items()},
                "required": list(template),
                "additionalProperties": False,
            }
        if isinstance(template, list):
            return {"type": "array", "items": build(template[0], f"{path}[]") if template else {}}
        if path in ("caption.resolution.H", "caption.resolution.W") or (
                isinstance(template, str) and template.startswith("int;")
        ):
            schema = {"type": "integer", "minimum": 1 if path.startswith("caption.resolution.") else 0}
        else:
            schema = {"type": "string"}
        if isinstance(template, str):
            schema["description"] = template
        if path == "caption.comprehensive_t2i_caption":
            schema["minLength"] = 1
        return schema

    return build(json.loads(T2I_JSON_TEMPLATE), "caption")


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
    model: str | None = None
    api_token: str | None = None
    timeout_s: float = 1200.0
    max_tokens: int = 8192
    max_retries: int = 5
    retry_base_delay_s: float = 1.0
    temperature: float | None = 0.7
    top_p: float | None = 0.8
    top_k: int | None = 20
    min_p: float | None = None
    enable_thinking: bool = False
    connection_max_retries: int = 2
    connection_pool_size: int = 4


class PromptUpsamplerClient:
    """Small OpenAI-compatible chat-completions client with explicit retries.

    The client deliberately avoids depending on the OpenAI SDK so it can be
    used in lightweight eval scripts.
    """

    def __init__(
            self,
            config: PromptUpsamplerConfig,
            *,
            session: requests.Session | None = None,
            sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._base_url = _normalize_openai_base_url(config.endpoint_url)
        self._session = session
        self._thread_state = local()
        self._model_lock = Lock()
        self._sleep = sleep

    def list_models(self) -> list[str]:
        """Fetch model ids from an OpenAI-compatible ``/models`` endpoint."""
        payload = self._with_retries("list models", lambda: self._request_json("GET", f"{self._base_url}/models"))
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Model list response missing 'data' list.")
        models: list[str] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(item["id"])
        if not models:
            raise ValueError("Model list response did not include any model ids.")
        return models

    def upsample_t2i(
            self,
            prompt: str,
            *,
            additional_instruction:str,
            resolution: str,
            aspect_ratio: str,
            user_prompt: str | None = None,
            debug_dir: str | Path | None = None,
    ) -> str:
        """Upsample a text-to-image prompt and return a compact JSON string."""
        messages = build_t2i_messages(
            prompt,
            additional_instruction=additional_instruction,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            user_prompt=user_prompt,
        )
        return self._upsample_messages_with_parameters(
            messages,
            lambda data: _apply_t2i_output_parameters(data, resolution=resolution, aspect_ratio=aspect_ratio),
            debug_dir=Path(debug_dir) if debug_dir is not None else None,
        )

    def upsample_messages(self, messages: list[dict[str, Any]]) -> str:
        """Upsample a pre-built chat message list without injecting metadata."""

        def _call() -> str:
            content = self._chat_completion(messages)
            return extract_json_object_text(content)

        return self._with_retries("upsample prompt", _call)

    def _upsample_messages_with_parameters(
            self,
            messages: list[dict[str, Any]],
            apply_parameters: Callable[[dict[str, Any]], dict[str, Any]],
            *,
            debug_dir: Path | None = None,
    ) -> str:
        """Call the model, parse the JSON object, and pin output parameters.

        Pinning metadata after the model response makes the output robust to
        small formatting/copy mistakes by the LLM and keeps inference metadata
        exactly aligned with CLI args.
        """

        attempt = 0

        def _call() -> str:
            nonlocal attempt
            response_path = debug_dir / str(attempt) / "response.json" if debug_dir is not None else None
            attempt += 1
            content = self._chat_completion(
                messages, **({"response_path": response_path} if response_path is not None else {}),
            )
            data = apply_parameters(extract_json_object(content))
            validate_t2i_caption(data)
            return _compact_json_object(data)

        return self._with_retries("upsample prompt", _call)

    def _get_model(self) -> str:
        """Resolve the model name from config, env, or endpoint discovery."""
        with self._model_lock:
            if self.config.model:
                return self.config.model
            env_model = os.environ.get("PROMPT_UPSAMPLER_MODEL")
            if env_model:
                self.config.model = env_model
                return env_model
            self.config.model = self.list_models()[0]
            return self.config.model

    def _chat_completion(self, messages: list[dict[str, Any]], *, response_path: Path | None = None) -> str:
        """Send messages to the configured endpoint and return assistant text."""
        model = self._get_model()
        log.debug(f"[prompt-upsampling] _chat_completion: model={model}, base_url={self._base_url}")

        # Keep provider-specific sampling params optional. Several gateways
        # reject unsupported keys instead of ignoring them.
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "t2i_caption",
                    "strict": True,
                    "schema": build_t2i_json_schema(),
                },
            },
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            payload["top_k"] = self.config.top_k
        if self.config.min_p is not None:
            payload["min_p"] = self.config.min_p
        response = self._request_json(
            "POST", f"{self._base_url}/chat/completions", payload=payload,
            **({"response_path": response_path} if response_path is not None else {}),
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Chat completion response missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("Chat completion choice must be an object.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Chat completion choice missing message.")
        finish_reason = first_choice.get("finish_reason")
        reasoning_fields = {
            field: len(message[field]) for field in ("reasoning", "reasoning_content")
            if isinstance(message.get(field), str)
        }
        details = (
            f"finish_reason={finish_reason!r}, max_tokens={self.config.max_tokens}, "
            f"usage={response.get('usage')!r}, reasoning_chars={reasoning_fields}"
        )
        if finish_reason == "length":
            raise ValueError(
                f"Chat completion truncated ({details}). Output token budget exhausted; "
                "disable thinking or increase --max-tokens within the server context limit."
            )
        if finish_reason not in (None, "stop"):
            raise ValueError(f"Chat completion did not finish normally ({details}).")
        if message.get("refusal"):
            raise ValueError(f"Chat completion refused: {message['refusal']} ({details}).")
        try:
            return _message_content_to_text(message.get("content"))
        except ValueError as exc:
            # Some server reasoning parsers place the final JSON in this field.
            # Accept only a complete caption matching the template, never prose.
            for field in reasoning_fields:
                try:
                    data = validate_t2i_caption(extract_json_object(message[field]))
                except (ValueError, TypeError):
                    continue
                log.warning("message.content is empty; recovered caption JSON from message.%s", field)
                return _compact_json_object(data)
            raise ValueError(
                f"Chat completion message.content is empty or unsupported ({details}; "
                f"content_type={type(message.get('content')).__name__}). "
                "No complete caption JSON found in reasoning fields; check the server chat/reasoning parser."
            ) from exc

    def _request_json(
            self, method: str, url: str, payload: dict[str, Any] | None = None,
            *, response_path: Path | None = None,
    ) -> dict[str, Any]:
        """Issue one HTTP request and parse a JSON object response."""
        headers = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
        if payload is not None:
            headers["Content-Type"] = "application/json"

        token = self.config.api_token
        if token:
            headers["Authorization"] = f"Bearer {token}"

        log.debug(
            f"[prompt-upsampling] _request_json: {method} {url} token={'***' + token[-4:] if token else '(none)'}"
        )
        try:
            # requests.Session carries mutable state: keep one per worker.
            session = self._session
            if session is None:
                if not hasattr(self._thread_state, "session"):
                    self._thread_state.session = _make_session(self.config)
                session = self._thread_state.session
            response = session.request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout_s,
            )
        except requests.RequestException as exc:
            log.debug(f"[prompt-upsampling] _request_json FAILED: {exc}")
            raise RuntimeError(f"Failed to reach {url}: {exc}") from exc

        if response_path is not None:
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(response.text, encoding="utf-8")
            log.info("Raw API response saved: %s", response_path)
        log.debug(f"[prompt-upsampling] _request_json response: status={response.status_code}")
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} from {url}: {response.text[:1000]}")

        try:
            parsed = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Response from {url} was not valid JSON: {response.text[:1000]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Response from {url} must be a JSON object.")
        return parsed

    def _with_retries(self, operation: str, fn: Callable[[], Any]) -> Any:
        """Retry transient endpoint/model failures with exponential backoff."""
        if self.config.max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt == self.config.max_retries - 1:
                    break
                self._sleep(self.config.retry_base_delay_s * (2 ** attempt))
        raise RuntimeError(
            f"Prompt upsampler failed to {operation} after {self.config.max_retries} attempts: {last_exc}"
        ) from last_exc


def _normalize_openai_base_url(url: str) -> str:
    """Normalize user-provided endpoint strings to a request base URL."""
    normalized = url.strip().rstrip("/")
    if not normalized:
        raise ValueError("endpoint_url cannot be empty.")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _make_session(config: PromptUpsamplerConfig) -> requests.Session:
    """Create a requests session with connection-level retry behavior."""
    session = requests.Session()
    retry = Retry(
        total=config.connection_max_retries,
        connect=config.connection_max_retries,
        read=0,
        status=0,
        backoff_factor=0.25,
        allowed_methods=None,
    )
    adapter = HTTPAdapter(
        pool_connections=config.connection_pool_size,
        pool_maxsize=config.connection_pool_size,
        max_retries=retry,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _message_content_to_text(content: Any) -> str:
    """Convert OpenAI message content variants into plain assistant text."""
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        text = "".join(parts).strip()
        if text:
            return text
    raise ValueError("Chat completion message content is empty or unsupported.")


def _output_stem(source: Path, input_path: str | Path) -> str:
    """Name nested outputs after their containing folder; keep direct filenames."""
    root = Path(input_path)
    if source != root and source.parent != root:
        return source.parent.name
    return source.stem


def _read_prompts(path: str | Path) -> list[tuple[Path, str]]:
    """Read one whole-file prompt per .txt file, in filename order.

    Directory inputs include all nested .txt files. Empty files and duplicate
    output names are invalid because outputs share one directory.
    """
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise ValueError(f"Expected a .txt file: {source}")
        files = [source]
    elif source.is_dir():
        files = sorted(
            (item for item in source.rglob("*") if item.is_file() and item.suffix.lower() == ".txt"),
            key=lambda item: (item.name.casefold(), item.name, item.relative_to(source).as_posix()),
        )
        if not files:
            raise ValueError(f"No .txt files found in {source}.")
    else:
        raise ValueError(f"Input path does not exist: {source}")
    output_sources: dict[str, Path] = {}
    for file in files:
        output_name = f"{_output_stem(file, source)}.json"
        output_key = output_name.casefold()
        if output_key in output_sources:
            raise ValueError(
                f"Duplicate output filename {output_name!r}: "
                f"{output_sources[output_key]} and {file}. "
                "Use unique containing folder names for nested inputs and unique filenames for direct inputs."
            )
        output_sources[output_key] = file
    prompts = []
    for file in files:
        prompt = file.read_text(encoding="utf-8-sig").strip()
        if not prompt:
            raise ValueError(f"Prompt file is empty: {file}")
        prompts.append((file, prompt))
    return prompts


def _optional_float(value: str) -> float | None:
    """Parse a float CLI argument that can also be disabled with ``none``."""
    if value.strip().lower() in {"none", "null"}:
        return None
    return float(value)


def configure_prompting_templates(
        *,
        prompt_template_path: str | Path | None = None,
        json_template_path: str | Path | None = None,
) -> None:
    """Override the text-to-image instruction template and JSON schema."""
    global T2I_JSON_TEMPLATE, T2I_PROMPT_TEMPLATE
    if json_template_path is not None:
        T2I_JSON_TEMPLATE = Path(json_template_path).read_text(encoding="utf-8").rstrip("\n")
    if prompt_template_path is not None:
        T2I_PROMPT_TEMPLATE = Template(Path(prompt_template_path).read_text(encoding="utf-8").rstrip("\n"))


def _prompt_record(upsampled_json: str) -> dict[str, Any]:
    """Return the caption object directly, matching the active JSON template."""
    return validate_t2i_caption(extract_json_object(upsampled_json))


def upsample_prompt_file(
        client: PromptUpsamplerClient,
        *,
        additional_instruction:str,
        input_path: str | Path,
        output_path: str | Path,
        resolution: str,
        aspect_ratio: str,
        max_workers: int = 16,
        debug: bool = False,
) -> list[dict[str, Any]]:
    """Upsample a .txt file or directory into structured .json captions.

    Directory inputs are searched recursively; outputs are written to one
    directory. Nested inputs use their containing folder's name; direct inputs
    and single-file inputs use their filename stem, all with a .json extension.
    Each output file contains the structured caption object directly.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1.")
    prompts = _read_prompts(input_path)
    results: dict[int, dict[str, Any]] = {}
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    failures: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                client.upsample_t2i, prompt, additional_instruction=additional_instruction, resolution=resolution, aspect_ratio=aspect_ratio,
                **({"debug_dir": output / _output_stem(source, input_path)} if debug else {}),
            ): index
            for index, (source, prompt) in enumerate(prompts)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Upsampling", unit="prompt"):
            index = futures[future]
            source, _ = prompts[index]
            try:
                result = _prompt_record(future.result())
                (output / f"{_output_stem(source, input_path)}.json").write_text(
                    json.dumps(result, ensure_ascii=JSON_ENSURE_ASCII, indent=2) + "\n", encoding="utf-8"
                )
                results[index] = result
                # log.info("Saved %s (%d/%d completed)", source.name, len(results) + len(failures), len(prompts))
            except Exception:
                failures.append(source)
                log.exception("Failed to upsample %s", source)
    if failures:
        raise RuntimeError(
            f"Failed to upsample {len(failures)}/{len(prompts)} files: {', '.join(str(p) for p in failures)}")
    return [results[index] for index in range(len(prompts))]


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the standalone CLI parser for batch prompt upsampling."""
    parser = argparse.ArgumentParser(description="Upsample .txt files into Cosmos3 text-to-image JSON prompts.")
    parser.add_argument("--input", required=True,
                        help="Single .txt file or directory searched recursively for .txt files; each whole file is one prompt. Output names must be unique.")
    parser.add_argument("--output", required=True,
                        help="Output directory; nested inputs use their containing folder's name, direct inputs use their filename stem, with a .json extension.")
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("PROMPT_UPSAMPLER_ENDPOINT_URL"),
        help="OpenAI-compatible endpoint URL. Defaults to PROMPT_UPSAMPLER_ENDPOINT_URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PROMPT_UPSAMPLER_MODEL"),
        help="Model name. Defaults to PROMPT_UPSAMPLER_MODEL.",
    )
    parser.add_argument(
        "--api-token",
        default=os.environ.get("PROMPT_UPSAMPLER_API_TOKEN"),
        help="API token. Defaults to PROMPT_UPSAMPLER_API_TOKEN.",
    )
    parser.add_argument(
        "--prompt-template",
        default=os.environ.get("PROMPT_UPSAMPLER_PROMPT_TEMPLATE"),
        help=(
            "Prompt template path for text-to-image. Defaults to PROMPT_UPSAMPLER_PROMPT_TEMPLATE, "
            "then the built-in template."
        ),
    )
    parser.add_argument(
        "--json-template",
        default=os.environ.get("PROMPT_UPSAMPLER_JSON_TEMPLATE"),
        help=(
            "JSON schema template path for text-to-image. Defaults to PROMPT_UPSAMPLER_JSON_TEMPLATE, "
            "then the built-in template."
        ),
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
                        help="Maximum parallel requests (default: 16).")
    parser.add_argument("--retry-base-delay-s", type=float, default=1.0, help="Base exponential retry delay.")
    parser.add_argument(
        "--temperature", type=_optional_float, default=None, help="Optional sampling temperature, or 'none'."
    )
    parser.add_argument("--top-p", type=_optional_float, default=None, help="Optional sampling top-p, or 'none'.")
    parser.add_argument("--top-k", type=int, default=20, help="Sampling top-k.")
    parser.add_argument("--min-p", type=float, default=None, help="Optional sampling min-p.")
    parser.add_argument(
        "--structured-output", action="store_true",
        help="Compatibility flag: JSON Schema output is always enforced, including without this flag.",
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=False,
        help="Enable model thinking via chat_template_kwargs (default: disabled).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--debug", action="store_true",
        help="Save raw API responses to <output>/<output-stem>/<attempt>/response.json (attempts start at 0).",
    )
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point used by ``python scripts/prompt_upsampling.py``."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.endpoint_url is None:
        parser.error("--endpoint-url is required unless PROMPT_UPSAMPLER_ENDPOINT_URL is set.")
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1.")

    configure_prompting_templates(
        prompt_template_path=args.prompt_template,
        json_template_path=args.json_template,
    )

    # Build the client after CLI validation so endpoint/model/token mistakes
    # fail before any input file is processed.
    config = PromptUpsamplerConfig(
        endpoint_url=args.endpoint_url,
        model=args.model,
        api_token=args.api_token,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_base_delay_s=args.retry_base_delay_s,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        enable_thinking=args.enable_thinking,
    )
    client = PromptUpsamplerClient(config)
    results = upsample_prompt_file(
        client,
        additional_instruction=args.additional_instruction,
        input_path=args.input,
        output_path=args.output,
        resolution=args.resolution,
        aspect_ratio=args.aspect_ratio,
        max_workers=args.max_workers,
        debug=args.debug,
    )
    log.info("Wrote %d upsampled prompts to %s", len(results), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
