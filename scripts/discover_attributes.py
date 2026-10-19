import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

from openai import APITimeoutError, BadRequestError

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(_PACKAGE_DIR))

from src.request.client import chat_completion, create_client, resolve_model, run_batch, with_retries
from src.request.response_errors import EmptyResponseError, response_content
from src.request.response_parsing import parse_json_response

log = logging.getLogger(__name__)
ATTRIBUTE_FIELDS = ("foreground", "background", "lighting", "weather", "visibility")
ATTRIBUTE_SCHEMA = {
    "type": "object",
    "properties": {
        field: {"type": "array", "items": {"type": "string"}}
        for field in ATTRIBUTE_FIELDS
    },
    "required": list(ATTRIBUTE_FIELDS),
    "additionalProperties": False,
}
# A size heuristic, not a tokenizer: the server still enforces its token limit.
CONSOLIDATION_MAX_CHARS = 48000
REQUEST_PROGRESS_SECONDS = 30.0


def load_examples(input_dir: Path) -> list[dict]:
    """Read every JSON file in filename order, validating before API requests."""
    files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not files:
        raise ValueError(f"Nessun file JSON trovato in {input_dir}.")

    examples = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Impossibile leggere {path.name}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"{path.name} non contiene un oggetto JSON.")

        examples.append(data)

    return examples


def _validate_attributes(data, input_data) -> dict[str, list[str]]:
    if not isinstance(data, dict) or set(data) != set(ATTRIBUTE_FIELDS):
        raise ValueError("Attribute response must contain exactly the requested fields.")
    for field, values in data.items():
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"{field} must be an array of nonempty strings.")
        if isinstance(input_data, dict) and bool(values) != bool(input_data[field]):
            raise ValueError(f"Consolidation must preserve empty/nonempty fields: {field}.")
        data[field] = list(dict.fromkeys(values))
    return data


def _write_json_atomic(path: Path, data) -> None:
    """Keep an existing result intact if writing or execution is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".attributes-",
            suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _request_too_large(error: Exception) -> bool:
    if isinstance(error, RuntimeError) and error.__cause__ is not None:
        return _request_too_large(error.__cause__)
    if isinstance(error, BadRequestError):
        message = str(error).lower()
        return any(term in message for term in (
            "maximum context length", "context length", "context_length_exceeded",
            "max_model_len", "too many tokens",
        ))
    return isinstance(error, ValueError) and str(error).startswith("Chat completion truncated")


def _consolidation_should_split(error: Exception) -> bool:
    if isinstance(error, RuntimeError) and error.__cause__ is not None:
        return _consolidation_should_split(error.__cause__)
    return isinstance(error, EmptyResponseError) or _request_too_large(error)


async def _request_json(
    args: argparse.Namespace, client, model: str,
    system_prompt: str, input_data: list[dict] | dict,
    *, operation: str, response_stem: str, schema: dict, response_name: str,
    validate, split_on_limit: bool = False, compatible_schema: dict | None = None,
) -> dict:
    """Request validated JSON and preserve compatible extraction checkpoints."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))},
    ]
    enable_thinking = getattr(args, "enable_thinking", True)
    # Qwen3.8-27B recommended thinking / instruct sampling profiles.
    parameters = dict(
        model=model, messages=messages, max_tokens=args.max_tokens,
        temperature=1.0 if enable_thinking else 0.7,
        top_p=0.95 if enable_thinking else 0.8,
        presence_penalty=0.0 if enable_thinking else 1.5,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": response_name, "strict": True, "schema": schema,
            },
        },
        extra_body={
            "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )
    if enable_thinking:
        parameters["reasoning_effort"] = args.reasoning_effort
    # Timeout, worker count and retry limits may change without invalidating
    # completed requests; input, prompt, model and generation parameters may not.
    def request_signature(request_parameters):
        return hashlib.sha256(json.dumps(
            {"server": args.server, "parameters": request_parameters},
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()

    signature = request_signature(parameters)
    checkpoint_path = args.output / "checkpoints" / f"{response_stem}.{signature}.json"
    parameter_variants = [parameters]
    if compatible_schema is not None:
        # A stricter schema may enforce invariants that validation already
        # checked. Reuse older results only after that same validation succeeds.
        previous_parameters = {**parameters, "response_format": {
            **parameters["response_format"], "json_schema": {
                **parameters["response_format"]["json_schema"], "schema": compatible_schema,
            },
        }}
        parameter_variants.append(previous_parameters)
    previous_max_tokens = getattr(args, "resume_max_tokens", None)
    if previous_max_tokens is not None and previous_max_tokens != args.max_tokens:
        # Explicitly reuse completed, validated requests from the old budget.
        # Input, model, prompt, sampling and all other parameters still match.
        parameter_variants += [{**variant, "max_tokens": previous_max_tokens}
                               for variant in list(parameter_variants)]
    checkpoint_candidates = []
    for variant in parameter_variants:
        saved_signature = request_signature(variant)
        saved_path = args.output / "checkpoints" / f"{response_stem}.{saved_signature}.json"
        checkpoint_candidates.append((saved_path, saved_signature, variant["max_tokens"]))
    for saved_path, saved_signature, saved_max_tokens in (checkpoint_candidates if getattr(args, "resume", True) else ()):
        if not saved_path.is_file():
            continue
        try:
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("request_sha256") != saved_signature:
                raise ValueError("Checkpoint request signature does not match.")
            data = validate(saved.get("attributes"))
        except (OSError, ValueError) as exc:
            log.warning("Ignoring invalid checkpoint %s: %s", saved_path, exc)
        else:
            log.info("Reusing completed request: %s (checkpoint max_tokens=%d)", operation, saved_max_tokens)
            return data

    def final_content(response, response_path):
        text = response_content(
            response, max_tokens=args.max_tokens, response_path=response_path,
        )
        # Some servers return thinking inline instead of reasoning_content.
        text = re.sub(r"\A\s*<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
        if not text.strip():
            raise EmptyResponseError(
                f"Chat completion contains no final answer "
                f"(max_tokens={args.max_tokens}, usage={response.usage!r}, "
                f"response={response_path or 'not saved (enable --debug)'})."
            )
        return text

    async def attempt(index: int) -> dict:
        response_path = (
            args.output / "debug" / f"{response_stem}.attempt-{index}.json"
            if args.debug else None
        )
        started = asyncio.get_running_loop().time()
        log.info(
            "%s: starting attempt %d/%d; input_chars=%d; max_tokens=%d; thinking=%s; reasoning_effort=%s",
            operation, index + 1, args.max_retries, len(messages[1]["content"]),
            args.max_tokens, enable_thinking, parameters.get("reasoning_effort", "disabled"),
        )
        pending = asyncio.create_task(chat_completion(
            client, response_path=response_path, **parameters,
        ))
        try:
            while True:
                done, _ = await asyncio.wait({pending}, timeout=REQUEST_PROGRESS_SECONDS)
                if done:
                    response = await pending
                    break
                log.info(
                    "%s: attempt %d/%d still waiting after %.0fs; "
                    "no complete HTTP response yet (HTTP timeout=%ss)",
                    operation, index + 1, args.max_retries,
                    asyncio.get_running_loop().time() - started, args.timeout,
                )
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        log.info(
            "%s: response received after %.1fs; finish_reason=%s; usage=%s",
            operation, asyncio.get_running_loop().time() - started,
            response.choices[0].finish_reason if response.choices else None, response.usage,
        )
        try:
            text = final_content(response, response_path)
            return validate(parse_json_response(text))
        except ValueError as exc:
            if not split_on_limit or response_path is not None:
                raise
            suffix = "empty" if isinstance(exc, EmptyResponseError) else "invalid"
            response_path = args.output / "debug" / f"{response_stem}.attempt-{index}.{suffix}.json"
            _write_json_atomic(response_path, response.model_dump(mode="json"))
            log.warning("Invalid final answer (%s); response snapshot saved: %s", exc, response_path)
            if isinstance(exc, EmptyResponseError) or _request_too_large(exc):
                # Include the diagnostic path in empty/truncated response errors.
                final_content(response, response_path)
            raise

    data = await with_retries(
        operation, attempt,
        max_attempts=args.max_retries, delay_s=1.0, server=args.server,
        # A slow consolidation must not silently restart the same long request.
        # A timeout is not evidence of a context limit, so do not split it either.
        retry_if=(lambda exc: not isinstance(exc, APITimeoutError) and not _consolidation_should_split(exc)) if split_on_limit else None,
    )
    _write_json_atomic(checkpoint_path, {"request_sha256": signature, "attributes": data})
    return data


async def _request_attributes(
    args: argparse.Namespace, client, model: str,
    system_prompt: str, input_data: list[dict] | dict[str, list[str]],
    *, operation: str, response_stem: str, split_on_limit: bool = False,
) -> dict[str, list[str]]:
    return await _request_json(
        args, client, model, system_prompt, input_data,
        operation=operation, response_stem=response_stem,
        schema=ATTRIBUTE_SCHEMA, response_name="discovered_attributes",
        validate=lambda data: _validate_attributes(data, input_data),
        split_on_limit=split_on_limit,
    )


def _combine_attributes(left, right) -> dict[str, list[str]]:
    return {field: list(dict.fromkeys(left[field] + right[field])) for field in ATTRIBUTE_FIELDS}


async def _merge_attribute_pair(args, client, model, prompt, left, right, *, operation, response_stem):
    """Merge two vocabularies; fall back to independent fields, never all pairs."""
    expected = _combine_attributes(left, right)
    active_fields = [field for field in ATTRIBUTE_FIELDS if expected[field]]
    thinking_status = "enabled" if getattr(args, "enable_thinking", True) else "disabled"

    async def request(first, second, stem, label):
        payload = {"left": first, "right": second}
        combined = _combine_attributes(first, second)
        schema = {
            **ATTRIBUTE_SCHEMA,
            "properties": {
                field: {**ATTRIBUTE_SCHEMA["properties"][field],
                        **({"minItems": 1} if combined[field] else {"maxItems": 0})}
                for field in ATTRIBUTE_FIELDS
            },
        }
        return await _request_json(
            args, client, model, prompt, payload, operation=label, response_stem=stem,
            schema=schema, response_name="discovered_attributes",
            validate=lambda data: _validate_attributes(data, combined),
            split_on_limit=True, compatible_schema=ATTRIBUTE_SCHEMA,
        )

    # Large multi-field pairs go directly to independent field requests.
    failure = None
    if len(active_fields) <= 1 or len(json.dumps({"left": left, "right": right}, ensure_ascii=False)) <= CONSOLIDATION_MAX_CHARS:
        try:
            return await request(left, right, response_stem, operation)
        except RuntimeError as exc:
            if not _consolidation_should_split(exc):
                raise
            failure = exc

    def cannot_merge(field, cause):
        if cause is not None and isinstance(cause.__cause__, EmptyResponseError):
            return RuntimeError(
                f"{operation}: no final answer for field {field}. Thinking is {thinking_status}. "
                "Inspect the response snapshot under output/debug/ and the server reasoning parser."
            )
        return RuntimeError(
            f"{operation}: field {field} cannot be merged within the available context/output budget. "
            "Increase the server context length or adjust --consolidation-max-tokens. "
            "Completed merges are checkpointed; no attributes have been truncated."
        )

    if len(active_fields) <= 1:
        # The full request was already limited to this one field: repeating it
        # under a different name would send the same failing payload again.
        raise cannot_merge(active_fields[0] if active_fields else "empty vocabulary", failure) from failure
    log.warning("%s: merging %d fields separately; thinking is %s", operation, len(active_fields), thinking_status)
    result = {field: [] for field in ATTRIBUTE_FIELDS}
    # Stay inside the parent worker so field fallbacks respect --max-workers.
    for field in active_fields:
        first = {name: left[name] if name == field else [] for name in ATTRIBUTE_FIELDS}
        second = {name: right[name] if name == field else [] for name in ATTRIBUTE_FIELDS}
        try:
            merged = await request(first, second, f"{response_stem}.{field}", f"{operation}/{field}")
        except RuntimeError as exc:
            if not _consolidation_should_split(exc):
                raise
            raise cannot_merge(field, exc) from exc
        result[field] = merged[field]
    return result


async def consolidate_attributes(args, client, model, prompt, batches) -> dict[str, list[str]]:
    """Reduce extraction JSONs pairwise, with independent merges per tree level.

    N inputs require N-1 logical merges (one normalization request for N=1).
    Odd nodes pass to the next level unchanged. Completed requests are cached.
    """
    if not batches:
        raise ValueError("No attribute JSONs to consolidate.")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1.")
    nodes = [batches] if isinstance(batches, dict) else list(batches)
    nodes = [_validate_attributes(dict(node) if isinstance(node, dict) else node, node) for node in nodes]
    consolidation_args = argparse.Namespace(**vars(args))
    consolidation_args.max_tokens = getattr(args, "consolidation_max_tokens", None) or args.max_tokens
    consolidation_args.reasoning_effort = getattr(args, "consolidation_reasoning_effort", None) or args.reasoning_effort
    consolidation_args.enable_thinking = consolidation_args.reasoning_effort != "disabled"
    if len(nodes) == 1:
        return await _merge_attribute_pair(
            consolidation_args, client, model, prompt, nodes[0], {field: [] for field in ATTRIBUTE_FIELDS},
            operation="Consolidate attributes", response_stem="attributes.consolidation",
        )
    level = 1
    while len(nodes) > 1:
        current = nodes
        pair_starts = list(range(0, len(current) - 1, 2))
        log.info(
            "Consolidation round %d: %d JSONs -> %d merges, %d carried; max_workers=%d; thinking=%s",
            level, len(current), len(pair_starts), len(current) % 2,
            args.max_workers, consolidation_args.reasoning_effort if consolidation_args.enable_thinking else "disabled",
        )

        async def merge(start):
            pair = start // 2 + 1
            return await _merge_attribute_pair(
                consolidation_args, client, model, prompt, current[start], current[start + 1],
                operation=f"Consolidate attributes round {level} pair {pair}",
                response_stem=f"attributes.consolidation.round-{level:04d}.pair-{pair:04d}",
            )

        nodes = await run_batch(
            pair_starts, merge, max_workers=args.max_workers,
            desc=f"Consolidating round {level}", unit="pair",
        )
        if len(current) % 2:
            nodes.append(current[-1])
        level += 1
    return nodes[0]


async def discover_attributes(
    args: argparse.Namespace, system_prompt: str, examples: list[dict],
) -> dict[str, list[str]]:
    """Checkpoint completed requests; publish attributes only after consolidation."""
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1.")
    if getattr(args, "consolidation_max_tokens", None) is not None and args.consolidation_max_tokens < 1:
        raise ValueError("--consolidation-max-tokens must be >= 1.")
    if getattr(args, "resume_max_tokens", None) is not None and args.resume_max_tokens < 1:
        raise ValueError("--resume-max-tokens must be >= 1.")
    if not examples:
        raise ValueError("No examples to analyze.")
    consolidation_prompt = (_PACKAGE_DIR / "src/templates/consolidate_attributes.txt").read_text(encoding="utf-8")
    total_batches = (len(examples) + args.batch_size - 1) // args.batch_size
    async with create_client(
        args.server, api_token=args.api_token, timeout_s=args.timeout,
    ) as client:
        model = await resolve_model(client, args.model)
        log.info(
            "Using model: %s; %d examples in %d batches; max_workers=%d",
            model, len(examples), total_batches, args.max_workers,
        )

        async def process(start: int) -> dict[str, list[str]]:
            batch_number = start // args.batch_size + 1
            batch = examples[start:start + args.batch_size]
            log.info("Analyzing batch %d/%d (%d examples)", batch_number, total_batches, len(batch))
            return await _request_attributes(
                args, client, model, system_prompt, batch,
                operation=f"Discover attributes batch {batch_number}",
                response_stem=f"attributes.batch-{batch_number:04d}",
            )

        results = await run_batch(
            list(range(0, len(examples), args.batch_size)), process,
            max_workers=args.max_workers, desc="Discovering attributes", unit="batch",
        )
        log.info("Consolidating %d extraction JSONs with pairwise tree reduction", len(results))
        data = await consolidate_attributes(args, client, model, consolidation_prompt, results)

    output_path = args.output / "attributes.json"
    _write_json_atomic(output_path, data)
    log.info("Attributes saved: %s", output_path)
    return data


def main():
    parser = argparse.ArgumentParser(description="Discover visual attributes with Qwen3.8-27B thinking.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20, help="Maximum descriptions per request; all files are analyzed.")
    parser.add_argument("--max-workers", "--max_workers", type=int, default=16, help="Maximum concurrent extraction batches or merges within a consolidation round; use 1 for sequential requests.")
    parser.add_argument("--consolidation-max-tokens", type=int, default=None, help="Separate consolidation output budget; defaults to --max-tokens and does not invalidate extraction checkpoints.")
    parser.add_argument("--consolidation-reasoning-effort", choices=("disabled", "low", "medium", "xhigh"), default=None, help="Thinking level for consolidation only; defaults to --reasoning-effort. Use disabled to turn off thinking and apply Qwen instruct sampling. Extraction checkpoints remain compatible.")
    parser.add_argument("--server", required=True, help="OpenAI-compatible endpoint URL.")
    parser.add_argument("--model", default=None, help="Model ID; defaults to the first model returned by the endpoint.")
    parser.add_argument("--api-token", "--api_token", default=None)
    parser.add_argument(
        "--max-tokens", type=int, default=65536,
        help="Output budget, including thinking. Input plus output must fit the server context limit.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum attempts, including the first.")
    parser.add_argument("--timeout", type=float, default=1600.0, help="HTTP timeout in seconds per attempt; increase for slow or queued responses.")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    parser.add_argument("--debug", action="store_true", help="Save raw API responses under output/debug/.")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse matching completed requests from output/checkpoints/ (default); --no-resume regenerates them.",
    )
    parser.add_argument("--resume-max-tokens", type=int, default=None, help="Also reuse validated checkpoints created with this previous output budget. Only max_tokens may differ; new requests use the current budgets.")
    args = parser.parse_args()

    for option in ("batch_size", "max_workers", "max_tokens", "max_retries"):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be >= 1.")
    if args.consolidation_max_tokens is not None and args.consolidation_max_tokens < 1:
        parser.error("--consolidation-max-tokens must be >= 1.")
    if args.resume_max_tokens is not None and args.resume_max_tokens < 1:
        parser.error("--resume-max-tokens must be >= 1.")
    if not args.timeout > 0:
        parser.error("--timeout must be > 0.")
    if not args.server.strip():
        parser.error("--server cannot be empty.")
    if args.model is not None and not args.model.strip():
        parser.error("--model cannot be empty.")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_dir.is_dir():
        parser.error("Input directory not found.")
    if output_dir.exists() and not output_dir.is_dir():
        parser.error("Output path must be a directory.")
    args.output = output_dir

    try:
        examples = load_examples(input_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    system_prompt = (_PACKAGE_DIR / "src/templates/discover_attributes.txt").read_text(encoding="utf-8")
    try:
        asyncio.run(discover_attributes(args, system_prompt, examples))
    except RuntimeError as exc:
        log.error("%s", exc)
        if isinstance(exc.__cause__, APITimeoutError):
            log.error(
                "Timeout HTTP: nessuna risposta completa entro il limite configurato (%s s). "
                "Controlla sul server il throughput e le richieste Running/Waiting prima di aumentare "
                "--timeout o cambiare --max-workers %d.",
                args.timeout, args.max_workers,
            )
        log.error(
            "I checkpoint completati sono conservati in %s. "
            "Rilancia con gli stessi input, output e batch-size per riprendere automaticamente. "
            "attributes.json non e' stato aggiornato.",
            args.output / "checkpoints",
        )
        parser.exit(1)

if __name__ == "__main__":
    main()
