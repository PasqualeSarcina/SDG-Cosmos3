"""Generate compact scene plans using local vLLM or an OpenAI-compatible API.

Example (Linux/WSL with the caption-vllm dependency group installed)::

    python scripts/generate_thinkink.py -i data/captions -o data/plans

API example (base dependencies only; OPENAI_API_KEY supplies authentication)::

    python scripts/generate_thinkink.py -i data/captions -o data/plans --server http://localhost:8000/v1

The API server must support Qwen thinking and the vLLM sampling/chat-template
extensions. It may return reasoning_content (or reasoning) separately, or native
<think>...</think> text. Responses without thinking are rejected.
Use --reasoning-effort low|medium|xhigh in either mode; the default is medium.
With --server, --batch-size controls API concurrency; --max-workers only controls
the CPU threads used to read XML files.

Inputs are scanned recursively. Each input's relative path is preserved as an
.xml file containing one <scene_plan> block. Only the final scene plan is saved;
native thinking is retained in raw responses under output/debug/ with --debug.
Teacher responses marked <annotation_issue> are skipped without retries or files,
including debug files, and counted separately from failures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from xml.sax.saxutils import escape

import tyro
from jsonschema import ValidationError
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

if TYPE_CHECKING:
    from vllm.outputs import CompletionOutput

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(_PACKAGE_DIR))

from src.validation.scene_draft import validate_scene_draft
from src.request.client import create_client, normalize_base_url, resolve_model

log = logging.getLogger(__name__)
REASONING_EFFORT = "medium"
# Fixed Qwen3.8-27B thinking-mode sampling recommendations:
# https://huggingface.co/Qwen/Qwen3.8-27B#best-practices
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 0.0
REPETITION_PENALTY = 1.0
DEFAULT_TEMPLATE = _PACKAGE_DIR / "src/templates/scene_plan_teacher_system.txt"


@dataclass
class Args:
    """Annotate scene_draft XMLs locally or via --server (base dependencies only)."""

    input: Annotated[Path, tyro.conf.arg(aliases=["-i"])]
    """XML file or directory, scanned recursively, excluding debug directories."""
    output: Annotated[Path, tyro.conf.arg(aliases=["-o"])]
    """Destination for UTF-8 XML plans with a <scene_plan> root; existing files are never overwritten."""
    model: str = "Qwen/Qwen3.8-27B"
    """Local checkpoint/Hugging Face ID; ignored with --server (auto-resolved)."""
    reasoning_effort: Annotated[Literal["low", "medium", "xhigh"], tyro.conf.arg(aliases=["--reasoning_effort"])] = REASONING_EFFORT
    """Thinking effort applied to the local chat template or API request."""
    server: str | None = None
    """OpenAI-compatible server URL; omit to use local vLLM and CUDA."""
    api_token: Annotated[str | None, tyro.conf.arg(aliases=["--api_token"])] = field(default=None, repr=False)
    """API key; defaults to OPENAI_API_KEY, or EMPTY for an unauthenticated server."""
    timeout_s: float = 300.0
    """API request timeout in seconds."""
    batch_size: Annotated[int, tyro.conf.arg(aliases=["--batch_size"])] = 32
    """Scenes per generation group; also the maximum concurrent API requests with --server."""
    max_workers: Annotated[int, tyro.conf.arg(aliases=["--max_workers"])] = 4
    """CPU threads for reading XMLs; does not limit API concurrency."""
    max_tokens: Annotated[int, tyro.conf.arg(aliases=["--max_tokens"])] = 8192
    """Maximum new tokens per response, including native thinking and final plan."""
    max_retries: Annotated[int, tyro.conf.arg(aliases=["--max_retries"])] = 3
    """Maximum generation attempts per source, including format failures."""
    gpu_memory_utilization: Annotated[float, tyro.conf.arg(aliases=["--gpu_memory_utilization"])] = 0.8
    spec_draft_n_max: int = 0
    """Optional MTP speculative tokens per step; disabled by default."""
    prompt_template_path: Annotated[Path, tyro.conf.arg(aliases=["--prompt_template_path"])] = DEFAULT_TEMPLATE
    """UTF-8 system prompt; the user message contains only the scene_draft JSON."""
    debug: bool = False
    """Save raw responses, finish reasons and token counts under output/debug/."""
    scene_items: list[Path] = field(init=False, repr=False)
    input_root: Path = field(init=False, repr=False)
    prompt_template: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.reasoning_effort not in {"low", "medium", "xhigh"}:
            raise ValueError("--reasoning-effort must be low, medium, or xhigh.")
        if self.server is not None:
            self.server = normalize_base_url(self.server)
            if self.spec_draft_n_max:
                raise ValueError("--spec-draft-n-max is local-only; configure speculation on the API server.")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("--timeout-s must be finite and > 0.")
        for name in ("batch_size", "max_workers", "max_tokens", "max_retries"):
            if getattr(self, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 1.")
        if self.spec_draft_n_max < 0:
            raise ValueError("--spec-draft-n-max must be >= 0.")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("--gpu-memory-utilization must be in (0, 1].")
        if not self.model.strip():
            raise ValueError("--model cannot be empty.")
        self.input = Path(self.input).resolve()
        self.output = Path(self.output).resolve()
        if self.output.exists() and not self.output.is_dir():
            raise ValueError(f"Output path must be a directory: {self.output}")
        if self.input == self.output:
            raise ValueError("Input and output must be different paths.")
        if self.input.is_dir():
            self.input_root = self.input
            excluded_output = self.output if self.output.is_relative_to(self.input) else None
            self.scene_items = sorted(
                (p for p in self.input.rglob("*") if p.is_file() and p.suffix.lower() == ".xml"
                 and (excluded_output is None or not p.is_relative_to(excluded_output))
                 and "debug" not in {part.casefold() for part in p.relative_to(self.input).parts[:-1]}),
                key=lambda p: (p.as_posix().casefold(), p.as_posix()),
            )
        elif self.input.is_file() and self.input.suffix.lower() == ".xml":
            self.input_root = self.input.parent
            self.scene_items = [self.input]
        else:
            raise ValueError(f"XML input path is missing or unsupported: {self.input}")
        if not self.scene_items:
            raise ValueError(f"No XML inputs found in {self.input}.")
        names: set[str] = set()
        for path in self.scene_items:
            name = self.plan_path(path).relative_to(self.output).as_posix().casefold()
            if name in names:
                raise ValueError(f"Ambiguous output plan path: {name!r}.")
            names.add(name)
        _check_output_paths(self)
        self.prompt_template = Path(self.prompt_template_path).read_text(encoding="utf-8-sig")
        if not self.prompt_template.strip():
            raise ValueError(f"Prompt template file is empty: {self.prompt_template_path}")

    def plan_path(self, source: Path) -> Path:
        return self.output / source.relative_to(self.input_root).with_suffix(".xml")


def _check_output_paths(args: Args) -> None:
    for source in args.scene_items:
        path = args.plan_path(source)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing plan: {path}")


def _read_scene(path: Path) -> str | Exception:
    """Extract only the draft; Cosmos XMLs have two top-level application tags."""
    try:
        text = path.read_text(encoding="utf-8-sig")
        if text.count("<scene_draft>") != 1 or text.count("</scene_draft>") != 1:
            raise ValueError("Expected exactly one <scene_draft>...</scene_draft> block.")
        match = re.search(r"<scene_draft>\s*(.*?)\s*</scene_draft>", text, re.DOTALL)
        if match is None or not match.group(1).strip():
            raise ValueError("Missing or empty <scene_draft> block.")
        draft = match.group(1).strip()
        scene = json.loads(draft)
        if not isinstance(scene, dict):
            raise ValueError("scene_draft must contain a JSON object.")
        validate_scene_draft(scene)
        return draft
    except (OSError, ValueError, ValidationError) as exc:
        return exc


class VLLMScenePlanner:
    def __init__(self, args: Args):
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG" if args.debug else "WARNING"
        os.environ["TRANSFORMERS_VERBOSITY"] = "debug" if args.debug else "warning"
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "This script requires vLLM. Install the caption-vllm dependency group "
                "on Linux/WSL with compatible CUDA GPUs."
            ) from exc
        import torch
        from transformers import AutoTokenizer

        for logger_name in ("vllm", "transformers", "huggingface_hub"):
            logging.getLogger(logger_name).setLevel(logging.DEBUG if args.debug else logging.WARNING)
        gpu_count = torch.cuda.device_count()
        if not torch.cuda.is_available() or gpu_count < 1:
            raise RuntimeError("This script requires at least one CUDA GPU visible to PyTorch/vLLM.")
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=False)
        template = self.tokenizer.get_chat_template()
        if "reasoning_effort" not in template or "enable_thinking" not in template:
            raise ValueError("The checkpoint chat template must support enable_thinking and reasoning_effort.")
        # Verify the actual rendering before allocating the model on the GPUs.
        self._prompt("{}")
        self.sampling_params = SamplingParams(
            n=1, max_tokens=args.max_tokens,
            temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            min_p=MIN_P, presence_penalty=PRESENCE_PENALTY,
            repetition_penalty=REPETITION_PENALTY, ignore_eos=False,
            # Keep </think> and chat boundaries for strict reasoning/final splitting.
            skip_special_tokens=False,
        )
        # Text engine optimizations from caption_from_image_nohttp. No grammar
        # constrains the reasoning channel; the final plan is validated afterwards.
        self.model = LLM(
            model=args.model,
            tensor_parallel_size=gpu_count,
            distributed_executor_backend="mp",
            max_num_seqs=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            language_model_only=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            compilation_config=3,
            enforce_eager=False,
            performance_mode="throughput",
            disable_log_stats=not args.debug,
            generation_config="auto",
            override_generation_config={"max_new_tokens": args.max_tokens},
            speculative_config=(
                {"method": "mtp", "num_speculative_tokens": args.spec_draft_n_max}
                if args.spec_draft_n_max else None
            ),
        )
        log.debug("Local teacher=%s, effort=%s, GPUs=%d, MTP tokens=%d",
                  args.model, args.reasoning_effort, gpu_count, args.spec_draft_n_max)

    def _prompt(self, draft: str) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.args.prompt_template},
             {"role": "user", "content": draft}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=True, reasoning_effort=self.args.reasoning_effort, preserve_thinking=True,
        )
        if not prompt.rstrip().endswith("<think>"):
            raise ValueError("The checkpoint template did not render an open thinking prefix.")
        return prompt

    def generate(self, drafts: list[str]) -> list[CompletionOutput | Exception]:
        if not drafts:
            return []
        prompts = [self._prompt(draft) for draft in drafts]
        outputs = self.model.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        if len(outputs) != len(drafts):
            raise ValueError("vLLM returned a different number of responses than input drafts.")
        results: list[CompletionOutput | Exception] = []
        for request in outputs:
            if not request.finished or len(request.outputs) != 1:
                results.append(ValueError("Expected exactly one finished vLLM completion per draft."))
            else:
                results.append(request.outputs[0])
        return results


@dataclass
class APICompletion:
    text: str
    token_count: int | None
    finish_reason: str | None
    raw_response: dict


class APIScenePlanner:
    """Remote Qwen/vLLM teacher; no local model or CUDA libraries are imported."""

    def __init__(self, args: Args):
        self.args = args
        self.client = create_client(
            args.server, api_token=args.api_token or os.getenv("OPENAI_API_KEY"),
            timeout_s=args.timeout_s,
        )
        self.runner = asyncio.Runner()
        try:
            self.model = self.runner.run(resolve_model(self.client, None))
        except BaseException:
            self.close()
            raise
        log.debug("API teacher=%s, effort=%s", self.model, args.reasoning_effort)

    def close(self) -> None:
        try:
            self.runner.run(self.client.close())
        finally:
            self.runner.close()

    async def _generate_one(self, draft: str) -> APICompletion | Exception:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.args.prompt_template},
                          {"role": "user", "content": draft}],
                n=1, max_tokens=self.args.max_tokens,
                temperature=TEMPERATURE, top_p=TOP_P,
                presence_penalty=PRESENCE_PENALTY,
                reasoning_effort=self.args.reasoning_effort,
                extra_body={
                    "top_k": TOP_K, "min_p": MIN_P,
                    "repetition_penalty": REPETITION_PENALTY,
                    "chat_template_kwargs": {"enable_thinking": True,
                                             "reasoning_effort": self.args.reasoning_effort,
                                             "preserve_thinking": True},
                },
            )
            if len(response.choices) != 1:
                raise ValueError("Expected exactly one API completion per draft.")
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            # Normalize to the same continuation format as the local teacher.
            # With a reasoning parser the API has already separated the channels.
            if reasoning:
                if not isinstance(reasoning, str):
                    raise ValueError("Expected API reasoning to be text.")
                text = f"{reasoning}\n</think>\n{content}"
            else:
                text = content.strip().removeprefix("<think>")
            return APICompletion(
                text=text,
                token_count=response.usage.completion_tokens if response.usage else None,
                finish_reason=choice.finish_reason,
                raw_response=response.model_dump(mode="json"),
            )
        except Exception as exc:
            return exc

    def generate(self, drafts: list[str]) -> list[APICompletion | Exception]:
        async def generate_batch() -> list[APICompletion | Exception]:
            semaphore = asyncio.Semaphore(self.args.batch_size)

            async def generate_limited(draft: str) -> APICompletion | Exception:
                async with semaphore:
                    return await self._generate_one(draft)

            return list(await asyncio.gather(*(generate_limited(draft) for draft in drafts)))

        return self.runner.run(generate_batch())


class AnnotationIssue(ValueError):
    """The teacher reported inconsistent source data; do not retry for a positive result."""


def _parse_plan(raw: str) -> str:
    # The opening <think> is already part of the prompt, not the continuation.
    # Never search for scene_plan inside native reasoning, which may quote tags.
    if raw.count("</think>") != 1 or "<think>" in raw:
        raise ValueError("Expected one native thinking boundary; cannot isolate the final plan.")
    reasoning, final = raw.split("</think>", 1)
    if not reasoning.strip():
        raise ValueError("The model did not produce native thinking.")
    final = final.strip()
    # Depending on vLLM's stop-token decoding, the terminal EOS may be included.
    final = final.removesuffix("<|im_end|>").strip()
    match = re.fullmatch(r"<(scene_plan|annotation_issue)>\s*(.*?)\s*</\1>", final, re.DOTALL)
    if match is None:
        raise ValueError("Expected only one final <scene_plan> or <annotation_issue> block.")
    kind, body = match.groups()
    body = body.strip()
    if kind == "annotation_issue":
        raise AnnotationIssue("Teacher marked the source as unusable.")
    if not body or re.search(r"<[^>]*>|```", body):
        raise ValueError("Empty plan or unexpected markup/chat tokens inside the final block.")
    if re.search(r"\n\s*\n", body):
        raise ValueError("Scene plan must be a single compact paragraph.")
    words = len(body.split())
    return body


def _save_plan(args: Args, path: Path, completion: CompletionOutput | APICompletion, attempt: int) -> None:
    token_count = (completion.token_count if isinstance(completion, APICompletion)
                   else len(completion.token_ids))
    annotation_issue = False
    try:
        if completion.finish_reason == "length":
            raise ValueError(f"Response truncated at {token_count if token_count is not None else 'unknown'} tokens; increase --max-tokens "
                             "and ensure the context has room for both input and output.")
        if completion.finish_reason != "stop":
            raise ValueError(f"Completion did not finish normally: {completion.finish_reason!r}.")
        plan = _parse_plan(completion.text)
    except AnnotationIssue:
        annotation_issue = True
        raise
    finally:
        if args.debug and not annotation_issue:
            debug_path = args.output / "debug" / path.relative_to(args.input_root)
            debug_path = debug_path.with_suffix(f".attempt-{attempt}.json")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(json.dumps(
                {"text": completion.text, "token_count": token_count,
                 "finish_reason": completion.finish_reason, "reasoning_effort": args.reasoning_effort,
                 **({"raw_response": completion.raw_response} if isinstance(completion, APICompletion) else {})},
                ensure_ascii=False, indent=2,
            ) + "\n", encoding="utf-8")
    destination = args.plan_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file atomically without replacing existing annotations.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=destination.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(f"<scene_plan>\n{escape(plan)}\n</scene_plan>\n")
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def generate_thinkink(args: Args) -> list[str]:
    """Return failed XML paths; source issues are skipped without retries or files."""
    _check_output_paths(args)
    failed: list[str] = []
    succeeded = 0
    skipped = 0
    backend = None
    with ExitStack() as resources, logging_redirect_tqdm(), ThreadPoolExecutor(max_workers=args.max_workers) as pool, tqdm(
        total=len(args.scene_items), desc="Reading drafts", unit="scene", dynamic_ncols=True,
        mininterval=0.5, postfix={"ok": 0, "skipped": 0, "failed": 0},
    ) as progress:
        for start in range(0, len(args.scene_items), args.batch_size):
            paths = args.scene_items[start:start + args.batch_size]
            pending = []
            for path, draft in zip(paths, pool.map(_read_scene, paths)):
                if isinstance(draft, Exception):
                    log.error("Cannot read %s: %s", path, draft)
                    failed.append(path.relative_to(args.input_root).as_posix())
                    progress.update(1)
                else:
                    pending.append((path, draft))
            if pending and backend is None:
                progress.set_description("Connecting to API" if args.server else "Loading model")
                backend = APIScenePlanner(args) if args.server else VLLMScenePlanner(args)
                if args.server:
                    resources.callback(backend.close)
            progress.set_description("Planning scenes")
            progress.set_postfix(ok=succeeded, skipped=skipped, failed=len(failed), refresh=False)
            for attempt in range(1, args.max_retries + 1):
                if not pending:
                    break
                results = backend.generate([draft for _, draft in pending])
                if len(results) != len(pending):
                    raise ValueError("Backend response count does not match the pending scene count.")
                retry = []
                for (path, draft), result in zip(pending, results):
                    try:
                        if isinstance(result, Exception):
                            raise result
                        _save_plan(args, path, result, attempt)
                    except AnnotationIssue:
                        skipped += 1
                        log.info("Skipping source marked annotation_issue: %s", path)
                    except Exception as exc:
                        # Disk errors cannot be fixed by resampling.
                        if isinstance(exc, OSError) or attempt == args.max_retries:
                            log.error("Cannot annotate %s (attempt %d): %s", path, attempt, exc)
                            failed.append(path.relative_to(args.input_root).as_posix())
                        else:
                            log.debug("%s: attempt %d/%d failed: %s", path, attempt, args.max_retries,
                                      exc, exc_info=args.debug)
                            retry.append((path, draft))
                            continue
                    else:
                        succeeded += 1
                    progress.update(1)
                    progress.set_postfix(ok=succeeded, skipped=skipped, failed=len(failed), refresh=False)
                pending = retry
    log.warning("Scene plans saved: %d/%d; skipped: %d; failed: %d.",
                succeeded, len(args.scene_items), skipped, len(failed))
    if failed:
        log.error("Failed XMLs: %s", ", ".join(failed))
    return failed


def cli_main(argv: list[str] | None = None) -> int:
    args = tyro.cli(Args, args=argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s" if args.debug else "%(levelname)s: %(message)s",
    )
    return 1 if generate_thinkink(args) else 0


if __name__ == "__main__":
    sys.exit(cli_main())
