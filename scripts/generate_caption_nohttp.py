"""Invent COCO-style car captions locally with vLLM and save T2I JSON prompts."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, get_args

import tyro
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

if TYPE_CHECKING:
    from vllm.outputs import CompletionOutput

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(_PACKAGE_DIR))

from scripts.prompt_upsampling import JSON_ENSURE_ASCII, _apply_t2i_output_parameters, _OutputParameters
from src.request.response_parsing import parse_json_response
from src.validation.scene_draft import cosmos_structural_tag, validate_scene_draft

log = logging.getLogger(__name__)
USER_PROMPT = (
    "Generate a new, realistic, and visually diverse COCO-style scene containing "
    "at least one car. Invent the scene from scratch."
)


@dataclass
class Args:
    """Generate N independent JSON captions using vLLM on all visible CUDA GPUs."""

    n: Annotated[int, tyro.conf.arg(aliases=["-n"])]
    """Number of captions to generate, not the number of batches."""
    output: Annotated[Path, tyro.conf.arg(aliases=["-o"])]
    """Destination directory for caption_000001.json, caption_000002.json, etc."""
    model: str = "Qwen/Qwen3.8-27B"
    """Base checkpoint directory or Hugging Face model ID."""
    lora_path: Annotated[str | None, tyro.conf.arg(aliases=["--lora_path"])] = None
    """Optional PEFT LoRA directory or Hugging Face repository, used for all requests."""
    resolution: str = "720"
    """Resolution tier from prompt_upsampling.RESOLUTION_RATIO_DICT."""
    aspect_ratio: Annotated[str, tyro.conf.arg(aliases=["--aspect_ratio"])] = "16,9"
    """Canonical aspect ratio: 16,9, 9,16, 1,1, 4,3 or 3,4."""
    batch_size: Annotated[int, tyro.conf.arg(aliases=["--batch_size"])] = 32
    max_tokens: Annotated[int, tyro.conf.arg(aliases=["--max_tokens"])] = 3072
    max_retries: Annotated[int, tyro.conf.arg(aliases=["--max_retries"])] = 3
    """Maximum attempts per caption, including the initial attempt."""
    gpu_memory_utilization: Annotated[float, tyro.conf.arg(aliases=["--gpu_memory_utilization"])] = 0.8
    spec_draft_n_max: int = 1
    """MTP tokens per step; set to 0 for checkpoints without MTP support."""
    temperature: float = 0.7
    top_p: Annotated[float, tyro.conf.arg(aliases=["--top_p"])] = 0.8
    top_k: Annotated[int, tyro.conf.arg(aliases=["--top_k"])] = 20
    min_p: Annotated[float, tyro.conf.arg(aliases=["--min_p"])] = 0.0
    presence_penalty: Annotated[float, tyro.conf.arg(aliases=["--presence_penalty"])] = 1.5
    repetition_penalty: Annotated[float, tyro.conf.arg(aliases=["--repetition_penalty"])] = 1.0
    prompt_template_path: Annotated[Path | None, tyro.conf.arg(aliases=["--prompt_template_path"])] = (
        _PACKAGE_DIR / "src/templates/generate_caption.txt"
    )
    """UTF-8 system prompt; defaults to generate_caption.txt."""
    debug: bool = False
    """Save raw responses and token diagnostics as .txt files under output/debug/."""
    prompt_template: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("n", "batch_size", "max_tokens", "max_retries"):
            if getattr(self, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 1.")
        if self.spec_draft_n_max < 0:
            raise ValueError("--spec-draft-n-max must be >= 0.")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("--gpu-memory-utilization must be in (0, 1].")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("--temperature must be finite and >= 0.")
        if not 0 < self.top_p <= 1:
            raise ValueError("--top-p must be in (0, 1].")
        if self.top_k < 0:
            raise ValueError("--top-k must be >= 0.")
        if not 0 <= self.min_p <= 1:
            raise ValueError("--min-p must be in [0, 1].")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("--presence-penalty must be in [-2, 2].")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("--repetition-penalty must be finite and > 0.")
        if not self.model.strip():
            raise ValueError("--model cannot be empty.")
        if self.lora_path is not None and not self.lora_path.strip():
            raise ValueError("--lora-path cannot be empty.")
        _OutputParameters(self.resolution, self.aspect_ratio)
        self.output = Path(self.output)
        _check_output_paths(self)
        template_path = Path(self.prompt_template_path or _PACKAGE_DIR / "src/templates/generate_caption.txt")
        self.prompt_template = template_path.read_text(encoding="utf-8-sig")
        if not self.prompt_template.strip():
            raise ValueError(f"Prompt template file is empty: {template_path}")


def _caption_path(args: Args, index: int) -> Path:
    return args.output / f"caption_{index:06d}.json"


def _check_output_paths(args: Args) -> None:
    if args.output.exists() and not args.output.is_dir():
        raise ValueError(f"Output path must be a directory: {args.output}")
    for index in range(1, args.n + 1):
        path = _caption_path(args, index)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing caption: {path}")


def _resolve_lora(source: str) -> Path:
    """Resolve local directories or download one Hub snapshot into the HF cache."""
    path = Path(source).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"LoRA path must be a directory: {path}")
        return path.resolve()
    if (path.is_absolute() or source.startswith((".", "~")) or "\\" in source
            or (path.parent != Path(".") and path.parent.is_dir())):
        raise FileNotFoundError(f"LoRA directory does not exist: {path}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=source,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"],
    ))


def _lora_rank(path: Path, supported_ranks: tuple[int, ...]) -> int:
    """Check adapter files and choose the smallest vLLM rank covering every layer."""
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise ValueError(f"Expected a PEFT LORA adapter configuration: {config_path}")
    rank_pattern = config.get("rank_pattern")
    if rank_pattern is None:
        rank_pattern = {}
    if not isinstance(rank_pattern, dict):
        raise ValueError(f"LoRA rank_pattern must be an object: {config_path}")
    ranks = [config.get("r"), *rank_pattern.values()]
    if any(type(rank) is not int or rank < 1 for rank in ranks):
        raise ValueError(f"LoRA r and rank_pattern values must be positive integers: {config_path}")
    required_rank = max(ranks)
    rank = next((value for value in sorted(supported_ranks) if value >= required_rank), None)
    if rank is None:
        raise ValueError(f"LoRA rank {required_rank} exceeds the supported vLLM limits {supported_ranks}.")
    if not any(
        weight.is_file() and weight.stat().st_size > 0
        for weight in (path / "adapter_model.safetensors", path / "adapter_model.bin")
    ):
        raise ValueError(f"Missing or empty LoRA weights in {path}; expected adapter_model.safetensors or .bin.")
    return rank


class VLLMCaptionGenerator:
    def __init__(self, args: Args):
        # Set before importing vLLM/Transformers, also for spawned worker processes.
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG" if args.debug else "WARNING"
        os.environ["TRANSFORMERS_VERBOSITY"] = "debug" if args.debug else "warning"
        try:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
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
        self.sampling_params_class = SamplingParams
        self.structured_outputs = StructuredOutputsParams(
            structural_tag=json.dumps(cosmos_structural_tag()),
        )
        self.lora_request = None
        lora_options = {}
        if args.lora_path is not None:
            from vllm.config.lora import MaxLoRARanks
            from vllm.lora.request import LoRARequest

            path = _resolve_lora(args.lora_path)
            rank = _lora_rank(path, get_args(MaxLoRARanks))
            self.lora_request = LoRARequest("caption_adapter", 1, str(path))
            lora_options = {"enable_lora": True, "max_loras": 1, "max_lora_rank": rank}
            log.debug("Using LoRA %s with max_lora_rank=%d", path, rank)
        # Text-only tokenization avoids initializing an image/video processor.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=False)
        self.prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": args.prompt_template},
                {"role": "user", "content": USER_PROMPT},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Keep the text engine optimizations from caption_from_image_nohttp,
        # while excluding the visual encoder and other multimodal modules.
        self.model = LLM(
            model=args.model,
            tensor_parallel_size=gpu_count,
            distributed_executor_backend="mp",
            structured_outputs_config={"backend": "xgrammar"},
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
            **lora_options,
        )
        log.debug("Using vLLM tensor_parallel_size=%d, MTP tokens=%d, GPU memory budget=%.0f%%",
                  gpu_count, args.spec_draft_n_max, args.gpu_memory_utilization * 100)

    def generate(self, count: int) -> list[CompletionOutput | Exception]:
        if count < 1:
            return []
        sampling_params = self.sampling_params_class.from_optional(
            n=1, max_tokens=self.args.max_tokens,
            temperature=self.args.temperature, top_p=self.args.top_p,
            top_k=self.args.top_k, min_p=self.args.min_p,
            presence_penalty=self.args.presence_penalty,
            repetition_penalty=self.args.repetition_penalty,
            ignore_eos=False,
            skip_special_tokens=True,
            structured_outputs=self.structured_outputs,
        )
        # No per-request fixed seed: repeated prompts must sample independently.
        options = {"lora_request": self.lora_request} if self.lora_request is not None else {}
        outputs = self.model.generate(
            [{"prompt": self.prompt} for _ in range(count)],
            sampling_params=sampling_params, use_tqdm=False, **options,
        )
        if len(outputs) != count:
            raise ValueError("The model returned a different number of captions than requested.")
        results: list[CompletionOutput | Exception] = []
        for request in outputs:
            try:
                if not request.finished or len(request.outputs) != 1:
                    raise ValueError("Expected one finished vLLM completion per request.")
                results.append(request.outputs[0])
            except (AttributeError, TypeError, ValueError) as exc:
                results.append(exc)
        return results


def _caption_json(args: Args, text: str) -> dict:
    """Validate the two model sections and assemble the canonical T2I object."""
    sections = re.fullmatch(
        r"\s*<scene_draft>(.*?)</scene_draft>\s*"
        r"<comprehensive_t2i_caption>(.*?)</comprehensive_t2i_caption>\s*",
        text, re.DOTALL,
    )
    if sections is None or not all(section.strip() for section in sections.groups()):
        raise ValueError("Expected both nonempty Cosmos sections in order, with no surrounding text.")
    scene = parse_json_response(sections[1])
    if not isinstance(scene, dict):
        raise ValueError("Invalid scene_draft JSON: expected an object.")
    validate_scene_draft(scene)
    scene["comprehensive_t2i_caption"] = sections[2].strip()
    return _apply_t2i_output_parameters(scene, resolution=args.resolution, aspect_ratio=args.aspect_ratio)


def _save_caption(args: Args, index: int, completion: CompletionOutput, attempt: int) -> None:
    path = _caption_path(args, index)
    if args.debug:
        debug_dir = args.output / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"{path.stem}.attempt-{attempt}.txt").write_text(
            json.dumps({"text": completion.text, "token_count": len(completion.token_ids),
                        "truncated": completion.finish_reason == "length", "finish_reason": completion.finish_reason},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if completion.finish_reason == "length":
        raise ValueError(
            f"Caption was truncated after {len(completion.token_ids)} tokens "
            f"(max_tokens={args.max_tokens}); increase --max-tokens."
        )
    if completion.finish_reason != "stop":
        raise ValueError(f"Completion did not finish normally: {completion.finish_reason!r}.")
    text = json.dumps(_caption_json(args, completion.text), indent=2, ensure_ascii=JSON_ENSURE_ASCII) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", encoding="utf-8", newline="\n")
    try:
        with stream:
            stream.write(text)
    except BaseException:
        # A failed write/flush must not leave a partial JSON for downstream T2I.
        path.unlink(missing_ok=True)
        raise


def generate_captions(args: Args) -> list[str]:
    """Load one model, save valid captions and return the names that exhausted retries."""
    _check_output_paths(args)
    failed: list[str] = []
    succeeded = 0
    with logging_redirect_tqdm(), tqdm(
        total=args.n, desc="Loading model", unit="caption", dynamic_ncols=True,
        mininterval=0.5, postfix={"ok": 0, "failed": 0},
    ) as progress:
        backend = VLLMCaptionGenerator(args)
        progress.set_description("Generating captions")
        for start in range(1, args.n + 1, args.batch_size):
            pending = list(range(start, min(start + args.batch_size, args.n + 1)))
            for attempt in range(1, args.max_retries + 1):
                if not pending:
                    break
                try:
                    results = backend.generate(len(pending))
                    if len(results) != len(pending):
                        raise ValueError("The backend returned a different number of captions than requested.")
                except Exception as exc:
                    results = [exc] * len(pending)
                retry = []
                for index, result in zip(pending, results):
                    path = _caption_path(args, index)
                    try:
                        if isinstance(result, Exception):
                            raise result
                        _save_caption(args, index, result, attempt)
                    except Exception as exc:
                        log.debug("%s: attempt %d/%d failed: %s", path.name, attempt, args.max_retries,
                                  exc, exc_info=args.debug)
                        if attempt < args.max_retries:
                            retry.append(index)
                            continue
                        log.error("Cannot generate %s after %d attempts: %s", path.name, attempt, exc)
                        failed.append(path.name)
                    else:
                        succeeded += 1
                    progress.set_postfix(ok=succeeded, failed=len(failed), refresh=False)
                    progress.update(1)
                pending = retry
    if failed:
        log.error("Failed captions: %s", ", ".join(failed))
    return failed


def cli_main(argv: list[str] | None = None) -> int:
    args = tyro.cli(Args, args=argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s" if args.debug else "%(levelname)s: %(message)s",
    )
    return 1 if generate_captions(args) else 0


if __name__ == "__main__":
    sys.exit(cli_main())
