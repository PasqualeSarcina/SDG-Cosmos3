"""Local image captioning using vLLM tensor parallelism across visible CUDA GPUs."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from vllm.outputs import CompletionOutput

import tyro
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(_PACKAGE_DIR))

from src.caption.structured_caption import (
    assemble_caption_json,
    extract_xml_tag,
    media_fields_from_metadata,
    parse_structured_caption,
)
from src.request.response_parsing import parse_json_response

log = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class Args:
    """Caption local images using vLLM on all visible CUDA GPUs."""

    input: Annotated[Path, tyro.conf.arg(aliases=["-i"])]
    """Image file or directory; subdirectories are ignored."""
    output: Annotated[Path, tyro.conf.arg(aliases=["-o"])]
    """Write one <original-image-stem>.json per image."""
    model: str = "Qwen/Qwen3.8-27B"
    """Local VLM checkpoint directory or Hugging Face ID; missing files are downloaded and cached."""
    batch_size: Annotated[int, tyro.conf.arg(aliases=["--batch_size"])] = 16
    """Independent images generated together on the selected GPUs."""
    max_workers: Annotated[int, tyro.conf.arg(aliases=["--max_workers"])] = 4
    """CPU threads for reading images, not concurrent generation calls."""
    max_retries: Annotated[int, tyro.conf.arg(aliases=["--max_retries"])] = 3
    """Maximum attempts per image, including format-validation failures."""
    gpu_memory_utilization: Annotated[float, tyro.conf.arg(aliases=["--gpu_memory_utilization"])] = 0.8
    """Fraction of each GPU's memory reserved for the vLLM engine, in (0, 1]."""
    spec_draft_n_max: int = 2
    """MTP speculative tokens per step; use 0 for checkpoints without MTP weights/support."""
    # Qwen/Qwen3.8-27B official Instruct (non-thinking) sampling defaults.
    temperature: float = 0.7
    """Sampling temperature; zero selects greedy generation."""
    top_p: Annotated[float, tyro.conf.arg(aliases=["--top_p"])] = 0.8
    top_k: Annotated[int, tyro.conf.arg(aliases=["--top_k"])] = 20
    min_p: Annotated[float, tyro.conf.arg(aliases=["--min_p"])] = 0.0
    presence_penalty: Annotated[float, tyro.conf.arg(aliases=["--presence_penalty"])] = 1.5
    """Additive penalty for tokens already generated in this caption, excluding the prompt."""
    repetition_penalty: Annotated[float, tyro.conf.arg(aliases=["--repetition_penalty"])] = 1.0
    prompt_template_path: Annotated[Path | None, tyro.conf.arg(aliases=["--prompt_template_path"])] = Path(_PACKAGE_DIR, "src", "templates", "cosmos_image_captioner.txt")
    """Custom UTF-8 instruction; otherwise use the template for the selected format."""
    enforce_cosmos_format: Annotated[bool, tyro.conf.arg(aliases=["--enforce_cosmos_format"])] = True
    """Expect Cosmos XML/JSON; disable for the flat description JSON schema."""
    debug: bool = False
    """Enable verbose logs and save generated text and token counts under output/debug/."""
    image_items: list[Path] = field(init=False, repr=False)
    prompt_template: str = field(init=False, repr=False)
    description_fields: set[str] = field(init=False, default_factory=set, repr=False)

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_workers", "max_tokens", "max_seq_length", "max_retries"):
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
        self.input = Path(self.input)
        self.output = Path(self.output)
        if self.output.exists() and not self.output.is_dir():
            raise ValueError(f"Output path must be a directory: {self.output}")
        if self.input.is_dir():
            self.image_items = sorted(
                (p for p in self.input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                key=lambda p: (p.name.casefold(), p.name),
            )
        elif self.input.is_file() and self.input.suffix.lower() in IMAGE_EXTENSIONS:
            self.image_items = [self.input]
        else:
            raise ValueError(f"Image path is missing or unsupported: {self.input}")
        if not self.image_items:
            raise ValueError(f"No image inputs found in {self.input}.")
        names: set[str] = set()
        for path in self.image_items:
            if path.stem.casefold() in names:
                raise ValueError(f"Ambiguous original image name: {path.stem!r}.")
            names.add(path.stem.casefold())
        template_name = "cosmos_image_captioner.txt" if self.enforce_cosmos_format else "simple_captioner.txt"
        template_path = Path(self.prompt_template_path or _PACKAGE_DIR / "src/templates" / template_name)
        self.prompt_template = template_path.read_text(encoding="utf-8-sig")
        if not self.prompt_template.strip():
            raise ValueError(f"Prompt template file is empty: {template_path}")
        if not self.enforce_cosmos_format:
            template = json.loads(
                (_PACKAGE_DIR / "src/schema/description_json_schema.json").read_text(encoding="utf-8-sig")
            )
            self.description_fields = set(template)
            # Inference stays local; JSON is validated after generation.
            self.prompt_template += (
                "\nReturn only a JSON object with exactly the following keys, all values strings. "
                "This specification takes precedence over any conflicting example above:\n"
                + json.dumps(template, ensure_ascii=False)
            )


class VLLMCaptioner:
    def __init__(self, args: Args):
        # Set before importing vLLM or the Hugging Face libraries, including worker processes.
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
        from transformers import AutoProcessor

        for logger_name in ("vllm", "transformers", "huggingface_hub"):
            logging.getLogger(logger_name).setLevel(logging.DEBUG if args.debug else logging.WARNING)
        gpu_count = torch.cuda.device_count()
        if not torch.cuda.is_available() or gpu_count < 1:
            raise RuntimeError("This script requires at least one CUDA GPU visible to PyTorch/vLLM.")
        self.args = args
        self.sampling_params_class = SamplingParams
        self.processor = AutoProcessor.from_pretrained(args.model, local_files_only=False)
        self.model = LLM(
            model=args.model,
            tensor_parallel_size=gpu_count,
            distributed_executor_backend="mp",
            gpu_memory_utilization=args.gpu_memory_utilization,
            language_model_only=False,
            mm_encoder_tp_mode=(
                "data" if gpu_count > 1 else "weights"
            ),
            # Cache multimodal
            mm_processor_cache_type=(
                "shm" if gpu_count > 1 else "lru"
            ),
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            compilation_config=3,
            enforce_eager=False,
            performance_mode="throughput",
            # Logging
            disable_log_stats=not args.debug,
            generation_config="auto",
            speculative_config=(
                {
                    "method": "mtp",
                    "num_speculative_tokens": args.spec_draft_n_max
                }
                if args.spec_draft_n_max
                else None
            ),
        )

        log.debug("Using vLLM tensor_parallel_size=%d, MTP tokens=%d, GPU memory budget=%.0f%%",
                  gpu_count, args.spec_draft_n_max, args.gpu_memory_utilization * 100)

    def generate(self, images: list[Image.Image], prompts: list[str]) -> list[CompletionOutput | Exception]:
        if len(images) != len(prompts):
            raise ValueError("Each input image must have exactly one prompt.")
        if not images:
            return []
        texts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [
                # Put the shared instruction before the unique image for prefix-cache reuse.
                {"type": "text", "text": prompt}, {"type": "image"},
            ]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ) for prompt in prompts]
        # vLLM handles image preprocessing, context admission and continuous batching.
        # Do not also tokenize images with the HF processor or transfer them to CUDA here.
        requests = [{"prompt": text, "multi_modal_data": {"image": image}}
                    for text, image in zip(texts, images)]
        sampling_params = self.sampling_params_class.from_optional(
            n=1,
            temperature=self.args.temperature, top_p=self.args.top_p,
            top_k=self.args.top_k, min_p=self.args.min_p,
            presence_penalty=self.args.presence_penalty,
            repetition_penalty=self.args.repetition_penalty,
            ignore_eos=False,
            skip_special_tokens=True,
        )
        outputs = self.model.generate(requests, sampling_params=sampling_params, use_tqdm=False)
        if len(outputs) != len(images):
            raise ValueError("The model returned a different number of captions than input images.")
        results: list[CompletionOutput | Exception] = []
        for request in outputs:
            try:
                if not request.finished or len(request.outputs) != 1:
                    raise ValueError("Expected one finished vLLM completion per image.")
                output = request.outputs[0]
                if output.finish_reason not in {"stop", "length"}:
                    raise ValueError(f"vLLM completion failed: finish_reason={output.finish_reason!r}.")
                results.append(output)
            except (AttributeError, TypeError, ValueError) as exc:
                results.append(exc)
        return results


def _read_image(path: Path) -> Image.Image | Exception:
    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except (OSError, ValueError) as exc:
        return exc


def _caption_data(args: Args, text: str, size: tuple[int, int]) -> dict:
    if not text.strip():
        raise ValueError("The model generated an empty caption.")
    if not args.enforce_cosmos_format:
        data = parse_json_response(text)
        if not isinstance(data, dict) or set(data) != args.description_fields:
            raise ValueError("Caption must contain exactly the description schema fields.")
        if any(not isinstance(value, str) for value in data.values()):
            raise ValueError("Description schema fields must be strings.")
        return data
    narrative = extract_xml_tag(text, "comprehensive_t2i_caption")
    scene = parse_structured_caption(text)
    if narrative is None and scene is not None and extract_xml_tag(text, "scene_draft") is None:
        narrative = scene.get("comprehensive_t2i_caption")
    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError("Missing or empty comprehensive_t2i_caption tag or JSON field.")
    if scene is None:
        raise ValueError("Invalid scene_draft JSON.")
    return assemble_caption_json(scene, narrative, media_fields_from_metadata(size))


def _save_caption(args: Args, path: Path, completion: CompletionOutput, size: tuple[int, int], attempt: int) -> None:
    if args.debug:
        debug_dir = args.output / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"{path.stem}.attempt-{attempt}.json").write_text(
            json.dumps({"text": completion.text, "token_count": len(completion.token_ids),
                        "truncated": completion.finish_reason == "length", "finish_reason": completion.finish_reason},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if completion.finish_reason == "length":
        raise ValueError("Caption was truncated; increase token budget.")
    if completion.finish_reason != "stop":
        raise ValueError(f"Completion did not finish normally: {completion.finish_reason!r}.")
    data = _caption_data(args, completion.text, size)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{path.stem}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def caption_from_image(args: Args) -> list[str]:
    """Return failed image names; load one model and bound both CPU and GPU batches."""
    failed: list[str] = []
    succeeded = 0
    with logging_redirect_tqdm(), ThreadPoolExecutor(max_workers=args.max_workers) as pool, tqdm(
        total=len(args.image_items), desc="Loading model", unit="image", dynamic_ncols=True,
        mininterval=0.5, postfix={"ok": 0, "failed": 0},
    ) as progress:
        backend = VLLMCaptioner(args)
        progress.set_description("Captioning")
        log.debug("Captioning %d images locally with vLLM, model=%s, batch_size=%d, CPU workers=%d",
                  len(args.image_items), args.model, args.batch_size, args.max_workers)
        for start in range(0, len(args.image_items), args.batch_size):
            paths = args.image_items[start:start + args.batch_size]
            loaded = list(pool.map(_read_image, paths))
            try:
                pending = []
                for path, image in zip(paths, loaded):
                    if isinstance(image, Exception):
                        log.error("Cannot read %s: %s", path, image)
                        failed.append(path.name)
                        progress.set_postfix(ok=succeeded, failed=len(failed), refresh=False)
                        progress.update(1)
                    else:
                        pending.append((path, image, args.prompt_template))
                for attempt in range(1, args.max_retries + 1):
                    if not pending:
                        break
                    results = backend.generate(
                        [item[1] for item in pending], [item[2] for item in pending],
                    )
                    retry = []
                    for (path, image, _), result in zip(pending, results):
                        try:
                            if isinstance(result, Exception):
                                raise result
                            _save_caption(args, path, result, image.size, attempt)
                        except Exception as exc:
                            log.debug("%s: attempt %d/%d failed: %s", path.name, attempt, args.max_retries, exc,
                                      exc_info=args.debug)
                            if attempt == args.max_retries:
                                log.error("Cannot caption %s after %d attempts: %s", path.name, attempt, exc)
                                failed.append(path.name)
                                progress.set_postfix(ok=succeeded, failed=len(failed), refresh=False)
                                progress.update(1)
                            else:
                                retry.append((path, image, args.prompt_template))
                        else:
                            succeeded += 1
                            progress.set_postfix(ok=succeeded, failed=len(failed), refresh=False)
                            progress.update(1)
                    pending = retry
            finally:
                for image in loaded:
                    if isinstance(image, Image.Image):
                        image.close()
    log.debug("%d/%d images successfully captioned", succeeded, len(args.image_items))
    if failed:
        log.error("Failed images: %s", ", ".join(failed))
    return failed


def cli_main(argv: list[str] | None = None) -> int:
    args = tyro.cli(Args, args=argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s" if args.debug else "%(levelname)s: %(message)s",
    )
    return 1 if caption_from_image(args) else 0


if __name__ == "__main__":
    sys.exit(cli_main())
