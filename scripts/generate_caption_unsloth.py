"""Generate Cosmos captions with Unsloth/PEFT, without a vLLM inference engine."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import tyro
from jsonschema import ValidationError
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(_PACKAGE_DIR))

# These helpers only serialize/validate files; importing them does not import vLLM.
from scripts.generate_caption_nohttp import USER_PROMPT, _check_output_paths, _save_caption
from src.caption.output_parameters import OutputParameters

log = logging.getLogger(__name__)
MODEL_ID = "unsloth/Qwen3.8-27B"
ADAPTER_NAME = "caption_adapter"
END_TAG = "</comprehensive_t2i_caption>"


@dataclass
class Args:
    """Generate captions sequentially on one CUDA GPU with Unsloth and PEFT."""

    n: Annotated[int, tyro.conf.arg(aliases=["-n"])]
    output: Annotated[Path, tyro.conf.arg(aliases=["-o"])]
    lora_path: Annotated[Path | None, tyro.conf.arg(aliases=["--lora_path"])] = None
    """Local original PEFT adapter/checkpoint with adapter_model.safetensors; omit for baseline."""
    max_tokens: Annotated[int, tyro.conf.arg(aliases=["--max_tokens"])] = 3072
    max_length: Annotated[int, tyro.conf.arg(aliases=["--max_length"])] = 6144
    """Context budget, including prompt and generated tokens."""
    max_retries: Annotated[int, tyro.conf.arg(aliases=["--max_retries"])] = 3
    temperature: float = 0.7
    top_p: Annotated[float, tyro.conf.arg(aliases=["--top_p"])] = 0.8
    top_k: Annotated[int, tyro.conf.arg(aliases=["--top_k"])] = 20
    min_p: Annotated[float, tyro.conf.arg(aliases=["--min_p"])] = 0.0
    presence_penalty: Annotated[float, tyro.conf.arg(aliases=["--presence_penalty"])] = 1.5
    repetition_penalty: Annotated[float, tyro.conf.arg(aliases=["--repetition_penalty"])] = 1.0
    seed: int = 3407
    """A different reproducible seed is derived for every caption and attempt."""
    resolution: str = "720"
    aspect_ratio: Annotated[str, tyro.conf.arg(aliases=["--aspect_ratio"])] = "16,9"
    prompt_template_path: Annotated[Path | None, tyro.conf.arg(aliases=["--prompt_template_path"])] = None
    debug: bool = False
    prompt_template: str = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("n", "max_tokens", "max_length", "max_retries"):
            if getattr(self, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 1.")
        if self.max_tokens >= self.max_length:
            raise ValueError("--max-length must leave room for both the prompt and --max-tokens.")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("--temperature must be finite and >= 0.")
        if not 0 < self.top_p <= 1 or self.top_k < 0 or not 0 <= self.min_p <= 1:
            raise ValueError("Invalid --top-p, --top-k or --min-p.")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("--presence-penalty must be in [-2, 2].")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("--repetition-penalty must be finite and > 0.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("--seed must be in [0, 2**32).")
        self.output = Path(self.output)
        _check_output_paths(self)
        OutputParameters(self.resolution, self.aspect_ratio)
        template = self.prompt_template_path or _PACKAGE_DIR / "src/templates/generate_caption.txt"
        self.prompt_template = Path(template).read_text(encoding="utf-8-sig")
        if not self.prompt_template.strip():
            raise ValueError("Prompt template is empty.")
        if self.lora_path is not None:
            self.lora_path = Path(self.lora_path).expanduser().resolve()
            config = json.loads((self.lora_path / "adapter_config.json").read_text(encoding="utf-8-sig"))
            if config.get("peft_type") != "LORA":
                raise ValueError("Expected a PEFT LORA adapter.")
            weights = self.lora_path / "adapter_model.safetensors"
            if not weights.is_file() or weights.stat().st_size == 0:
                raise ValueError(f"Missing or empty adapter weights: {weights}")


@dataclass
class Completion:
    text: str
    token_ids: list[int]
    finish_reason: str


class PresencePenalty:
    """Penalize each token seen in the generated continuation once, excluding the prompt."""

    def __init__(self, prompt_length: int, penalty: float):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            seen = input_ids[row, self.prompt_length:].unique()
            scores[row, seen] -= self.penalty
        return scores


def verify_adapter_weights(model, path: Path) -> dict:
    """Require an enabled adapter and an exact match against every saved tensor."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    status = model.get_model_status()
    if (status.enabled is not True or status.active_adapters != [ADAPTER_NAME]
            or status.merged_adapters or status.num_adapter_layers < 1):
        raise RuntimeError(f"LoRA is not consistently active: {status}")
    actual = get_peft_model_state_dict(model, adapter_name=ADAPTER_NAME, save_embedding_layers=False)
    nonzero_b = 0
    with safe_open(str(path / "adapter_model.safetensors"), framework="pt", device="cpu") as saved:
        expected_keys = set(saved.keys())
        if not expected_keys or expected_keys != set(actual):
            missing = sorted(expected_keys - set(actual))[:3]
            extra = sorted(set(actual) - expected_keys)[:3]
            raise RuntimeError(f"LoRA key mismatch: missing={missing}, unexpected={extra}.")
        for name in sorted(expected_keys):
            loaded = actual[name].detach().cpu()
            expected = saved.get_tensor(name).to(dtype=loaded.dtype)
            if not torch.isfinite(loaded).all() or not torch.equal(loaded, expected):
                raise RuntimeError(f"Loaded LoRA tensor differs from checkpoint: {name}")
            if ".lora_B." in name and torch.count_nonzero(loaded).item():
                nonzero_b += 1
    if nonzero_b == 0:
        raise RuntimeError("All LoRA B tensors are zero; this adapter cannot change predictions.")
    return {"matched_tensors": len(actual), "active_layers": status.num_adapter_layers,
            "active_adapters": status.active_adapters, "nonzero_lora_b_tensors": nonzero_b}


def verify_adapter_effect(model, inputs) -> dict:
    """Compare next-token logits with/without LoRA against repeated-baseline noise."""
    import torch

    def logits():
        result = model(**inputs, use_cache=False, logits_to_keep=1)
        values = result.logits[:, -1, :].detach().float().cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError("Non-finite logits during adapter verification.")
        return values

    with torch.inference_mode():
        with model.disable_adapter():
            base = logits()
        adapted = logits()
        with model.disable_adapter():
            repeated = logits()
    delta = (adapted - base).abs().max().item()
    noise = (repeated - base).abs().max().item()
    if delta <= max(2 * noise, 1e-6):
        raise RuntimeError(
            f"Cannot establish a LoRA effect: max_logit_delta={delta:g}, baseline_noise={noise:g}."
        )
    return {"max_logit_delta": delta, "baseline_noise": noise}


class UnslothCaptionGenerator:
    def __init__(self, args: Args):
        # Unsloth must be imported before Transformers or PEFT.
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from unsloth import FastModel
        import torch
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Unsloth caption inference requires a CUDA GPU.")
        self.args = args
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        log.info("Loading %s on cuda:0 (%s), fast_inference=False.", MODEL_ID, dtype)
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=MODEL_ID, max_seq_length=args.max_length, dtype=dtype,
            load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
            text_only=True, fast_inference=False, device_map={"": 0},
        )
        verification = None
        if args.lora_path is not None:
            from peft import PeftModel

            log.info("Loading PEFT adapter: %s", args.lora_path)
            self.model = PeftModel.from_pretrained(
                self.model, str(args.lora_path), adapter_name=ADAPTER_NAME, is_trainable=False,
            )
            self.model.set_adapter(ADAPTER_NAME)
            if (args.lora_path / "tokenizer_config.json").is_file():
                self.tokenizer = AutoTokenizer.from_pretrained(str(args.lora_path), use_fast=True)
        FastModel.for_inference(self.model)
        self.model.eval()
        self.prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": args.prompt_template},
             {"role": "user", "content": USER_PROMPT}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        self.inputs = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False)
        self.inputs = self.inputs.to(self.model.get_input_embeddings().weight.device)
        self.prompt_length = self.inputs["input_ids"].shape[-1]
        if self.prompt_length + args.max_tokens > args.max_length:
            raise ValueError(
                f"Prompt ({self.prompt_length}) + output ({args.max_tokens}) exceeds "
                f"--max-length ({args.max_length})."
            )
        if args.lora_path is not None:
            verification = verify_adapter_weights(self.model, args.lora_path)
            verification.update(verify_adapter_effect(self.model, self.inputs))
            log.info("LoRA VERIFIED: %d tensors, %d active layers; max logit delta %.6g (noise %.6g).",
                     verification["matched_tensors"], verification["active_layers"],
                     verification["max_logit_delta"], verification["baseline_noise"])
        else:
            log.info("BASELINE: %s without any LoRA adapter.", MODEL_ID)
        versions = {}
        for package in ("unsloth", "transformers", "peft", "torch"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "unknown"
        self.manifest = {
            "model": MODEL_ID, "backend": "unsloth/transformers", "fast_inference": False,
            "compile_disabled": True, "batch_size": 1, "dtype": str(dtype),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(args).items()
                     if k != "prompt_template"},
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "prompt_tokens": self.prompt_length, "adapter_verification": verification,
            "versions": versions,
        }

    def generate(self, seed: int) -> Completion:
        import torch
        from transformers import LogitsProcessorList, set_seed

        set_seed(seed)
        options = dict(
            max_new_tokens=self.args.max_tokens, do_sample=self.args.temperature > 0,
            repetition_penalty=self.args.repetition_penalty, use_cache=True,
            stop_strings=[END_TAG], tokenizer=self.tokenizer,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                          else self.tokenizer.eos_token_id),
            logits_processor=LogitsProcessorList([
                PresencePenalty(self.prompt_length, self.args.presence_penalty),
            ]),
        )
        if self.args.temperature > 0:
            options.update(temperature=self.args.temperature, top_p=self.args.top_p,
                           top_k=self.args.top_k, min_p=self.args.min_p)
        with torch.inference_mode():
            sequences = self.model.generate(**self.inputs, **options)
        tokens = sequences[0, self.prompt_length:].tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        stopped = END_TAG in text or bool(tokens and tokens[-1] in eos_ids)
        return Completion(text, tokens, "stop" if stopped else "length")


def generate_captions(args: Args) -> list[str]:
    _check_output_paths(args)
    backend = UnslothCaptionGenerator(args)
    diagnostics = args.output / "debug"
    diagnostics.mkdir(parents=True, exist_ok=True)
    # Keep run metadata out of the T2I input JSONs in the output directory.
    (diagnostics / "inference_manifest.txt").write_text(
        json.dumps(backend.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    failed = []
    with logging_redirect_tqdm(), tqdm(total=args.n, desc="Generating captions", unit="caption") as progress:
        for index in range(1, args.n + 1):
            for attempt in range(1, args.max_retries + 1):
                seed = (args.seed + (index - 1) * args.max_retries + attempt - 1) % 2**32
                try:
                    result = backend.generate(seed)
                    _save_caption(args, index, result, attempt)
                except (ValueError, RuntimeError, ValidationError) as exc:
                    log.warning("caption_%06d attempt %d/%d (seed=%d): %s",
                                index, attempt, args.max_retries, seed, exc)
                    if attempt == args.max_retries:
                        failed.append(f"caption_{index:06d}.json")
                else:
                    break
            progress.update(1)
            progress.set_postfix(ok=index - len(failed), failed=len(failed))
    return failed


def cli_main(argv: list[str] | None = None) -> int:
    args = tyro.cli(Args, args=argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return 1 if generate_captions(args) else 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        log.error("Inference stopped: %s", exc, exc_info=args.debug)
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())
