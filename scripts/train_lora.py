"""Train a text-only Cosmos LoRA adapter from separately split caption files."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

import tyro


_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

log = logging.getLogger(__name__)


@dataclass
class Args:
    train: list[Path]
    """Training files or directories, already split by image origin/group."""
    validation: list[Path]
    """Validation files or directories, separate from training."""
    output_dir: Path = Path("checkpoints/cosmos-lora")
    """Checkpoints and metrics directory; the final adapter goes in adapter/."""
    max_length: int = 6144
    """Maximum complete chat length. Oversized examples fail rather than truncate."""
    batch_size: int = 4
    """Direct training batch size per GPU. Gradient accumulation is fixed at 1."""
    eval_batch_size: int = 2
    epochs: float | None = None
    """Maximum epochs; None trains until early stopping unless max_steps is positive."""
    max_steps: int = -1
    """Positive values override epochs; -1 uses epochs or early stopping when epochs=None."""
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_total_limit: int = 2
    early_stopping_patience: int = 3
    """Stop after this many evaluations without improvement in validation loss."""
    early_stopping_threshold: float = 0.001
    """Minimum absolute validation-loss improvement that resets early stopping."""
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    field_length_exponent: float = 0.25
    structure_weight: float = 0.05
    seed: int = 2026
    system_prompt_path: Path | None = None
    """Override src/templates/generate_caption.txt with a UTF-8 file."""
    resume_from_checkpoint: Path | None = None
    """Trainer checkpoint directory, including optimizer and trainer state."""
    run_name: str | None = None
    wandb_project: str = "sdg-cosmos3"
    preflight_only: bool = False
    """Load only the tokenizer and validate both datasets, without training."""
    train_paths: list[Path] = field(init=False, repr=False)
    validation_paths: list[Path] = field(init=False, repr=False)
    system_prompt: str = field(init=False, repr=False)

    def __post_init__(self):
        if not self.wandb_project.strip():
            raise ValueError("wandb_project must be nonempty.")
        for name in ("max_length", "batch_size", "eval_batch_size", "max_steps", "logging_steps",
                     "save_total_limit", "early_stopping_patience", "lora_rank",
                     "lora_alpha", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        for name in ("batch_size", "eval_batch_size", "logging_steps",
                     "save_total_limit", "early_stopping_patience", "lora_rank", "lora_alpha"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1.")
        if self.max_length < 2:
            raise ValueError("max_length must be >= 2.")
        if self.max_steps != -1 and self.max_steps < 1:
            raise ValueError("max_steps must be -1 or >= 1.")
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")
        if self.epochs is not None and (
            isinstance(self.epochs, bool) or not math.isfinite(self.epochs) or self.epochs <= 0
        ):
            raise ValueError("epochs must be None or finite and > 0.")
        for name in ("learning_rate", "structure_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0.")
        for name in ("weight_decay", "field_length_exponent", "early_stopping_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0.")
        if not 0 <= self.warmup_ratio <= 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("warmup_ratio must be in [0, 1]; lora_dropout in [0, 1).")
        if self.run_name is not None and not self.run_name.strip():
            raise ValueError("run_name must be nonempty when supplied.")

        for split in ("train", "validation"):
            paths = []
            seen = set()
            for source in getattr(self, split):
                source = Path(source)
                if source.is_dir():
                    candidates = sorted(
                        path for path in source.rglob("*")
                        if path.is_file() and path.suffix.lower() in (".xml", ".json", ".txt")
                        and "debug" not in path.relative_to(source).parts
                    )
                elif source.is_file():
                    candidates = [source]
                else:
                    raise ValueError(f"Caption path does not exist: {source}")
                if not candidates:
                    raise ValueError(f"No caption files found in {source}.")
                for path in candidates:
                    path = path.resolve()
                    if path in seen:
                        raise ValueError(f"Repeated caption path: {path}")
                    seen.add(path)
                    paths.append(path)
            if not paths:
                raise ValueError("Supply at least one caption file or directory per split.")
            setattr(self, f"{split}_paths", paths)
        overlap = set(self.train_paths) & set(self.validation_paths)
        if overlap:
            raise ValueError(f"Train/validation overlap: {sorted(overlap)[0]}")

        self.output_dir = Path(self.output_dir).resolve()
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"Output path must be a directory: {self.output_dir}")
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint).resolve()
            if not (self.resume_from_checkpoint / "trainer_state.json").is_file():
                raise ValueError("resume_from_checkpoint must contain trainer_state.json.")
        elif not self.preflight_only and self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise ValueError(
                f"Output directory is not empty: {self.output_dir}. Choose a new directory or resume a checkpoint."
            )
        if self.system_prompt_path is not None:
            self.system_prompt_path = Path(self.system_prompt_path).resolve()
        template = self.system_prompt_path or _PACKAGE_DIR / "src/templates/generate_caption.txt"
        self.system_prompt = template.read_text(encoding="utf-8-sig")
        if not self.system_prompt.strip():
            raise ValueError(f"System prompt file is empty: {template}")


def load_model(args: Args):
    """Import Unsloth before Transformers/TRL and pin one model to each GPU process."""
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    # Unsloth must patch the libraries before the local Trainer is imported.
    from unsloth import FastModel
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires a CUDA GPU. Use --preflight-only to validate data on CPU.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1 and torch.cuda.device_count() > 1:
        raise ValueError("Select one GPU with CUDA_VISIBLE_DEVICES or launch one process per GPU with torchrun.")
    torch.cuda.set_device(local_rank)
    bf16 = torch.cuda.is_bf16_supported()
    model, tokenizer = FastModel.from_pretrained(
        model_name="Qwen/Qwen3.8-27B",
        max_seq_length=args.max_length,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        text_only=True,
        return_logits=True,
        fast_inference=False,
        device_map={"": local_rank},
    )
    model = FastModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.config.use_cache = False
    # The custom loss implements language loss only, including for MoE models.
    text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    if hasattr(text_config, "router_aux_loss_coef"):
        text_config.router_aux_loss_coef = 0.0
    FastModel.for_training(model)
    return model, tokenizer, bf16


def _prepare_datasets(args: Args, tokenizer):
    from src.lora.collator import CosmosDataCollator
    from src.lora.dataset import load_caption_dataset

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define a padding token or an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    options = dict(max_length=args.max_length, system_prompt=args.system_prompt)
    train_dataset = load_caption_dataset(args.train_paths, tokenizer, **options)
    validation_dataset = load_caption_dataset(args.validation_paths, tokenizer, **options)
    collator = CosmosDataCollator(tokenizer, max_length=args.max_length)
    summary = {}
    for name, dataset in (("train", train_dataset), ("validation", validation_dataset)):
        lengths = [len(ids) for ids in dataset["input_ids"]]
        summary[name] = {"examples": len(dataset), "min_tokens": min(lengths), "max_tokens": max(lengths)}
        collator([dataset[0]])
        log.info("%s: %d examples, %d-%d tokens", name, len(dataset), min(lengths), max(lengths))
    return train_dataset, validation_dataset, collator, summary


def train_lora(args: Args) -> dict:
    if args.preflight_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", use_fast=True)
        _, _, _, summary = _prepare_datasets(args, tokenizer)
        return summary

    output_dir = args.output_dir
    os.environ["WANDB_PROJECT"] = args.wandb_project
    model, tokenizer, bf16 = load_model(args)
    train_dataset, validation_dataset, collator, summary = _prepare_datasets(args, tokenizer)

    from trl import SFTConfig
    from transformers import EarlyStoppingCallback
    from src.lora.checkpointing import RecoveryCheckpointCallback
    from src.lora.trainer import Trainer

    early_stopping_only = args.epochs is None and args.max_steps < 0
    config = SFTConfig(
        output_dir=str(output_dir),
        run_name=args.run_name or output_dir.name,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        # Trainer requires a finite step budget even when the callback decides
        # when to stop. This sentinel supplies one without a practical epoch cap.
        num_train_epochs=args.epochs if args.epochs is not None else 0.0,
        max_steps=sys.maxsize if early_stopping_only else args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        weight_decay=args.weight_decay,
        lr_scheduler_type="constant_with_warmup" if early_stopping_only else "linear",
        optim="adamw_torch",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        eval_on_start=False,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        restore_callback_states_from_checkpoint=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        packing=False,
        eval_packing=False,
        padding_free=False,
        loss_type="nll",
        label_smoothing_factor=0.0,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=["wandb"],
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    # Explicit steps work with both APIs: newer Transformers removed
    # warmup_ratio and interpret warmup_steps >= 1 as an absolute count.
    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * config.world_size))
    # Without a planned end, warm up over a fraction of the first epoch, then
    # keep the learning rate constant until the early-stopping callback fires.
    if early_stopping_only:
        total_steps = steps_per_epoch
    elif args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = math.ceil(args.epochs * steps_per_epoch)
    config.warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    trainer = Trainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        field_length_exponent=args.field_length_exponent,
        structure_weight=args.structure_weight,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            ),
        ],
    )
    trainer.add_callback(RecoveryCheckpointCallback(trainer))
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(json.dumps({
            "model": "Qwen/Qwen3.8-27B", "load_in_4bit": False, "precision": "auto",
            "args": {item.name: getattr(args, item.name) for item in fields(args) if item.init},
            "train_paths": args.train_paths, "validation_paths": args.validation_paths, "data": summary,
        }, default=str, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result = trainer.train(resume_from_checkpoint=(str(args.resume_from_checkpoint)
                                                  if args.resume_from_checkpoint else None))
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    # Restore the best checkpoint when at least one was saved during training.
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(adapter_dir))
    return {"data": summary, "train": result.metrics, "eval": metrics, "adapter_dir": str(adapter_dir)}


def cli_main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = tyro.cli(Args, args=argv)
        result = train_lora(args)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        log.error("%s", error)
        return 1
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
