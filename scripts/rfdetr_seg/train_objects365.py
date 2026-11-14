"""Train or fine-tune RF-DETR-Seg Large on Car boxes and instance masks.

Use the project's rfdetrn dependency group. Pass the output of
scripts/add_objects365_car_masks.py as --dataset-dir. It adds SAM3 masks to the
dataset from scripts/download_objects365_car.py, in the same train/valid
COCO layout.

For fine-tuning on COCO Car, use the same train/valid layout and pass
--pretrain-weights with an Objects365 RF-DETR checkpoint. Only model weights
are loaded; optimizer, EMA counters, early stopping and W&B start a new run.

This is a fixed-architecture adaptation of RF-DETR v2, Table 6 and Appendix A.
It keeps the public Seg Large preset's 200 queries and trains only Car. The
optimizer and EMA use the public RF-DETR implementation for details absent
from the paper.

--batch-size sets the microbatch on each GPU. The effective batch per optimizer
update is batch_size * devices * num_nodes * grad_accum_steps.

Use --wandb-project to log metrics to W&B. The run identity is saved in the
Lightning checkpoint, so --resume reconnects to it automatically.
"""

from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
from unittest.mock import patch

import tyro


RFDETR_VERSION = "1.11.2"
DINOV2_MODEL = "facebook/dinov2-small"
DINOV2_REVISION = "150c8e7bb7cef2d30ec31b13a517af14840ee3f7"
WINDOW_BLOCKS = (0, 1, 3, 4, 6, 7, 9, 10)
# Eleven symmetric scales within 0.5x–1.5x of 504, divisible by 12 * 2.
TRAIN_RESOLUTIONS = tuple(range(264, 745, 48))
WANDB_RUN_KEY = "objects365_wandb_run"


@dataclass
class Args:
    # Car dataset in Roboflow COCO train/valid layout.
    dataset_dir: Path
    # Directory for logs and checkpoints.
    output_dir: Path
    # Maximum number of training epochs.
    epochs: int = 60
    # Training images per GPU per microbatch.
    batch_size: int = 4
    # Microbatches accumulated before each optimizer update.
    grad_accum_steps: int = 1
    # Minimum absolute validation mAP improvement; zero accepts any strict increase.
    early_stopping_min_delta: float = 0.0
    # Consecutive validations without a new accepted best before stopping.
    early_stopping_patience: int = 10
    # Reset exhausted patience on this resume, keeping the actual historical best.
    reset_early_stopping: bool = False
    # Base learning rate.
    lr: float = 1e-4
    # Encoder learning rate before component and layer decay.
    lr_encoder: float = 1.5e-4
    # GPUs per node.
    devices: int = 1
    # Number of nodes.
    num_nodes: int = 1
    # Data workers per GPU.
    num_workers: int = 4
    # Random seed.
    seed: int = 42
    # W&B project for online metric logging.
    wandb_project: str | None = None
    # W&B account or team that owns the project.
    wandb_entity: str | None = None
    # Local Meta DINOv2-S/14 safetensors file.
    dinov2_weights: Path | None = None
    # RF-DETR .pth or Lightning .ckpt weights for a new fine-tuning run.
    pretrain_weights: Path | None = None
    # Lightning checkpoint to resume from.
    resume: Path | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("Epochs must be positive.")
        if min(self.batch_size, self.devices, self.num_nodes, self.grad_accum_steps,
               self.early_stopping_patience) < 1 or self.num_workers < 0:
            raise ValueError("Batch size, devices, nodes, accumulation and patience must be positive. Workers must be nonnegative.")
        if not math.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0:
            raise ValueError("Early stopping min_delta must be finite and nonnegative.")
        if self.reset_early_stopping and self.resume is None:
            raise ValueError("--reset-early-stopping requires --resume.")
        if not all(math.isfinite(lr) and lr > 0 for lr in (self.lr, self.lr_encoder)):
            raise ValueError("Learning rates must be positive and finite.")
        checkpoints = (self.dinov2_weights, self.pretrain_weights, self.resume)
        if sum(path is not None for path in checkpoints) > 1:
            raise ValueError("--dinov2-weights, --pretrain-weights and --resume are mutually exclusive.")
        for split in ("train", "valid"):
            annotations = self.dataset_dir / split / "_annotations.coco.json"
            if not annotations.is_file():
                raise ValueError(f"Missing {annotations}")
        for checkpoint in checkpoints:
            if checkpoint is not None and not checkpoint.is_file():
                raise ValueError(f"Missing {checkpoint}")
        if self.resume is not None and self.resume.suffix != ".ckpt":
            raise ValueError("--resume requires a Lightning .ckpt checkpoint.")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"--output-dir must be a directory: {self.output_dir}")
        if self.resume is None and (self.output_dir / "last.ckpt").exists():
            raise ValueError("The output directory already contains last.ckpt. Use --resume or a new directory.")


def make_wandb_logger(args: Args):
    identity = None
    if args.resume is not None:
        import torch

        # Map tensors without reading their contents. Lightning restores them during fit.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
            raise ValueError("--resume requires a full training checkpoint with optimizer and scheduler state, such as last.ckpt.")
        identity = checkpoint.get(WANDB_RUN_KEY)

    if identity is not None:
        for key, requested in (("project", args.wandb_project), ("entity", args.wandb_entity)):
            if requested is not None and requested != identity[key]:
                raise ValueError(f"--wandb-{key} must match the run recorded in the checkpoint.")
        settings = {
            "id": identity["id"],
            "project": identity["project"],
            "entity": identity["entity"],
            "resume": "must",
        }
    elif args.wandb_project:
        settings = {"project": args.wandb_project, "entity": args.wandb_entity, "resume": "never"}
    elif args.wandb_entity:
        raise ValueError("--wandb-entity requires --wandb-project or a checkpoint with a W&B run.")
    else:
        return None

    from pytorch_lightning.loggers import WandbLogger

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return WandbLogger(save_dir=str(args.output_dir), log_model=False, mode="online", **settings)


def load_dinov2(encoder, weights_path):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch.nn.functional import interpolate

    if weights_path is None:
        weights_path = hf_hub_download(DINOV2_MODEL, "model.safetensors", revision=DINOV2_REVISION)

    weights = load_file(str(weights_path))
    target = encoder.state_dict()
    patch_key = "embeddings.patch_embeddings.projection.weight"
    position_key = "embeddings.position_embeddings"
    patch_weight = weights[patch_key]
    position_weight = weights[position_key]
    if patch_weight.shape != (384, 3, 14, 14) or position_weight.shape != (1, 37**2 + 1, 384):
        raise ValueError("Expected the Meta DINOv2-S/14 backbone checkpoint in safetensors format.")

    # FlexiViT PI-resize: invert the image-resize operator for patch filters.
    patch_size = target[patch_key].shape[-1]
    basis = torch.eye(14**2, dtype=torch.float64).reshape(-1, 1, 14, 14)
    resize = interpolate(basis, size=(patch_size, patch_size), mode="bicubic", align_corners=False, antialias=True)
    inverse_resize = torch.linalg.pinv(resize.flatten(1).T)
    resized_patch = patch_weight.flatten(2).double() @ inverse_resize
    weights[patch_key] = resized_patch.reshape_as(target[patch_key]).to(patch_weight.dtype)

    # Resize only the spatial positions. Preserve the pretrained class token position.
    grid_size = math.isqrt(target[position_key].shape[1] - 1)
    grid = position_weight[:, 1:].reshape(1, 37, 37, 384).permute(0, 3, 1, 2)
    grid = interpolate(grid.float(), size=(grid_size, grid_size), mode="bicubic", align_corners=False, antialias=True)
    grid = grid.permute(0, 2, 3, 1).reshape(1, -1, 384).to(position_weight.dtype)
    weights[position_key] = torch.cat((position_weight[:, :1], grid), dim=1)
    encoder.load_state_dict(weights, strict=True)
    print(f"Initialized DINOv2-S/14 backbone from {weights_path}")


def validate_annotations(train, valid):
    for split, dataset in (("train", train), ("valid", valid)):
        categories = dataset.coco.cats
        if len(categories) != 1 or next(iter(categories.values()))["name"].casefold() != "car":
            raise ValueError(f"The {split} split must declare a single Car category.")
        category_id = next(iter(categories))
        if dataset.cat2label != {category_id: 0}:
            raise ValueError(f"The {split} Car category must map to model label 0.")
        if not dataset.coco.anns:
            raise ValueError(f"The {split} split has no instance annotations.")
        for annotation in dataset.coco.anns.values():
            if annotation["category_id"] != category_id or not annotation.get("segmentation"):
                raise ValueError(
                    f"Expected a Car annotation with an instance mask in {split} for ID {annotation['id']}. "
                    "Use the output of scripts/add_objects365_car_masks.py."
                )


def paper_scales(resolution, expanded_scales, patch_size, num_windows):
    return list(TRAIN_RESOLUTIONS)


def align_compiled_state_dict(module, state_dict):
    """Match live or EMA checkpoint keys to the target's torch.compile wrappers."""
    def uncompiled_key(key):
        return ".".join(part for part in key.split(".") if part != "_orig_mod")

    target_state = module.state_dict()
    target_keys = {uncompiled_key(key): key for key in target_state}
    aligned = OrderedDict()
    for key, value in state_dict.items():
        target_key = key if key in target_state else target_keys.get(uncompiled_key(key), key)
        if target_key in aligned:
            raise RuntimeError(f"Multiple checkpoint keys map to {target_key!r}.")
        aligned[target_key] = value
    if aligned.keys() == state_dict.keys():
        return state_dict

    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        module_keys = {uncompiled_key(name): name for name, _ in module.named_modules()}
        aligned._metadata = OrderedDict(
            (module_keys.get(uncompiled_key(key), key), value) for key, value in metadata.items()
        )
    return aligned


def load_pretrain_weights(model, weights_path):
    """Load only network weights, keeping the new run's training state fresh."""
    import torch

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected an RF-DETR checkpoint with 'model' or 'state_dict' weights.")
    if "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint.get("state_dict"), dict):
        # Lightning checkpoints include the module's 'model.' prefix and may
        # also contain criterion buffers, which are not network weights.
        lightning_state = checkpoint["state_dict"]
        prefix = "model."
        state = OrderedDict(
            (key.removeprefix(prefix), value)
            for key, value in lightning_state.items()
            if key.startswith(prefix)
        )
        metadata = getattr(lightning_state, "_metadata", None)
        if metadata is not None:
            state._metadata = OrderedDict(
                ("" if key == "model" else key.removeprefix(prefix), value)
                for key, value in metadata.items()
                if key == "model" or key.startswith(prefix)
            )
    else:
        raise ValueError("Expected an RF-DETR checkpoint with 'model' or 'state_dict' weights.")
    if not isinstance(state, dict) or not state:
        raise ValueError("The checkpoint contains no model weights.")
    model.load_state_dict(align_compiled_state_dict(model, state), strict=True)
    print(f"Initialized model weights from {weights_path}; starting a new training run.")


def validate_distributed_environment(args: Args) -> None:
    """Reject mismatched torchrun topology before starting distributed collectives."""
    expected = args.num_nodes * args.devices
    if args.num_nodes > 1 and "RANK" not in os.environ:
        raise ValueError("Multi-node training requires torchrun on every node; use resume_sparks.py.")
    if "RANK" in os.environ:
        world = int(os.environ.get("WORLD_SIZE", "0"))
        local = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
        if world != expected or local != args.devices:
            raise ValueError(
                f"torchrun topology ({world} total, {local} local processes) does not match "
                f"--num-nodes {args.num_nodes} --devices {args.devices}."
            )


def configure_resumed_early_stopping(checkpoint, *, patience, reset=False):
    """Update only loaded callback bookkeeping; never rewrite the checkpoint file."""
    import torch

    callbacks = checkpoint.get("callbacks") or {}
    stoppers = [state for key, state in callbacks.items() if key.startswith("RFDETREarlyStopping{")]
    if len(stoppers) != 1:
        if reset:
            raise ValueError("Expected one RF-DETR early-stopping state in the resume checkpoint.")
        return
    state, = stoppers
    state["patience"] = patience
    if not reset:
        return

    # min_delta previously kept an older, lower baseline than the best saved EMA.
    candidates = [state.get("best_score")]
    for key, saved in callbacks.items():
        if key.startswith("BestModelCallback{"):
            candidates.extend(saved.get(name) for name in ("_best_ema", "_best_raw_regular", "best_model_score"))
    scores = [float(value) for value in candidates if value is not None and math.isfinite(float(value))]
    if not scores:
        raise ValueError("The checkpoint has no finite historical best for early stopping.")
    best = max(scores)
    previous = state.get("best_score")
    state.update(
        wait_count=0, stopped_epoch=0, patience=patience,
        best_score=previous.new_tensor(best) if isinstance(previous, torch.Tensor) else torch.tensor(best),
        stopping_reason=0, stopping_reason_message=None,
    )
    print(f"Reset resumed early stopping: wait_count=0/{patience}, historical best mAP={best:.8f}.")


def train(args: Args) -> None:
    validate_distributed_environment(args)
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    from torch._dynamo import config as dynamo_config
    from rfdetr.config import RFDETRSegLargeConfig, SegmentationTrainConfig
    from rfdetr.datasets import coco
    from rfdetr.models import lwdetr
    from rfdetr.models._defaults import ModelDefaults
    from rfdetr.models.lwdetr import build_model_from_config
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer, module_model, seed_everything
    from rfdetr.training.callbacks.ema import RFDETREMACallback

    wandb_logger = make_wandb_logger(args)

    class Objects365ModelModule(RFDETRModelModule):
        def on_load_checkpoint(self, checkpoint):
            super().on_load_checkpoint(checkpoint)
            checkpoint["state_dict"] = align_compiled_state_dict(self, checkpoint["state_dict"])
            configure_resumed_early_stopping(
                checkpoint, patience=args.early_stopping_patience, reset=args.reset_early_stopping,
            )

        def on_save_checkpoint(self, checkpoint):
            super().on_save_checkpoint(checkpoint)
            if wandb_logger is not None and self.trainer.is_global_zero:
                run = wandb_logger.experiment
                checkpoint[WANDB_RUN_KEY] = {"id": run.id, "project": run.project, "entity": run.entity}

    class Objects365DataModule(RFDETRDataModule):
        def setup(self, stage):
            already_built = self._dataset_train is not None and self._dataset_val is not None
            super().setup(stage)
            if stage == "fit" and not already_built:
                validate_annotations(self._dataset_train, self._dataset_val)

    class CarPostProcess(PostProcess):
        def forward(self, outputs, target_sizes, score_threshold=None):
            # Keep only the Car logit before ranking predictions for COCO AP.
            car_outputs = {**outputs, "pred_logits": outputs["pred_logits"][..., :1]}
            return super().forward(car_outputs, target_sizes, score_threshold)

    model_config = RFDETRSegLargeConfig(
        num_classes=1,
        num_queries=200,
        num_select=200,
        pretrain_weights=None,
        layer_norm=True,
        gradient_checkpointing=False,
        compile=True,
    )
    train_config = SegmentationTrainConfig(
        dataset_file="roboflow",
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        # RF-DETR restores the existing CSV history while constructing the trainer.
        resume=args.resume if rank_zero_only.rank == 0 else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        eval_batch_size=1,
        eval_max_dets=100,
        eval_backend="faster_coco_eval",
        compute_val_loss=True,
        eval_interval=1,
        num_workers=args.num_workers,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp" if args.devices * args.num_nodes > 1 else "auto",
        seed=args.seed,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_vit_layer_decay=0.8,
        lr_component_decay=0.7,
        lr_scheduler="torch.optim.lr_scheduler.ConstantLR",
        lr_scheduler_kwargs={"factor": 1.0},
        warmup_epochs=0.0,
        clip_max_norm=0.1,
        use_ema=True,
        augmentation_backend="torchvision",
        scale_jitter=True,
        square_resize_div_64=True,
        multi_scale="per-batch",
        expanded_scales=True,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        run_test=False,
        progress_bar="tqdm",
        train_log_on_step=True,
        train_log_sync_dist=True,
        wandb=False,
        tensorboard=False,
    )
    seed_everything(args.seed, workers=True)
    # Route windowed attention at the blocks named in Appendix A. Keep the
    # released Large preset's feature taps, which are indexed separately.
    defaults = ModelDefaults(window_block_indexes=list(WINDOW_BLOCKS))
    build_model = partial(build_model_from_config, defaults=defaults)
    with ExitStack() as stack:
        stack.enter_context(patch.object(module_model, "build_model_from_config", build_model))
        if args.pretrain_weights is not None or args.resume is not None:
            # RF-DETR 1.11.2 otherwise downloads DINOv2 during construction,
            # even though the full network will be loaded from our checkpoint.
            backbone_builder = lwdetr.build_backbone

            def without_encoder_weights(**kwargs):
                return backbone_builder(**{**kwargs, "load_dinov2_weights": False})

            stack.enter_context(patch.object(lwdetr, "build_backbone", without_encoder_weights))
        model = Objects365ModelModule(model_config, train_config)
    # RF-DETR defaults to permissive loading, which can silently skip all live
    # model weights when resuming across a change to torch.compile.
    model.strict_loading = True
    model.postprocess = CarPostProcess(num_select=train_config.eval_max_dets)

    if args.pretrain_weights is not None:
        load_pretrain_weights(model.model, args.pretrain_weights)
    elif args.resume is None:
        load_dinov2(model.model.backbone[0].encoder.encoder, args.dinov2_weights)

    data = Objects365DataModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config, log_every_n_steps=1)
    if wandb_logger is not None:
        trainer.loggers = [*trainer.loggers, wandb_logger]
    global_batch_size = args.batch_size * args.devices * args.num_nodes * args.grad_accum_steps
    # Allow shape-specialized variants for the eleven training scales and
    # validation instead of falling back to eager after the default eight.
    compile_recompile_limit = max(dynamo_config.recompile_limit, 32)
    print(
        f"RF-DETR-Seg Large | Car | 200 queries | {args.num_nodes} nodes x {args.devices} GPUs "
        f"| microbatch {args.batch_size}/GPU x accumulation {args.grad_accum_steps} = {global_batch_size} effective "
        f"| early stopping min_delta={args.early_stopping_min_delta}, patience={args.early_stopping_patience} "
        f"| train scales {TRAIN_RESOLUTIONS} | compile recompile limit {compile_recompile_limit}"
    )
    # RF-DETR 1.11.2 does not align compiled/uncompiled keys when restoring EMA.
    ema_load_weights = RFDETREMACallback._load_ema_weights

    def load_ema_weights(callback, module, state_dict):
        return ema_load_weights(callback, module, align_compiled_state_dict(module, state_dict))

    # Supply the same scale grid to the native dataset transforms and batch
    # resize hook. RF-DETR samples by optimizer step, shared across DDP ranks.
    with (
        patch.object(coco, "compute_multi_scale_scales", paper_scales),
        patch.object(module_model, "compute_multi_scale_scales", paper_scales),
        patch.object(RFDETREMACallback, "_load_ema_weights", load_ema_weights),
        dynamo_config.patch(recompile_limit=compile_recompile_limit),
    ):
        trainer.fit(model, datamodule=data, ckpt_path=args.resume)


def main(argv: list[str] | None = None) -> None:
    args = tyro.cli(Args, args=argv, description=__doc__)
    try:
        installed_version = version("rfdetr")
    except PackageNotFoundError:
        installed_version = None
    if installed_version != RFDETR_VERSION:
        raise SystemExit(f"Install RF-DETR {RFDETR_VERSION} with the project's rfdetrn dependency group.")
    train(args)


if __name__ == "__main__":
    main()
