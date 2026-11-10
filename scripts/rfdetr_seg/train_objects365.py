"""Pre-train RF-DETR-Seg Large on Objects365 Car boxes and instance masks.

Use the project's rfdetrn dependency group. Pass the output of
scripts/add_objects365_car_masks.py as --dataset-dir. It adds SAM3 masks to the
dataset from scripts/download_objects365_car.py, in the same train/valid
COCO layout.

This is a fixed-architecture adaptation of RF-DETR v2, Table 6 and Appendix A.
It keeps the public Seg Large preset's 200 queries and trains only Car. The
optimizer and EMA use the public RF-DETR implementation for details absent
from the paper.

--batch-size sets the training batch on each GPU. Each batch produces one
optimizer update across all GPUs.

Use --wandb-project to log metrics to W&B. The run identity is saved in the
Lightning checkpoint, so --resume reconnects to it automatically.
"""

from dataclasses import dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError, version
import math
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
    # Output of scripts/add_objects365_car_masks.py.
    dataset_dir: Path
    # Directory for logs and checkpoints.
    output_dir: Path
    # Training images per GPU per optimizer update.
    batch_size: int = 4
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
    # Lightning checkpoint to resume from.
    resume: Path | None = None

    def __post_init__(self) -> None:
        if min(self.batch_size, self.devices, self.num_nodes) < 1 or self.num_workers < 0:
            raise ValueError("Batch size, devices and nodes must be positive. Workers must be nonnegative.")
        if self.dinov2_weights is not None and self.resume is not None:
            raise ValueError("--dinov2-weights and --resume are mutually exclusive.")
        for split in ("train", "valid"):
            annotations = self.dataset_dir / split / "_annotations.coco.json"
            if not annotations.is_file():
                raise ValueError(f"Missing {annotations}")
        for checkpoint in (self.dinov2_weights, self.resume):
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


def train(args: Args) -> None:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    from rfdetr.config import RFDETRSegLargeConfig, SegmentationTrainConfig
    from rfdetr.datasets import coco
    from rfdetr.models._defaults import ModelDefaults
    from rfdetr.models.lwdetr import build_model_from_config
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer, module_model, seed_everything

    wandb_logger = make_wandb_logger(args)

    class Objects365ModelModule(RFDETRModelModule):
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
        epochs=60,
        batch_size=args.batch_size,
        grad_accum_steps=1,
        eval_batch_size=1,
        eval_max_dets=100,
        eval_backend="faster_coco_eval",
        compute_val_loss=True,
        num_workers=args.num_workers,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp" if args.devices * args.num_nodes > 1 else "auto",
        seed=args.seed,
        lr=1e-4,
        lr_encoder=1.5e-4,
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
        early_stopping=False,
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
    with patch.object(module_model, "build_model_from_config", build_model):
        model = Objects365ModelModule(model_config, train_config)
    model.postprocess = CarPostProcess(num_select=train_config.eval_max_dets)

    if args.resume is None:
        load_dinov2(model.model.backbone[0].encoder.encoder, args.dinov2_weights)

    data = Objects365DataModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config, log_every_n_steps=1)
    if wandb_logger is not None:
        trainer.loggers = [*trainer.loggers, wandb_logger]
    global_batch_size = args.batch_size * args.devices * args.num_nodes
    print(
        f"RF-DETR-Seg Large | Car | 200 queries | batch {args.batch_size}/GPU, {global_batch_size} global "
        f"| train scales {TRAIN_RESOLUTIONS}"
    )
    # Supply the same scale grid to the native dataset transforms and batch
    # resize hook. RF-DETR samples by optimizer step, shared across DDP ranks.
    with (
        patch.object(coco, "compute_multi_scale_scales", paper_scales),
        patch.object(module_model, "compute_multi_scale_scales", paper_scales),
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
