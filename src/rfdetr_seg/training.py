"""RF-DETR-Seg model setup, training and checkpoint selection."""

from contextlib import ExitStack
import inspect
import os
from pathlib import Path
import re
from shutil import copyfile
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from rfdetr import config as rfconfig
from rfdetr.models import lwdetr
from rfdetr.models.backbone import dinov2
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, module_model
import torch
import wandb

from .data import augmentation, split_coco, split_metadata
from .settings import (
    BOX_MAP, MASK_MAP, OBJECTIVE, RFDETR_VERSION, SECONDARY,
    load_config, validate_hyperparameters, write_json,
)


class CarModule(RFDETRModelModule):
    def evaluation_step(self, batch, *, compute_loss):
        samples, targets = batch
        outputs = self.model(samples)
        if compute_loss:
            loss_dict = self.criterion(outputs, targets)
            weights = self.criterion.weight_dict
            loss = sum(value * weights[key] for key, value in loss_dict.items() if key in weights)
            self._log_val_loss_metrics(loss, loss_dict, batch_size=len(targets))

        sizes = torch.stack([target["orig_size"] for target in targets])
        if outputs["pred_logits"].shape[-1] != 2:
            raise RuntimeError("Expected the car logit (0) and RF-DETR's extra classifier slot (1)")
        outputs["pred_logits"] = outputs["pred_logits"][..., :1].contiguous()
        return self.postprocess(outputs, sizes), targets

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, compute_loss=self._should_compute_val_loss)

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, compute_loss=False)


def build_module(model_config, train_config):
    if model_config.pretrain_weights is not None:
        raise ValueError("Expected pretrain_weights=None")

    def reject_pretrained(*args, **kwargs):
        raise RuntimeError("RF-DETR attempted to load pretrained weights during initialization")

    backbone_builder = lwdetr.build_backbone
    with ExitStack() as stack:
        stack.enter_context(patch.object(module_model, "load_pretrain_weights", reject_pretrained))
        stack.enter_context(patch.object(dinov2.AutoBackbone, "from_pretrained", reject_pretrained))
        stack.enter_context(patch.object(dinov2.WindowedDinov2WithRegistersBackbone, "from_pretrained", reject_pretrained))
        if "load_encoder_weights" in inspect.signature(RFDETRModelModule).parameters:
            return CarModule(model_config, train_config, load_encoder_weights=False)

        # In 1.11.2, pretrain_weights=None would request DINOv2 weights from build_backbone.
        if "load_dinov2_weights" not in inspect.signature(backbone_builder).parameters:
            raise RuntimeError("Unsupported RF-DETR backbone API. Check the pinned release before training.")

        def without_encoder_weights(**kwargs):
            if "load_dinov2_weights" not in kwargs:
                raise RuntimeError("RF-DETR changed the encoder construction path")
            return backbone_builder(**{**kwargs, "load_dinov2_weights": False})

        audited_builder = stack.enter_context(patch.object(lwdetr, "build_backbone", side_effect=without_encoder_weights))
        model = CarModule(model_config, train_config)
        if audited_builder.call_count != 1:
            raise RuntimeError("Expected exactly one RF-DETR backbone initialization")
        return model


def configs(config, lr, lr_encoder, weight_decay, *, epochs, num_nodes=1):
    cls = getattr(rfconfig, f"RFDETRSeg{config['variant']}Config")
    model_config = cls(pretrain_weights=None, num_classes=1, num_select=100, compile=True)
    train_config = rfconfig.SegmentationTrainConfig(
        dataset_dir=config["dataset_dir"], dataset_file="roboflow", output_dir=config["output_root"],
        augmentation_backend="albumentations", aug_config=augmentation(),
        epochs=epochs, eval_interval=config["eval_interval"],
        seed=config["seed"], batch_size=config["batch_size"], grad_accum_steps=1,
        eval_batch_size=1, num_workers=config["num_workers"], pack_targets=False,
        multi_scale=False, scale_jitter=False, square_resize_div_64=True, use_ema=False, compute_val_loss=True,
        lr=lr, lr_encoder=lr_encoder, weight_decay=weight_decay,
        lr_vit_layer_decay=1.0, lr_component_decay=1.0, lr_scheduler="cosine", warmup_epochs=0.5,
        eval_max_dets=100, eval_masks_head_resolution=False, train_log_on_step=True, train_log_sync_dist=True,
        accelerator="gpu", num_nodes=num_nodes, devices=1,
        strategy="ddp" if num_nodes > 1 else "auto", run_test=False, wandb=False, tensorboard=False,
    )
    return model_config, train_config


def instances(result, category_id, image_id):
    boxes = result["boxes"].detach().to(device="cpu", copy=True)
    scores = result["scores"].detach().cpu().tolist()
    if (result["labels"] != 0).any() or not torch.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise RuntimeError("Invalid car predictions")
    boxes[:, 2:] -= boxes[:, :2]
    masks = result["masks"].detach().cpu().numpy()[:, 0]
    encoded = mask_utils.encode(np.asfortranarray(masks.transpose(1, 2, 0), dtype=np.uint8)) if len(masks) else []
    records = []
    for box, score, rle in zip(boxes.tolist(), scores, encoded, strict=True):
        rle["counts"] = rle["counts"].decode("ascii")
        records.append({"image_id": image_id, "category_id": category_id, "bbox": box, "score": score, "segmentation": rle})
    return records


def coco_metrics(coco, by_image):
    if by_image.keys() != coco.imgs.keys():
        raise RuntimeError("Evaluation did not cover exactly the annotated COCO images")

    coco.dataset.setdefault("info", {})
    scores = []
    for task, key in (("segm", "segmentation"), ("bbox", "bbox")):
        detections = [
            {k: row[k] for k in ("image_id", "category_id", "score", key)}
            for image_id in sorted(by_image) for row in by_image[image_id]
        ]
        if not detections:
            scores.append(0.0)
            continue
        # Separate result records let COCO compute detection area from the right geometry.
        evaluator = COCOeval(coco, coco.loadRes(detections), task)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        score = float(evaluator.stats[0])
        if not 0 <= score <= 1:
            raise RuntimeError(f"Invalid COCO {task} AP {score}")
        scores.append(score)
    return scores


class Evaluation(pl.Callback):
    def __init__(self, split, checkpoint_dir, provenance=None):
        self.split = split
        self.checkpoint_dir = checkpoint_dir
        self.provenance = provenance
        self.predictions = {}
        self.history = []
        self.best = None
        self.metrics = None

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        results, targets = outputs
        category = split_coco(trainer.datamodule, self.split).label2cat[0]
        for result, target in zip(results, targets, strict=True):
            image_id = target["image_id"].item()
            self.predictions[image_id] = instances(result, category, image_id)

    on_test_batch_end = on_validation_batch_end

    def collect(self, trainer):
        predictions = self.predictions
        self.predictions = {}
        if trainer.world_size == 1:
            mask, box = coco_metrics(split_coco(trainer.datamodule, self.split), predictions)
        else:
            gathered = [None] * trainer.world_size if trainer.is_global_zero else None
            torch.distributed.gather_object(predictions, gathered, dst=0)
            metrics = None
            if gathered is not None:
                by_image = {}
                for shard in gathered:
                    for image_id, records in shard.items():
                        # DistributedSampler can repeat images when padding the validation split.
                        by_image.setdefault(image_id, records)
                metrics = coco_metrics(split_coco(trainer.datamodule, self.split), by_image)
            mask, box = trainer.strategy.broadcast(metrics, src=0)
        return {f"{self.split}/segm_mAP_50_95": mask, f"{self.split}/box_mAP_50_95": box}

    def on_validation_epoch_end(self, trainer, module):
        scores = self.collect(trainer)
        if self.provenance and self.provenance.get("sweep_id"):
            previous_best = self.best[MASK_MAP] if self.best is not None else scores[MASK_MAP]
            scores[OBJECTIVE] = max(previous_best, scores[MASK_MAP])
        module.log_dict(scores, on_step=False, on_epoch=True, batch_size=1, sync_dist=trainer.world_size > 1)
        record = {"epoch": trainer.current_epoch + 1, "step": trainer.global_step, **scores}
        self.history.append(record)
        if self.best is not None and record[MASK_MAP] <= self.best[MASK_MAP]:
            return
        if trainer.is_global_zero and self.checkpoint_dir is not None:
            temporary = self.checkpoint_dir / "best.pt.tmp"
            model = getattr(module.model, "_orig_mod", module.model)
            torch.save({
                "state_dict": model.state_dict(),
                "rfdetr_version": RFDETR_VERSION,
                "model_config": module.model_config.model_dump(mode="json"),
                "train_config": module.train_config.model_dump(mode="json"),
                "selection": record,
                "provenance": self.provenance,
                "data": {split: split_metadata(split_coco(trainer.datamodule, split)) for split in ("train", "val")},
            }, temporary)
            temporary.replace(self.checkpoint_dir / "best.pt")
        self.best = record

    def on_test_epoch_end(self, trainer, module):
        self.metrics = self.collect(trainer)


def trial_output(output_root, sweep_id, run_id):
    folder = sweep_id or "standalone"
    for identifier in (folder, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError(f"Unexpected run or sweep ID {identifier!r}")
    root = Path(output_root) / folder
    root.mkdir(parents=True, exist_ok=True)
    output = root / run_id
    output.mkdir(exist_ok=False)
    return output


def save_best_sweep_checkpoint(sweep_dir):
    candidates = []
    for path in sorted(sweep_dir.glob("*/best.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["selection"]
        candidates.append((record[MASK_MAP], path))
    if not candidates:
        raise RuntimeError(f"No validation checkpoint found in {sweep_dir}")
    score, winner = max(candidates, key=lambda item: item[0])
    destination = sweep_dir / "best.pt"
    temporary = sweep_dir / "best.pt.tmp"
    copyfile(winner, temporary)
    temporary.replace(destination)
    return destination, score


def training_topology():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and (
        not torch.distributed.is_torchelastic_launched() or os.environ.get("LOCAL_WORLD_SIZE") != "1"
    ):
        raise ValueError("Distributed RF-DETR training requires torchrun with one process per node (--nproc-per-node=1)")
    if not 0 <= rank < world_size:
        raise ValueError(f"Invalid training rank {rank} for world size {world_size}")
    return world_size, rank


def fit(config_path, *, entity=None, sweep_id=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = load_config(config_path)
    num_nodes, rank = training_topology()
    if num_nodes > 1 and (sweep_id or config["sweep_enabled"]):
        raise ValueError("Distributed RF-DETR training requires sweep_enabled=false")

    with ExitStack() as stack:
        output = None
        run = None
        provenance = None
        logger = False
        if rank == 0:
            run_id = os.environ["WANDB_RUN_ID"] if sweep_id else uuid4().hex[:12]
            output = trial_output(config["output_root"], sweep_id, run_id)
            print(f"Training run {run_id} in {output}", flush=True)
            run = stack.enter_context(wandb.init(
                entity=entity, project=config["wandb_project"], id=run_id,
                dir=str(output), mode="online", job_type="train",
            ))
            if (run.id, run.project) != (run_id, config["wandb_project"]) or (entity is not None and run.entity != entity):
                raise RuntimeError("W&B run does not match the expected run ID, entity and project")
            if (run.sweep_id or None) != sweep_id:
                raise RuntimeError("W&B run does not belong to the expected sweep")
            provenance = {"variant": config["variant"], "sweep_id": sweep_id, "run_id": run_id}
            logger = WandbLogger(experiment=run, log_model=False, save_dir=str(output))

        parameters = run.config if sweep_id else config
        lr = parameters["lr"]
        lr_encoder = parameters.get("lr_encoder", lr)
        weight_decay = config["weight_decay"]
        validate_hyperparameters(lr, lr_encoder, weight_decay)
        epochs = config["sweep_epochs"] if sweep_id else config["epochs"]
        model_config, train_config = configs(config, lr, lr_encoder, weight_decay, epochs=epochs, num_nodes=num_nodes)
        pl.seed_everything(config["seed"], workers=True)
        module = build_module(model_config, train_config)
        evaluation = Evaluation("val", output, provenance)
        early_stopping = EarlyStopping(
            monitor=MASK_MAP, mode="max", min_delta=0.001,
            patience=15, check_on_train_epoch_end=False, verbose=True,
        )
        if run is not None:
            run.config.update({
                "lr": lr, "lr_encoder": lr_encoder, "weight_decay": weight_decay,
                "variant": config["variant"], "resolution": model_config.resolution,
                "seed": config["seed"], "batch_size_per_gpu": train_config.batch_size,
                "effective_batch_size": train_config.batch_size * num_nodes, "num_nodes": num_nodes,
                "epochs": train_config.epochs, "eval_interval": train_config.eval_interval,
                "rfdetr_version": RFDETR_VERSION,
            })
        trainer = pl.Trainer(
            accelerator="gpu", num_nodes=num_nodes, devices=1, precision="bf16-mixed",
            strategy=DDPStrategy(find_unused_parameters=True) if num_nodes > 1 else "auto",
            max_epochs=train_config.epochs, accumulate_grad_batches=1,
            check_val_every_n_epoch=train_config.eval_interval, num_sanity_val_steps=0,
            callbacks=[evaluation, early_stopping], logger=logger, log_every_n_steps=1,
            enable_checkpointing=False, deterministic="warn", gradient_clip_val=0.1,
            default_root_dir=str(output or config["output_root"]),
        )
        data = RFDETRDataModule(model_config, train_config)
        trainer.fit(module, datamodule=data)
        best = evaluation.best
        if best is None or trainer.global_step <= 0:
            raise RuntimeError("Training finished without optimizer updates or validation metrics")
        if run is None:
            return
        if not (output / "best.pt").is_file():
            raise RuntimeError("Training finished without a validation checkpoint")
        if sweep_id:
            metrics = {OBJECTIVE: best[MASK_MAP], SECONDARY: best[BOX_MAP]}
            run.log({SECONDARY: metrics[SECONDARY]})
        else:
            metrics = {"best/mask_map": best[MASK_MAP], "best/box_map": best[BOX_MAP]}
        run.summary.update(metrics)
        write_json(output / "result.json", {
            "run_path": f"{run.entity}/{run.project}/{run.id}", "run_id": run.id,
            "sweep_id": sweep_id, "optimizer_steps": trainer.global_step,
            "best": best, "evaluations": evaluation.history,
            "objective": metrics, "validation_images": len(split_coco(data, "val").imgs),
        })
    write_json(output / "completed.json", {"status": "finished"})
    if sweep_id is None:
        print(f"Best checkpoint (mask mAP {best[MASK_MAP]:.6f}): {output / 'best.pt'}", flush=True)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["rfdetr_version"] != RFDETR_VERSION:
        raise ValueError("Checkpoint requires a different RF-DETR version")
    variant = checkpoint["provenance"]["variant"]
    model_config = getattr(rfconfig, f"RFDETRSeg{variant}Config").model_validate({
        **checkpoint["model_config"], "compile": False,
    })
    train_config = rfconfig.SegmentationTrainConfig.model_validate(checkpoint["train_config"])
    module = build_module(model_config, train_config)
    module.model.load_state_dict(checkpoint["state_dict"], strict=True)
    return checkpoint, module.eval()
