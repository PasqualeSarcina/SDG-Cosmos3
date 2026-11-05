"""RF-DETR-Seg model setup, training and checkpoint selection."""

from contextlib import ExitStack
import inspect
import os
from pathlib import Path
import re
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
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


def configs(config, lr, weight_decay, *, epochs=None):
    cls = getattr(rfconfig, f"RFDETRSeg{config['variant']}Config")
    model_config = cls(
        pretrain_weights=None, num_classes=1, num_select=100,
        resolution=config["resolution"], compile=True,
    )
    if model_config.resolution % (model_config.patch_size * model_config.num_windows):
        raise ValueError("resolution must be divisible by patch_size * num_windows for the selected variant")
    train_config = rfconfig.SegmentationTrainConfig(
        dataset_dir=config["dataset_dir"], dataset_file="roboflow", output_dir=config["output_root"],
        augmentation_backend="albumentations", aug_config=augmentation(),
        epochs=config["epochs"] if epochs is None else epochs, eval_interval=config["eval_interval"],
        seed=config["seed"], batch_size=config["batch_size"], grad_accum_steps=1,
        eval_batch_size=1, num_workers=config["num_workers"], pack_targets=False,
        multi_scale=False, scale_jitter=False, square_resize_div_64=True, use_ema=False, compute_val_loss=True,
        lr=lr, lr_encoder=lr, weight_decay=weight_decay,
        lr_vit_layer_decay=1.0, lr_component_decay=1.0, lr_scheduler="cosine", warmup_epochs=0.5,
        eval_max_dets=100, eval_masks_head_resolution=False, train_log_on_step=True, train_log_sync_dist=False,
        accelerator="gpu", num_nodes=1, devices=1, run_test=False, wandb=False, tensorboard=False,
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
        mask, box = coco_metrics(split_coco(trainer.datamodule, self.split), self.predictions)
        self.predictions.clear()
        return {f"{self.split}/segm_mAP_50_95": mask, f"{self.split}/box_mAP_50_95": box}

    def on_validation_epoch_end(self, trainer, module):
        scores = self.collect(trainer)
        if self.checkpoint_dir is None:
            previous_best = self.best[MASK_MAP] if self.best is not None else scores[MASK_MAP]
            scores[OBJECTIVE] = max(previous_best, scores[MASK_MAP])
        module.log_dict(scores, on_step=False, on_epoch=True, batch_size=1)
        record = {"epoch": trainer.current_epoch + 1, "step": trainer.global_step, **scores}
        self.history.append(record)
        if self.best is not None and record[MASK_MAP] <= self.best[MASK_MAP]:
            return
        if self.checkpoint_dir is not None:
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


def best_sweep_run(runs):
    candidates = []
    for run in runs:
        if run.state in {"running", "pending", "queued"}:
            raise RuntimeError("The sweep still has active runs. Finalize after the W&B agent finishes")
        if run.state == "finished":
            score = run.summary.get(OBJECTIVE)
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                candidates.append((score, run))
    if not candidates:
        raise RuntimeError(f"No finished W&B run reported {OBJECTIVE}")
    score, run = min(candidates, key=lambda item: (-item[0], item[1].id))
    return run, score


def fit(config_path, *, entity, sweep_id=None, selection=None):
    if (sweep_id is None) == (selection is None):
        raise ValueError("Specify sweep_id for a trial or selection for final training")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = load_config(config_path)
    epochs = config["sweep_epochs"] if sweep_id else config["epochs"]
    run_id = os.environ["WANDB_RUN_ID"] if sweep_id else uuid4().hex[:12]
    output = trial_output(config["output_root"], sweep_id, run_id)
    print(f"Trial {run_id} in {output}", flush=True)
    checkpoint_dir = None if sweep_id else output
    with wandb.init(entity=entity, project=config["wandb_project"], id=run_id,
                    dir=str(output), mode="online", job_type="final_train" if selection is not None else "train") as run:
        if (run.id, run.entity, run.project) != (run_id, entity, config["wandb_project"]):
            raise RuntimeError("W&B run does not match the expected run ID, entity and project")
        if (run.sweep_id or None) != sweep_id:
            raise RuntimeError("W&B run does not belong to the expected sweep")

        parameters = run.config if sweep_id else selection
        lr, weight_decay = parameters["lr"], parameters["weight_decay"]
        validate_hyperparameters(lr, weight_decay)
        model_config, train_config = configs(config, lr, weight_decay, epochs=epochs)
        pl.seed_everything(config["seed"], workers=True)
        module = build_module(model_config, train_config)
        provenance = {"variant": config["variant"]}
        if selection is not None:
            provenance["sweep_selection"] = selection
        evaluation = Evaluation("val", checkpoint_dir, provenance)
        early_stopping = EarlyStopping(
            monitor=MASK_MAP, mode="max", min_delta=0.001,
            patience=10 if sweep_id else 15, check_on_train_epoch_end=False, verbose=True,
        )
        run.config.update({
            "lr": lr, "lr_encoder": lr, "weight_decay": weight_decay,
            "variant": config["variant"], "resolution": model_config.resolution,
            "seed": config["seed"], "effective_batch_size": train_config.batch_size,
            "epochs": train_config.epochs, "eval_interval": train_config.eval_interval,
            "rfdetr_version": RFDETR_VERSION,
        })
        if selection is not None:
            run.config.update({"sweep_selection": selection})
        logger = WandbLogger(experiment=run, log_model=False, save_dir=str(output))
        trainer = pl.Trainer(
            accelerator="gpu", num_nodes=1, devices=1, precision="bf16-mixed",
            max_epochs=train_config.epochs, accumulate_grad_batches=1,
            check_val_every_n_epoch=train_config.eval_interval, num_sanity_val_steps=0,
            callbacks=[evaluation, early_stopping], logger=logger, log_every_n_steps=1,
            enable_checkpointing=False, deterministic="warn", gradient_clip_val=0.1,
            default_root_dir=str(output),
        )
        data = RFDETRDataModule(model_config, train_config)
        trainer.fit(module, datamodule=data)
        best = evaluation.best
        if best is None or trainer.global_step <= 0:
            raise RuntimeError("Training finished without optimizer updates or validation metrics")
        if checkpoint_dir is not None and not (checkpoint_dir / "best.pt").is_file():
            raise RuntimeError("Training finished without a validation checkpoint")
        metrics = {OBJECTIVE: best[MASK_MAP], SECONDARY: best[BOX_MAP]}
        run.log({SECONDARY: metrics[SECONDARY]} if sweep_id else metrics)
        run.summary.update(metrics)
        write_json(output / "result.json", {
            "run_path": f"{run.entity}/{run.project}/{run.id}", "run_id": run.id,
            "sweep_id": sweep_id, "optimizer_steps": trainer.global_step,
            "best": best, "evaluations": evaluation.history,
            "objective": metrics, "validation_images": len(split_coco(data, "val").imgs),
        })
    write_json(output / "completed.json", {"status": "finished"})
    if selection is not None:
        print(f"Final checkpoint: {output / 'best.pt'}", flush=True)


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
