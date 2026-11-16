"""Evaluate a selected checkpoint with RF-DETR's native test callback."""

from functools import partial
import json
from unittest.mock import patch

import pytorch_lightning as pl
import torch
from rfdetr.config import RFDETRSegLargeConfig, SegmentationTrainConfig
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64
from rfdetr.models import lwdetr
from rfdetr.models._defaults import ModelDefaults
from rfdetr.models.postprocess import PostProcess
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, module_model
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

from .settings import RFDETR_VERSION, write_json


class NativeTestEvaluation(COCOEvalCallback):
    """Evaluate individual-instance targets without restoring crowd annotations."""

    def _get_crowd_regions(self, trainer, split):
        # CocoDetection removes iscrowd targets for the training loss.
        # Keep that same target set for every test metric.
        return {}

    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, tuple) and len(outputs) == 2:
            results, targets = outputs
            outputs = {"results": results, "targets": targets}
        return super().on_test_batch_end(trainer, module, outputs, batch, batch_idx, dataloader_idx)


class CarPostProcess(PostProcess):
    """Keep the car logit before top-k ranking, as in train_objects365.py."""

    def forward(self, outputs, target_sizes, score_threshold=None):
        outputs = {**outputs, "pred_logits": outputs["pred_logits"][..., :1].contiguous()}
        return super().forward(outputs, target_sizes, score_threshold)


def load_test_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "model" not in checkpoint or "args" not in checkpoint:
        from .training import load_checkpoint
        return load_checkpoint(path)
    if checkpoint["rfdetr_version"] != RFDETR_VERSION:
        raise ValueError("Checkpoint requires a different RF-DETR version")
    if checkpoint.get("model_name") != "RFDETRSegLarge":
        raise ValueError("The exported-checkpoint loader supports the project's RFDETRSegLarge architecture")
    config = RFDETRSegLargeConfig.model_validate({
        **checkpoint["model_config"], "compile": False, "pretrain_weights": None,
    })
    train_config = SegmentationTrainConfig.model_validate({
        **checkpoint["args"], "compute_test_loss": False, "num_workers": 0,
        "devices": 1, "num_nodes": 1, "strategy": "auto", "pack_targets": False,
    })
    builder = lwdetr.build_backbone
    defaults = ModelDefaults(window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10])
    with patch.object(lwdetr, "build_backbone", lambda **kw: builder(**{**kw, "load_dinov2_weights": False})), \
            patch.object(module_model, "build_model_from_config", partial(lwdetr.build_model_from_config, defaults=defaults)):
        module = RFDETRModelModule(config, train_config)
    weights = {".".join(part for part in key.split(".") if part != "_orig_mod"): value
               for key, value in checkpoint["model"].items()}
    module.model.load_state_dict(weights, strict=True)
    module.postprocess = CarPostProcess(num_select=train_config.eval_max_dets)
    return checkpoint, module.eval()


class AnnotatedTestData(RFDETRDataModule):
    """Use an explicit COCO JSON and image directory with native val transforms."""

    def __init__(self, model_config, train_config, annotations):
        super().__init__(model_config, train_config)
        self.annotations = annotations

    def setup(self, stage):
        if stage != "test":
            raise ValueError("AnnotatedTestData is intended for test evaluation")
        with self.annotations.open(encoding="utf-8") as stream:
            annotation = json.load(stream)
        category, = [row for row in annotation["categories"] if row["name"] == "car"]
        if any(row["category_id"] != category["id"] for row in annotation["annotations"]):
            raise ValueError("Explicit test annotations must contain only the car category")
        from pathlib import Path
        root = Path(self.train_config.dataset_dir)
        image_root = root / "data" if (root / "data").is_dir() else root
        config = self.model_config
        transform = make_coco_transforms_square_div_64(
            "val", config.resolution, patch_size=config.patch_size,
            num_windows=config.num_windows, scale_jitter=False,
        )
        self._dataset_test = CocoDetection(
            image_root, self.annotations, transforms=transform,
            include_masks=config.segmentation_head, remap_category_ids=True,
            cat2label={category["id"]: 0},
        )


def evaluate_test(args):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    _, module = load_test_checkpoint(args.checkpoint)
    train_config = module.train_config.model_copy(update={
        "dataset_dir": str(args.dataset_dir.resolve()), "dataset_file": "roboflow",
        "num_workers": 0, "eval_batch_size": 1,
    })
    evaluator = NativeTestEvaluation(
        max_dets=train_config.eval_max_dets,
        segmentation=module.model_config.segmentation_head,
        eval_interval=train_config.eval_interval,
        log_per_class_metrics=train_config.log_per_class_metrics,
        eval_backend=train_config.eval_backend,
        keypoint_oks_sigmas=train_config.keypoint_oks_sigmas,
        eval_base_model=train_config.eval_base_model,
    )
    trainer = pl.Trainer(accelerator=args.device, devices=1, precision="32-true",
                         callbacks=[evaluator], logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    annotations = getattr(args, "annotations", None)
    data = (AnnotatedTestData(module.model_config, train_config, annotations) if annotations is not None
            else RFDETRDataModule(module.model_config, train_config))
    results = trainer.test(module, datamodule=data)
    if len(results) != 1 or "test/mAP_50_95" not in results[0]:
        raise RuntimeError("RF-DETR callback did not return one complete test evaluation")
    metrics = dict(results[0])
    metrics["test/box_mAP_50_95"] = metrics["test/mAP_50_95"]
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))
