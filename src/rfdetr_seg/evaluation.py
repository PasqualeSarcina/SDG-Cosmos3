"""Evaluate a selected checkpoint with RF-DETR's native test callback."""

import json

import pytorch_lightning as pl
from rfdetr.training import RFDETRDataModule
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

from .training import load_checkpoint
from .settings import write_json


class NativeTestEvaluation(COCOEvalCallback):
    """Adapt the project's (results, targets) output to RF-DETR's mapping API."""

    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, tuple) and len(outputs) == 2:
            results, targets = outputs
            outputs = {"results": results, "targets": targets}
        return super().on_test_batch_end(trainer, module, outputs, batch, batch_idx, dataloader_idx)


def evaluate_test(args):
    _, module = load_checkpoint(args.checkpoint)
    train_config = module.train_config.model_copy(update={
        "dataset_dir": str(args.dataset_dir.resolve()), "dataset_file": "roboflow",
    })
    evaluator = NativeTestEvaluation(
        max_dets=train_config.eval_max_dets,
        segmentation=module.model_config.segmentation_head,
        eval_backend="faster_coco_eval",
    )
    trainer = pl.Trainer(accelerator=args.device, devices=1, precision="32-true",
                         callbacks=[evaluator], logger=False, enable_checkpointing=False)
    results = trainer.test(module, datamodule=RFDETRDataModule(module.model_config, train_config))
    if len(results) != 1 or "test/mAP_50_95" not in results[0]:
        raise RuntimeError("RF-DETR callback did not return one complete test evaluation")
    metrics = dict(results[0])
    # Preserve the historical project key while exposing all native callback metrics.
    metrics["test/box_mAP_50_95"] = metrics["test/mAP_50_95"]
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))
