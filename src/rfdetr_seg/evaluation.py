"""Evaluate a selected checkpoint on the test split."""

import json

import pytorch_lightning as pl
from rfdetr.training import RFDETRDataModule

from .training import Evaluation, load_checkpoint
from .settings import write_json


def evaluate_test(args):
    _, module = load_checkpoint(args.checkpoint)
    train_config = module.train_config.model_copy(update={
        "dataset_dir": str(args.dataset_dir.resolve()), "dataset_file": "roboflow",
    })
    evaluator = Evaluation("test", None)
    trainer = pl.Trainer(accelerator=args.device, devices=1, precision="32-true",
                         callbacks=[evaluator], logger=False, enable_checkpointing=False)
    trainer.test(module, datamodule=RFDETRDataModule(module.model_config, train_config))
    write_json(args.output, evaluator.metrics)
    print(json.dumps(evaluator.metrics, indent=2))
