"""Experiment configuration and provenance."""

import json
from math import isfinite
from pathlib import Path

RFDETR_VERSION = "1.11.2"
MASK_MAP = "val/segm_mAP_50_95"
BOX_MAP = "val/box_mAP_50_95"
OBJECTIVE = "sweep/best_mask_map"
SECONDARY = "sweep/best_checkpoint_box_map"
VARIANTS = ("Nano", "Small", "Medium", "Large", "XLarge", "2XLarge")
REPO_ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    config = read_json(path)
    config.pop("grad_accum_steps", None)  # Accept experiment files copied from the old template.
    config.pop("dataset_lock", None)
    if config["variant"] not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    for key in ("resolution", "epochs", "eval_interval", "batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "num_workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["eval_interval"] > config["epochs"]:
        raise ValueError("eval_interval cannot exceed epochs")
    sweep_epochs = config.setdefault("sweep_epochs", min(config["epochs"], max(20, config["eval_interval"])))
    if type(sweep_epochs) is not int or not config["eval_interval"] <= sweep_epochs <= config["epochs"]:
        raise ValueError("sweep_epochs must be an integer between eval_interval and epochs")
    for key in ("dataset_dir", "output_root"):
        if not Path(config[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    return config


def validate_hyperparameters(lr, weight_decay):
    if not isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
