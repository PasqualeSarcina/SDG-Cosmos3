"""Train RF-DETR-Seg and save the best validation checkpoint."""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

import torch
import tyro
import wandb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rfdetr_seg.settings import OBJECTIVE, REPO_ROOT, load_config
from src.rfdetr_seg.training import fit, save_best_sweep_checkpoint, training_topology


@dataclass
class Args:
    """Train once or launch a W&B sweep when enabled in the experiment config."""

    config: Path
    sweep_count: int = 9
    trial: bool = False  # Used by the W&B agent to launch a single trial.

    def __post_init__(self) -> None:
        if not self.config.is_file():
            raise ValueError(f"--config must point to an experiment file: {self.config}")
        if self.sweep_count < 1:
            raise ValueError("--sweep-count must be a positive integer")


def main(argv: list[str] | None = None) -> None:
    args = tyro.cli(Args, args=argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA GPU with BF16 support")
    if args.trial:
        fit(config_path, entity=os.environ["WANDB_ENTITY"], sweep_id=os.environ["WANDB_SWEEP_ID"])
        return
    if not config["sweep_enabled"]:
        fit(config_path)
        return

    if training_topology()[0] > 1:
        raise ValueError("Distributed RF-DETR training requires sweep_enabled=false")
    sweep_file = REPO_ROOT / "src" / "rfdetr_seg" / "sweep.yaml"
    sweep_config = yaml.safe_load(sweep_file.read_text(encoding="utf-8"))
    if not isinstance(sweep_config, dict) or sweep_config.get("metric") != {"name": OBJECTIVE, "goal": "maximize"}:
        raise ValueError(f"sweep.yaml must maximize {OBJECTIVE}")
    sweep_config["program"] = str(Path(__file__).resolve())
    sweep_config["command"] = [sys.executable, "${program}", "--config", str(config_path), "--trial"]
    project = config["wandb_project"]
    sweep_id = wandb.sweep(sweep_config, project=project)
    api = wandb.Api()
    entity = api.settings["entity"] or api.default_entity
    sweep_path = f"{entity}/{project}/{sweep_id}"
    # The CLI agent owns the trial process and can terminate it on a pruning command.
    subprocess.run(
        [sys.executable, "-m", "wandb", "agent", "--count", str(args.sweep_count), sweep_path],
        check=True,
    )
    wandb.teardown()
    checkpoint, score = save_best_sweep_checkpoint(Path(config["output_root"]) / sweep_id)
    print(f"Best sweep checkpoint (mask mAP {score:.6f}): {checkpoint}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
