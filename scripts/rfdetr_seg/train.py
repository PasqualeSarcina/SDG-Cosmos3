"""Run a W&B sweep and train the final RF-DETR-Seg model on a DGX Spark."""

from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import sys

import torch
import tyro
import wandb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rfdetr_seg.settings import OBJECTIVE, REPO_ROOT, load_config
from src.rfdetr_seg.training import best_sweep_run, fit


@dataclass
class Args:
    """Run the sweep, select the best hyperparameters and save the final checkpoint."""

    config: Path
    sweep_count: int = 10

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

    sweep_file = REPO_ROOT / "src" / "rfdetr_seg" / "sweep.yaml"
    sweep_config = yaml.safe_load(sweep_file.read_text(encoding="utf-8"))
    if not isinstance(sweep_config, dict) or sweep_config.get("metric") != {"name": OBJECTIVE, "goal": "maximize"}:
        raise ValueError(f"sweep.yaml must maximize {OBJECTIVE}")
    project = config["wandb_project"]
    sweep_id = wandb.sweep(sweep_config, project=project)
    api = wandb.Api()
    entity = api.settings["entity"] or api.default_entity
    processes = get_context("spawn")

    def run_trial():
        # The W&B callback runs in a thread. Keep each CUDA trainer in a fresh process.
        worker = processes.Process(target=fit, args=(config_path,), kwargs={"entity": entity, "sweep_id": sweep_id})
        worker.start()
        try:
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(f"Sweep trial exited with status {worker.exitcode}")
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
            worker.close()

    wandb.agent(sweep_id, function=run_trial, entity=entity, project=project, count=args.sweep_count)
    wandb.teardown()
    sweep_path = f"{entity}/{project}/{sweep_id}"
    winner, score = best_sweep_run(wandb.Api().sweep(sweep_path).runs)
    selection = {
        "sweep": sweep_path, "run_id": winner.id, "best_mask_map": score,
        "lr": winner.config["lr"], "weight_decay": winner.config["weight_decay"],
    }
    print(f"Selected W&B run {winner.id} with mask mAP {score:.6f}", flush=True)
    fit(config_path, entity=entity, selection=selection)


if __name__ == "__main__":
    raise SystemExit(main())
