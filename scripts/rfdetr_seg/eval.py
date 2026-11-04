"""Evaluate a selected RF-DETR-Seg checkpoint on the annotated test set."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rfdetr_seg.evaluation import evaluate_test


@dataclass
class Args:
    """Evaluate an RF-DETR-Seg checkpoint on the held-out test set."""

    checkpoint: Path
    output: Path
    dataset_dir: Path
    device: Literal["gpu", "cpu"] = "gpu"

    def __post_init__(self) -> None:
        if not self.checkpoint.is_file():
            raise ValueError(f"--checkpoint must point to a file: {self.checkpoint}")
        if self.output.is_dir() or not self.output.parent.is_dir():
            raise ValueError(f"--output must be a file path in an existing directory: {self.output}")


def main(argv: list[str] | None = None) -> None:
    args = tyro.cli(Args, args=argv)
    evaluate_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
