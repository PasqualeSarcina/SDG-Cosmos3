"""Timed recovery snapshots, independent of epoch validation and best-model selection."""

from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

import torch
from transformers import TrainerCallback
from transformers.trainer_callback import ExportableState

if TYPE_CHECKING:
    from transformers import Trainer


log = logging.getLogger(__name__)


class RecoveryCheckpointCallback(TrainerCallback):
    """Save every 30 minutes at step boundaries, without requesting evaluation.

    All DDP ranks participate in snapshots; rank zero decides when they are due.
    Completed snapshots live in recovery/checkpoint-<step>, with two retained.
    Epoch checkpoints and the final adapter remain under the original output_dir.
    The wall clock starts afresh on each train() invocation, including resume.
    """

    interval_seconds = 30 * 60
    keep = 2

    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        training_args = trainer.args
        if training_args is None:
            raise ValueError("Recovery checkpoints require configured training arguments.")
        output_dir = training_args.output_dir
        if output_dir is None:
            raise ValueError("Recovery checkpoints require an output directory.")
        self.training_args = training_args
        self.output_dir = Path(output_dir).resolve()
        self.next_save_at: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.next_save_at = monotonic() + self.interval_seconds
        return control

    def on_step_end(self, args, state, control, **kwargs):
        due = self.next_save_at is not None and monotonic() >= self.next_save_at
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            flag = torch.tensor(int(due), dtype=torch.int32, device=args.device)
            torch.distributed.broadcast(flag, src=0)
            due = bool(flag.item())
        if due:
            self._save_recovery_checkpoint()
            self.next_save_at = monotonic() + self.interval_seconds
        return control

    def _save_recovery_checkpoint(self):
        trainer = self.trainer
        root = self.output_dir / "recovery"
        name = f"checkpoint-{trainer.state.global_step}"
        destination = root / name
        # A rewind/resume may revisit a step already saved by an earlier run.
        if destination.is_dir():
            return
        pending = root / f".{name}.incomplete"
        pending.mkdir(parents=True, exist_ok=True)
        log.info("Saving recovery checkpoint at step %s to %s", trainer.state.global_step, destination)

        # Use the same model/optimizer/scaler/RNG writers as Trainer, without its
        # best-metric update or checkpoint rotation in the epoch directory.
        trainer.save_model(str(pending), _internal_call=True)
        trainer._save_optimizer_and_scheduler(str(pending))
        trainer._save_scaler(str(pending))
        trainer._save_rng_state(str(pending))
        if self.training_args.should_save:
            snapshot = copy.deepcopy(trainer.state)
            snapshot.stateful_callbacks = {}
            for callback in [*trainer.callback_handler.callbacks, trainer.control]:
                if isinstance(callback, ExportableState):
                    key = type(callback).__name__
                    value = callback.state()
                    if key in snapshot.stateful_callbacks:
                        previous = snapshot.stateful_callbacks[key]
                        if not isinstance(previous, list):
                            previous = [previous]
                            snapshot.stateful_callbacks[key] = previous
                        previous.append(value)
                    else:
                        snapshot.stateful_callbacks[key] = value
            snapshot.save_to_json(str(pending / "trainer_state.json"))
        trainer.accelerator.wait_for_everyone()
        if self.training_args.should_save:
            pending.rename(destination)
            self._rotate_recovery_checkpoints(root)
        trainer.accelerator.wait_for_everyone()

    def _rotate_recovery_checkpoints(self, root: Path):
        root = root.resolve()
        checkpoints = sorted(
            (path for path in root.glob("checkpoint-*")
             if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()),
            key=lambda path: int(path.name.removeprefix("checkpoint-")),
        )
        for path in checkpoints[:-self.keep]:
            # Only delete generated directories directly inside the recovery root.
            if path.is_symlink() or path.resolve().parent != root:
                raise ValueError(f"Refusing to remove recovery checkpoint outside {root}: {path}")
            shutil.rmtree(path)

    def on_save(self, args, state, control, **kwargs):
        # This event belongs to the ordinary, evaluated epoch checkpoints only.
        # Recovery saves above never trigger it or change best_model_checkpoint.
        if args.should_save and state.best_model_checkpoint is not None:
            checkpoint = Path(state.best_model_checkpoint)
            if checkpoint.name == f"checkpoint-{state.global_step}":
                root = self.output_dir
                pending = root / ".best_checkpoint.json.tmp"
                pending.write_text(json.dumps({
                    "checkpoint": checkpoint.name,
                    "epoch": state.epoch,
                    "global_step": state.global_step,
                    "eval_loss": state.best_metric,
                }, indent=2) + "\n", encoding="utf-8")
                pending.replace(root / "best_checkpoint.json")
        return control
