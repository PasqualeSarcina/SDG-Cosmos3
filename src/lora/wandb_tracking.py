"""Keep the W&B run identity with ordinary and recovery trainer checkpoints."""

import json
import logging
from pathlib import Path

from transformers.integrations import WandbCallback


log = logging.getLogger(__name__)
RUN_FILE = "wandb_run.json"


def _read_identity(directory):
    path = Path(directory) / RUN_FILE
    if not path.is_file():
        return None
    identity = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(identity, dict) or identity.get("schema_version") != 1
            or any(not isinstance(identity.get(key), str) or not identity[key].strip()
                   for key in ("id", "project"))
            or (identity.get("entity") is not None and not isinstance(identity["entity"], str))):
        raise ValueError(f"Invalid W&B run identity: {path}")
    return identity


def _save_identity(directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / ("." + RUN_FILE + ".tmp")
    pending.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    pending.replace(directory / RUN_FILE)


def copy_wandb_identity(run_directory, checkpoint_directory):
    identity = _read_identity(run_directory)
    if identity is not None:
        _save_identity(checkpoint_directory, identity)


class ResumableWandbCallback(WandbCallback):
    """Reuse the saved run; leave metric reporting to Transformers' integration."""

    def __init__(self, resume_from_checkpoint=None):
        super().__init__()
        self.resume_from_checkpoint = resume_from_checkpoint

    def setup(self, args, state, model, **kwargs):
        if state.is_world_process_zero and self.resume_from_checkpoint is not None:
            checkpoint = Path(self.resume_from_checkpoint)
            # Prefer the checkpoint's own identity, including after it is moved.
            # The run-root fallback also covers checkpoints predating the copy.
            root = checkpoint.parent.parent if checkpoint.parent.name == "recovery" else checkpoint.parent
            identity = _read_identity(checkpoint) or _read_identity(root)
            if identity is None:
                log.warning("No %s found for %s; W&B cannot automatically identify the previous run. "
                            "Use WANDB_RUN_ID and WANDB_RESUME=must to select it explicitly.", RUN_FILE, checkpoint)
            elif self._wandb.run is not None:
                active = self._wandb.run
                if (active.id != identity["id"] or active.project != identity["project"]
                        or (identity.get("entity") is not None and active.entity != identity["entity"])):
                    raise ValueError("The active W&B run differs from the checkpoint's saved run.")
            else:
                self._wandb.init(id=identity["id"], project=identity["project"],
                                 entity=identity.get("entity"), resume="must")
        super().setup(args, state, model, **kwargs)
        if state.is_world_process_zero:
            run = self._wandb.run
            if run is not None and getattr(run, "disabled", False) is not True:
                _save_identity(args.output_dir, dict(schema_version=1, id=run.id,
                                                     project=run.project, entity=run.entity or None))
