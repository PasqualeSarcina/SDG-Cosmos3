"""Persist and validate the data/template contract independently of checkpoints."""

import json
from pathlib import Path

from src.lora.dataset import USER_PROMPT
from src.lora.thinking_dataset import STUDENT_REASONING_EFFORT, sha256


CONFIG_NAME = "thinking_config.json"


def make_training_protocol(args, tokenizer, audits):
    template = tokenizer.get_chat_template()
    records = []
    for split in ("train", "validation"):
        if args.thinking_mode == "supervised":
            records.extend({"split": split, **r} for r in audits[split]["records"])
        else:
            # Preserve order as well as content: changes can alter batch-normalized
            # field weights and the resumed dataloader's sampling sequence.
            records.extend({"split": split, "source_sha256": sha256(p.read_bytes())}
                           for p in getattr(args, f"{split}_paths"))
    return {
        "schema_version": 1, "thinking_mode": args.thinking_mode,
        "thinking_weight": args.thinking_weight if args.thinking_mode == "supervised" else None,
        "response_weight": args.response_weight if args.thinking_mode == "supervised" else 1.0,
        "field_length_exponent": args.field_length_exponent, "structure_weight": args.structure_weight,
        "reasoning_effort": STUDENT_REASONING_EFFORT if args.thinking_mode == "supervised" else None,
        "model": "Qwen/Qwen3.8-27B", "model_revision": args.model_revision or "unspecified",
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash", "unknown"),
        "chat_template_sha256": sha256(template.encode("utf-8")),
        "system_prompt_sha256": sha256(args.system_prompt.encode("utf-8")),
        "user_prompt_sha256": sha256(USER_PROMPT.encode("utf-8")),
        "dataset_sha256": sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")),
        "protocol": audits["train"]["protocol"] if args.thinking_mode == "supervised" else None,
    }


def check_resume(checkpoint, current):
    if checkpoint is None:
        return
    path = Path(checkpoint) / CONFIG_NAME
    if not path.is_file():
        if current["thinking_mode"] != "off":
            raise ValueError("A supervised run cannot resume a legacy checkpoint without thinking_config.json.")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved != current:
        differing = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise ValueError("Incompatible checkpoint training protocol: " + ", ".join(differing))


def save_training_protocol(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / ("." + CONFIG_NAME + ".tmp")
    pending.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(directory / CONFIG_NAME)
