"""Copy a text-only Qwen3.8-27B PEFT adapter into the vLLM multimodal namespace.

Only safetensors keys change. Tensor descriptors and payload bytes stay identical.
No model, GPU, Torch, Unsloth or vLLM import is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)
SOURCE_PREFIX = "base_model.model.model.layers."
TARGET_PREFIX = "base_model.model.model.language_model.layers."
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja", "chat_template.json",
)
CHUNK_SIZE = 8 * 1024 * 1024


def _read_header(path: Path) -> tuple[dict, int]:
    from safetensors import SafetensorError, safe_open

    # The library checks the format, shapes, offsets and payload length without
    # materializing tensors. In particular, this does not convert BF16 to NumPy.
    try:
        with safe_open(str(path), framework="numpy") as tensors:
            keys = set(tensors.keys())
    except SafetensorError as exc:
        raise ValueError(f"Invalid safetensors file: {path}: {exc}") from exc
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 100_000_000:
            raise ValueError("Unexpectedly large safetensors header.")
        header = json.loads(stream.read(length))
    if keys != set(header) - {"__metadata__"}:
        raise ValueError("Safetensors key validation failed.")
    return header, 8 + length


def _converted_header(header: dict, rank: int) -> tuple[dict, list[dict]]:
    result = {}
    pairs: dict[str, set[str]] = {}
    examples = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            result[name] = descriptor
            continue
        if name.startswith(TARGET_PREFIX):
            raise ValueError("This adapter already has converted keys; use the original checkpoint.")
        if not name.startswith(SOURCE_PREFIX):
            raise ValueError(f"Unsupported tensor prefix: {name}")
        if name.endswith(".lora_A.weight"):
            kind = "A"
        elif name.endswith(".lora_B.weight"):
            kind = "B"
        else:
            raise ValueError(f"Expected a plain LoRA A/B tensor: {name}")
        shape = descriptor["shape"]
        if len(shape) != 2 or shape[0 if kind == "A" else 1] != rank:
            raise ValueError(f"Unexpected rank/shape for {name}: {shape}")
        module = name.rsplit(".lora_", 1)[0]
        pairs.setdefault(module, set()).add(kind)
        converted = TARGET_PREFIX + name[len(SOURCE_PREFIX):]
        if converted in result:
            raise ValueError(f"Tensor name collision: {converted}")
        result[converted] = descriptor
        if len(examples) < 3:
            examples.append({"original": name, "converted": converted})
    if not pairs or any(kinds != {"A", "B"} for kinds in pairs.values()):
        raise ValueError("Each adapted module must contain both LoRA A and B tensors.")
    return result, examples


def _payload_digest(path: Path, offset: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        while data := stream.read(CHUNK_SIZE):
            digest.update(data)
            size += len(data)
    return digest.hexdigest(), size


def convert_adapter(source: Path, output: Path) -> dict:
    """Write a new adapter directory; refuse existing destinations and unknown layouts."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser()
    if not source.is_dir():
        raise ValueError(f"Input must be an adapter directory: {source}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    output = output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the original checkpoint directory.")
    config_bytes = (source / "adapter_config.json").read_bytes()
    config = json.loads(config_bytes.decode("utf-8-sig"))
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise ValueError("Expected a PEFT LORA adapter.")
    if config.get("base_model_name_or_path") not in {"unsloth/Qwen3.8-27B", "Qwen/Qwen3.8-27B"}:
        raise ValueError("This converter is specific to text-only Qwen3.8-27B adapters.")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("This converter requires uniform rank and alpha (empty rank/alpha patterns).")
    if (config.get("modules_to_save") or config.get("use_dora") or config.get("lora_bias")
            or config.get("bias", "none") != "none"):
        raise ValueError("Expected plain LoRA A/B weights without DoRA, saved modules or biases.")
    rank = config.get("r")
    if type(rank) is not int or rank < 1:
        raise ValueError("Adapter rank must be a positive integer.")
    weights = source / "adapter_model.safetensors"
    original, source_offset = _read_header(weights)
    converted, examples = _converted_header(original, rank)
    header = json.dumps(converted, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    created: list[Path] = []
    try:
        destination = output / weights.name
        created.append(destination)
        source_digest = hashlib.sha256()
        payload_bytes = 0
        with weights.open("rb") as reader, destination.open("xb") as writer:
            writer.write(struct.pack("<Q", len(header)))
            writer.write(header)
            reader.seek(source_offset)
            while data := reader.read(CHUNK_SIZE):
                writer.write(data)
                source_digest.update(data)
                payload_bytes += len(data)

        # Read back independently; unchanged offsets plus byte-identical payload
        # prove that every tensor kept its original values, dtype and shape.
        written_header, written_offset = _read_header(destination)
        written_digest, written_bytes = _payload_digest(destination, written_offset)
        if (written_header != converted or written_digest != source_digest.hexdigest()
                or written_bytes != payload_bytes):
            raise RuntimeError("Converted adapter failed tensor payload verification.")

        config_out = output / "adapter_config.json"
        created.append(config_out)
        with config_out.open("xb") as stream:
            stream.write(config_bytes)
        copied = []
        for name in TOKENIZER_FILES:
            path = source / name
            if path.is_file():
                target = output / name
                created.append(target)
                with path.open("rb") as reader, target.open("xb") as writer:
                    while data := reader.read(CHUNK_SIZE):
                        writer.write(data)
                copied.append(name)

        tensor_count = len(converted) - ("__metadata__" in converted)
        report = {
            "source": str(source), "output": str(output),
            "purpose": "Qwen3.8-27B adapter for the vLLM Qwen3_5ForConditionalGeneration wrapper",
            "source_prefix": SOURCE_PREFIX, "target_prefix": TARGET_PREFIX,
            "tensor_count": tensor_count, "adapted_modules": tensor_count // 2,
            "rank": rank, "lora_alpha": config.get("lora_alpha"),
            "dtype_counts": dict(Counter(v["dtype"] for k, v in converted.items() if k != "__metadata__")),
            "payload_bytes": payload_bytes, "source_payload_sha256": source_digest.hexdigest(),
            "output_payload_sha256": written_digest, "tensor_payload_identical": True,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "config_preserved": True, "copied_tokenizer_files": copied, "examples": examples,
            "runtime_inference_verified": False,
        }
        manifest = output / "conversion_manifest.json"
        created.append(manifest)
        with manifest.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return report
    except BaseException:
        # Only delete individual files this invocation created, never recursively
        # remove a directory or touch the source checkpoint.
        for path in reversed(created):
            path.unlink(missing_ok=True)
        output.rmdir()
        raise


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Original local PEFT adapter/checkpoint.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="New directory for the vLLM adapter copy.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = convert_adapter(args.input, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("Conversion failed: %s", exc)
        return 1
    log.info("Converted %d tensors (%d modules) to %s", result["tensor_count"],
             result["adapted_modules"], result["output"])
    log.info("Tensor data VERIFIED byte-for-byte: SHA-256 %s", result["output_payload_sha256"])
    log.info("Conversion verified. Actual vLLM adapter activation must still be checked at inference time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
