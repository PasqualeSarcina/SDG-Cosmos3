#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE="qwen35-lora:cu130"

mkdir -p checkpoints outputs

docker build \
    -f docker/Dockerfile.lora \
    -t "$IMAGE" \
    .

docker run \
    --rm \
    -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$PWD:/workspace" \
    -v /workspace/.venv \
    -v "$HOME/.cache:/root/.cache" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -w /workspace \
    "$IMAGE" \
    "$@"
