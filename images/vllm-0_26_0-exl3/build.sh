#!/usr/bin/env bash
set -euo pipefail

# Build this custom vLLM image from the current directory.
# Usage: ./build.sh [image-tag]   (default: vllm-exl3-v26:latest)

cd "$(dirname "$0")"
IMAGE="${1:-vllm-exl3-v26:latest}"

docker build -t "$IMAGE" .
echo "Built $IMAGE"
