#!/bin/bash
set -euo pipefail

OVERLAY_DIR=/patch/dspark_overlay
VLLM_DIR=/usr/local/lib/python3.12/dist-packages/vllm

echo "DSPARK_SETUP copying overlay files..."

cp "$OVERLAY_DIR/vllm/models/deepseek_v4/nvidia/dspark.py" \
   "$VLLM_DIR/models/deepseek_v4/nvidia/dspark.py"
cp "$OVERLAY_DIR/vllm/models/deepseek_v4/nvidia/dspark_kernels.py" \
   "$VLLM_DIR/models/deepseek_v4/nvidia/dspark_kernels.py"
cp "$OVERLAY_DIR/vllm/v1/spec_decode/dspark.py" \
   "$VLLM_DIR/v1/spec_decode/dspark.py"
cp "$OVERLAY_DIR/vllm/v1/spec_decode/dspark_proposer.py" \
   "$VLLM_DIR/v1/spec_decode/dspark_proposer.py"
cp "$OVERLAY_DIR/vllm/v1/worker/gpu_model_runner.py" \
   "$VLLM_DIR/v1/worker/gpu_model_runner.py"

python3 -m py_compile "$VLLM_DIR/models/deepseek_v4/nvidia/dspark.py"
python3 -m py_compile "$VLLM_DIR/models/deepseek_v4/nvidia/dspark_kernels.py"
python3 -m py_compile "$VLLM_DIR/v1/spec_decode/dspark.py"
python3 -m py_compile "$VLLM_DIR/v1/spec_decode/dspark_proposer.py"
python3 -m py_compile "$VLLM_DIR/v1/worker/gpu_model_runner.py"

echo "DSPARK_SETUP_OK"
