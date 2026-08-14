# vLLM 0.26.0 + EXL3 Trellis backend + DSPark native support
#
# Layers:
# 1. Base: vllm/vllm-openai:v0.26.0 (has native DSPark, DeepSeek V4, EPLB, Eagle3)
# 2. SparkInfer (brandonmmusic-max/b12x@exl3-trellis-fused -> sparkinfer pkg)
# 3. ExLlamaV3 v1.4.1 (cloned at build + ARM64 patch set, compiled for SM121)
# 4. InstantTensor (fast weight loading, built from sdist on ARM64)
# 5. EXL3 vLLM backend overlay (exl3.py quantization + model deltas)

FROM vllm/vllm-openai:v0.26.0

# git is needed to pip-install the sparkinfer repo from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# --- Layer 2: SparkInfer (pure Python + Triton JIT kernels) ---
# Pinned to the exact commit the working vllm-exl3-spark-sm121 image was
# built from (branch exl3-trellis-fused at 669a12dd, sparkinfer 1.0.1).
RUN pip install --no-deps --no-cache-dir \
    git+https://github.com/brandonmmusic-max/b12x.git@669a12ddc7cf3021e91a25f398b1a883b703fd12

# --- Layer 3: ExLlamaV3 v1.4.1 (compiled from source for ARM64 SM121) ---
# Instead of vendoring the full source, clone the pinned upstream tag and
# apply the ARM64 patch set from exllamav3-patches/ (see that folder's
# README.md for what each patch does and how to regenerate them).
ARG EXLLAMAV3_TAG=v1.4.1
ENV TORCH_CUDA_ARCH_LIST="12.1"
# The v0.26.0 base ships CUDA headers under nvidia/cu13/include (the
# /usr/local/cuda/include symlink only has a partial set). Without these on the
# include path, torch's ATen headers fail on cusparse.h/cublas.h.
ENV CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
ENV C_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
COPY exllamav3-patches/ /patches/
RUN git clone --depth 1 --branch "${EXLLAMAV3_TAG}" \
        https://github.com/turboderp-org/exllamav3.git /tmp/exllamav3 && \
    cd /tmp/exllamav3 && \
    for p in /patches/*.patch; do git apply "$p"; done && \
    rm -rf /patches && \
    pip install --no-build-isolation --no-cache-dir /tmp/exllamav3/ && \
    rm -rf /tmp/exllamav3/

# --- Layer 4: InstantTensor fast weight loading ---
# instanttensor has no aarch64 wheel; build from the sdist (nvcc + CUDA headers
# are already in scope from the ExLlamaV3 layer, TORCH_CUDA_ARCH_LIST=12.1).
# The patch preserves scalar tensor shapes and clones each view before
# InstantTensor reuses its loader buffer — required for EXL3 mcg/mul1 scalars.
# vLLM v0.26.0 already wires `--load-format instanttensor` (weight_utils.py),
# but its safe_open() call passes copy=True which instanttensor 0.1.5 does not
# accept; patch_instanttensor_loader.py drops it (the package patch already
# clones every tensor, so copy is redundant — matches the 0xSero/LIL recipe).
COPY instanttensor.patch /tmp/instanttensor.patch
RUN pip install --no-cache-dir instanttensor==0.1.5 && \
    patch --dry-run -p1 -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/instanttensor.patch && \
    patch -p1 -d /usr/local/lib/python3.12/dist-packages \
      < /tmp/instanttensor.patch && \
    rm /tmp/instanttensor.patch
COPY patch_instanttensor_loader.py /tmp/patch_instanttensor_loader.py
RUN python3 /tmp/patch_instanttensor_loader.py

# --- Layer 5: EXL3 quantization module ---
COPY overlay/vllm/model_executor/layers/quantization/exl3.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py

# --- Register exl3 in quantization registry + config overrides ---
COPY patch_exl3_init.py /tmp/patch_exl3_init.py
RUN python3 /tmp/patch_exl3_init.py
COPY patch_model_config.py /tmp/patch_model_config.py
RUN python3 /tmp/patch_model_config.py

# --- Register VLLM_EXL3_* env vars ---
COPY patch_envs_exl3.py /tmp/patch_envs_exl3.py
RUN python3 /tmp/patch_envs_exl3.py

# --- Apply EXL3 deltas onto native DeepseekV4 model.py (keeps DSPark) ---
COPY patch_model_exl3.py /tmp/patch_model_exl3.py
RUN python3 /tmp/patch_model_exl3.py

# --- REAP K216 router fallback (216 experts -> pure Torch) ---
COPY patch_router_k216.py /tmp/patch_router_k216.py
RUN python3 /tmp/patch_router_k216.py

# --- REAP K216 fused router kernel (216 experts -> fused DSv4 Triton topk) ---
COPY patch_dsv4_topk_k216.py /tmp/patch_dsv4_topk_k216.py
RUN python3 /tmp/patch_dsv4_topk_k216.py

# --- DSPark draft-layer construction: build with the draft's expert count
#     (compact K64 draft) instead of the target's 216 ---
COPY patch_dspark_exl3.py /tmp/patch_dspark_exl3.py
RUN python3 /tmp/patch_dspark_exl3.py

# --- SM12x DSpark decode fix: pad the non-causal SWA index width (256) up to
#     a topk FlashInfer's sparse_mla_sm120_decode_dsv4 kernel instantiates
#     ({128, 512, 1024}); otherwise the 5-token draft decode falls through to
#     the prefill orchestrator which asserts num_tokens > 64 ---
COPY patch_sparse_swa_sm12x.py /tmp/patch_sparse_swa_sm12x.py
RUN python3 /tmp/patch_sparse_swa_sm12x.py

# --- Stamp version with dev postfix (shows in startup banner) ---
COPY patch_version.py /tmp/patch_version.py
RUN python3 /tmp/patch_version.py

# --- Compact DSPark draft tooling (K64) ---
COPY scripts/build_dspark_draft.py /opt/recipe/build_dspark_draft.py
COPY scripts/serve_ds4_exl3.sh /opt/recipe/serve_ds4_exl3.sh
RUN chmod +x /opt/recipe/serve_ds4_exl3.sh

# --- Verify ---
COPY verify_exl3.py /tmp/verify_exl3.py
RUN python3 /tmp/verify_exl3.py

CMD ["vllm", "serve"]
