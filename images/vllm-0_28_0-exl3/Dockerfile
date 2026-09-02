# vLLM latest-main (0.28.x) + EXL3 Trellis backend + DSPark native support.
#
# Built entirely FROM SOURCE on a DGX Spark (GB10 / SM121 / ARM64 / CUDA 13):
#   * vllm wheel compiled in a builder stage from a pinned upstream commit
#     (no vllm/vllm-openai base image; the runtime image is a minimal
#     nvidia/cuda base with only what serving + JIT need).
#   * SparkInfer (b12x exl3-trellis-fused fork, pinned commit) +
#     ExLlamaV3 (ARM64 SM121 patch set, compiled for SM121) +
#     InstantTensor (fast mmap weight loading).
#   * 11 source patches (vllm-patches/) + the exl3.py quantization overlay
#     are applied to the wheel build tree, so the wheel itself carries them.
#
# Bump guide: change VLLM_COMMIT below, then rebuild. If an upstream change
# drifted an anchor, `git apply` fails loudly on the offending patch -- fix
# the hunk (or drop the patch if upstream adopted it) and rebuild.
# Upstream PRs already merged in the pinned commit are dropped from
# vllm-patches/ (see README.md patch table for the triage ledger).

FROM nvidia/cuda:13.0.3-devel-ubuntu24.04 AS vllm-builder

# Pinned vllm-project/vllm main commit (bump = change the ARG below).
ARG VLLM_COMMIT=62588e0592ad5af8d3d4a536ace01afb44ebaed5
ARG VLLM_SCM_VERSION=0.28.0+exl3.dspark.sm121
ARG TORCH_CUDA_ARCH_LIST=12.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git ca-certificates cmake ninja-build \
        python3 python3-dev python3-venv python3-pip \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# vllm-rs (Rust frontend) is built by setup.py; Rust channel from rust-toolchain.toml.
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain 1.95 \
    && ln -sf /root/.cargo/bin/cargo /usr/local/bin/cargo \
    && ln -sf /root/.cargo/bin/rustc /usr/local/bin/rustc

WORKDIR /build

# Clone ONLY the pinned commit (shallow, no history/tags).
RUN git init vllm && cd vllm \
    && git remote add origin https://github.com/vllm-project/vllm.git \
    && git fetch --depth 1 origin "${VLLM_COMMIT}" \
    && git checkout -q FETCH_HEAD

# Apply the EXL3/DSPark source patch set to the build tree.
COPY vllm-patches/ /build/vllm-patches/
RUN cd vllm && for p in /build/vllm-patches/*.patch; do git apply "$p"; done
# EXL3 quantization overlay (new module, copied into the tree).
COPY overlay/ /build/overlay/
RUN cp /build/overlay/vllm/model_executor/layers/quantization/exl3.py \
       /build/vllm/vllm/model_executor/layers/quantization/exl3.py \
    && ls -la /build/vllm/vllm/model_executor/layers/quantization/exl3.py

# torch + build requirements (mirror of requirements/build/cuda.txt).
RUN python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0 \
    && python3 -m pip install --no-cache-dir \
        cmake ninja packaging "setuptools>=77.0.3,<81.0.0" setuptools-scm \
        setuptools-rust wheel jinja2 protobuf

# Compile the wheel (single arch: SM121; version stamped via setuptools-scm).
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_SCM_VERSION}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
# vllm's setup.py computes ninja jobs as MAX_JOBS // NVCC_THREADS (so a
# 20-core build = 10 parallel compiles at ~7 GiB peak each).
ENV MAX_JOBS=20 NVCC_THREADS=2
RUN cd vllm && python3 setup.py bdist_wheel --dist-dir=/wheels --py-limited-api=cp38

# ────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ARG VLLM_SCM_VERSION=0.28.0+exl3.dspark.sm121
ARG EXLLAMAV3_TAG=v1.4.6
ARG TORCH_CUDA_ARCH_LIST=12.1

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CUDA_VERSION=13.0.3

# Python + CUDA dev tools (nvcc/headers for runtime JIT: FlashInfer autotune,
# DeepGEMM, exllamav3 compile) + git (sparkinfer pip install from GitHub).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip git \
        cuda-nvcc-13-0 cuda-cudart-13-0 cuda-nvrtc-13-0 cuda-cuobjdump-13-0 \
        libcurand-dev-13-0 libcublas-dev-13-0 libcusparse-dev-13-0 \
        libcusolver-dev-13-0 \
        libnuma-dev numactl \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

WORKDIR /workspace

# ── vLLM wheel + pinned runtime deps ──────────────────────────────────────
# requirements/cuda.txt carries the flashinfer.ai extra-index needed for
# flashinfer-cubin, so install it BEFORE the wheel (deps then resolve from
# the installed set, exactly like upstream's runtime stage).
COPY --from=vllm-builder /wheels /wheels
COPY --from=vllm-builder /build/vllm/requirements/common.txt /tmp/common.txt
COPY --from=vllm-builder /build/vllm/requirements/cuda.txt /tmp/cuda.txt
RUN python3 -m pip install --no-cache-dir \
        -r /tmp/cuda.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
    && python3 -m pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /tmp/common.txt /tmp/cuda.txt /wheels

# ── SparkInfer (Trellis MoE + prefill kernels, pinned fork commit) ────────
# Pinned to the exact commit the working vllm-exl3-spark-sm121 image was
# built from (branch exl3-trellis-fused, commit pinned in the URL below).
RUN python3 -m pip install --no-deps --no-cache-dir \
    git+https://github.com/brandonmmusic-max/b12x.git@669a12ddc7cf3021e91a25f398b1a883b703fd12

# SparkInfer kernel backports from local-inference-lab/b12x (fail-closed
# anchor scripts, applied to the installed sparkinfer package):
# #150: preallocate the W4A16 route histogram (expert_counts) so the fast
#      count+prefix kernel runs during CUDA-graph capture without allocation.
# #228: keep graphed tiny-decode routes with inactive expert ids in range
#      (graph padding / non-local EP routes) to avoid OOB weight reads.
COPY sparkinfer-patches/ /tmp/sparkinfer-patches/
RUN python3 /tmp/sparkinfer-patches/patch_w4a16_expert_counts.py && \
    python3 /tmp/sparkinfer-patches/patch_tiny_decode_route_clamp.py && \
    rm -rf /tmp/sparkinfer-patches

# ── ExLlamaV3 (compiled from source for ARM64 SM121) ──────────────────────
# Cloned at build from the pinned upstream tag + ARM64 patch set
# (exllamav3-patches/README.md documents each patch).
COPY exllamav3-patches/ /patches/
RUN git clone --depth 1 --branch "${EXLLAMAV3_TAG}" \
        https://github.com/turboderp-org/exllamav3.git /tmp/exllamav3 && \
    cd /tmp/exllamav3 && \
    for p in /patches/*.patch; do git apply "$p"; done && \
    rm -rf /patches && \
    MAX_JOBS=8 python3 -m pip install --no-build-isolation --no-cache-dir /tmp/exllamav3/ && \
    rm -rf /tmp/exllamav3/

# ── InstantTensor fast weight loading (built from sdist on ARM64) ──────────
# The scalar-shape view + clone fixes we used to patch into the older release
# are upstreamed; vllm passes copy=True natively.
RUN python3 -m pip install --no-cache-dir instanttensor==0.1.9

# ── Compact DSPark draft tooling ──────────────────────────────────────────
# The draft builder (build_dspark_draft.py) and the container entrypoint
# (serve_ds4_exl3.sh) are NOT baked in: both live in the repo under
# models/files/ and are volume-mounted read-only over /opt/recipe/ by the
# model YAML, so draft-building/serving logic can change without an image
# rebuild.

# ── Verify ────────────────────────────────────────────────────────────────
COPY verify_exl3.py /tmp/verify_exl3.py
RUN python3 /tmp/verify_exl3.py

CMD ["vllm", "serve"]