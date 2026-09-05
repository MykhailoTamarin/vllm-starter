#!/usr/bin/env python3
# Runtime patcher for the OFFICIAL nvidia/Qwen3.8-Flash-Next-NVFP4 checkpoint on
# vllm/vllm-openai:qwen38-flash-next, single DGX Spark (GB10, TP=1).
#
# Runs INSIDE the container before `vllm serve` and edits the installed vLLM
# in place (same pattern as patch_flashinfer_b12x_fp8.py). Idempotent: each
# target is skipped once its marker is present, so restarts are cheap.
#
# What is patched and why (minimal set for the official checkpoint):
#
# 1. PLE dispatch for MIXED_PRECISION ModelOpt checkpoints (ple_layer.py).
#    The official checkpoint is quant_algo=MIXED_PRECISION: NVFP4 routed
#    experts + FP8 MTP experts + FP8 per-tensor PLE table. The image's stock
#    _get_ple_embedding_quant_method only selects the FP8 PLE method for
#    whole-checkpoint Fp8Config, so the PLE table would get no quant method
#    and fail to load. This is the fix NVIDIA's model card points at
#    (upstream vLLM commit d4d703caf, PR #54882), ported onto the image's
#    qwen3_8_flash_next implementation.
#
# 1b. MTP draft MoE dispatch for the official checkpoint (modelopt.py).
#    The checkpoint quantizes the MTP routed experts as 128x128
#    block-scaled FP8 (quant_algo "FP8_BLOCK_SCALES", copied byte-for-byte
#    from the official FP8 checkpoint). Two gaps break the MTP draft:
#      - the draft layer is constructed at prefix `mtp.layers.<48>` while
#        the checkpoint lists the same tensors under `mtp.layers.<0>`, so
#        _resolve_quant_algo finds nothing;
#      - get_quant_method only dispatches "FP8"/"NVFP4"/"W4A16_NVFP4"/
#        "MXFP8", and ModelOptFp8MoEMethod is per-tensor anyway.
#    Fix: add the checkpoint-numbered mtp candidate and dispatch
#    FP8_BLOCK_SCALES to the standard Fp8MoEMethod, which fully supports
#    RoutedExperts + 128x128 block scales (the same path the official FP8
#    checkpoint's MoE uses).
#
# 1c. Async PLE MTP metadata transfers (short_conv_attn.py).
#    Port of upstream vLLM edc0fb7e0 (#55054): the PleShortConv metadata
#    builder moved per-step CPU tensors (spec/non-spec request indices,
#    accepted-token counts) to the GPU with synchronous .to(device) calls
#    on every spec-decode step. Replaced with async_tensor_h2d so the
#    copies overlap GPU work instead of serialising the decode hot path.
#    The image's file already imports async_tensor_h2d and the patch's
#    context matches byte-for-byte (git apply --check verified).
#
# 1d. Reuse HC combine-norm for MTP input (ops/hc.py, hyperconnection.py,
#     model.py, mtp.py). Port of upstream vLLM fd4a15126 (#54687): the MTP
#     draft eagerly broadcast the embedding residual over all 4 HC branches
#     (inputs_embeds.unsqueeze(-2) + hidden_states) and then ran a separate
#     combine+RMSNorm. Instead the embedding is passed as a pending HC
#     combine into the draft's first layer, fusing into the existing
#     combine+norm Triton kernels (new unit-weight path when injection is
#     None) — one less materialization per draft step. _hc_combine_norm_
#     kernel also flattens to a 1D grid. Main-model behavior is unchanged
#     (its mixers always carry an injection).
#
# 2. GB10 PLE-offload fixes + mmap packed table (protocol/ple_offload_layer/
#    connector/worker). Applied by Mia's generators VERBATIM
#    (patch_ple_offload.py, AGPL-3.0-or-later, vendored from
#    MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark):
#      - GB10 reports CAN_USE_STREAM_MEM_OPS=0, so the stock cuStreamWait/
#        WriteValue32 handshake deadlocks the GPU worker right after CUDA
#        graph capture. Replaced with a host-side handshake (the GPU worker
#        blocks until the CPU worker publishes a sequence number).
#      - The PLE table is served from a pre-packed, memory-mapped file
#        (VLLM_PLE_PACKED_TABLE_DIR) instead of anonymous RAM, advised
#        MADV_RANDOM so 160-byte row lookups don't fault 64 KiB windows.
#
# 3. FP8-aware packed-table attach (worker.py). Adaptation of Mia's
#    _attach_packed_table: the official checkpoint's PLE rows are full-width
#    FP8-e4m3 (160 B/row) with one global BF16 scale applied on the GPU, not
#    Mia's packed NVFP4 layout (90 B/row). The mmap is viewed as
#    float8_e4m3fn so the CPU fast path's index_select lands directly in the
#    fp8 IPC output buffer.
#
# 4. QSA FP8 KV cache (ops/qsa.py + qsa.py). Applied by Mia's generator
#    verbatim (patch_qsa_fp8_kv.py; FP8-KV approach credited to
#    lancelind/qwen3.8-Flash-DGX). INERT unless --kv-cache-dtype is fp8:
#    KV_QUANT_MODE is a tl.constexpr, so the BF16 path compiles identically.
#    Included so `--kv-cache-dtype fp8` can be enabled without re-shipping.
#
# NOT patched (not needed for the official checkpoint):
#   - Mia's NVFP4-PLE machinery (patch_ple_layer.py): the official PLE table
#     is FP8, and the stock image already handles FP8 PLE load + offload.
#   - Mia's ModelOpt MXFP8 fallback (patch_modelopt_mxfp8.py): the official
#     checkpoint has no MXFP8 layers (attention/vision stay BF16).
#
# Failures are loud: any missing file or unmatched anchor aborts before
# `vllm serve` starts.
import os
import shutil
import sys
import tempfile

import vllm

VLLM_DIR = os.path.dirname(vllm.__file__)
PKG = os.path.join(VLLM_DIR, "models", "qwen3_8_flash_next")
HERE = os.path.dirname(os.path.abspath(__file__))

# target rel-path -> (marker that proves the patch is applied, mia generator)
MIA_TARGETS = {
    "model_executor/layers/ple_offload_layer.py": (
        "CAN_USE_STREAM_MEM_OPS=0", "patch_ple_offload.py",
    ),
    "v1/ple_offload/protocol.py": (
        "done_flag: torch.Tensor | None = None", "patch_ple_offload.py",
    ),
    "v1/ple_offload/connector.py": (
        "_wait_done", "patch_ple_offload.py",
    ),
    "v1/ple_offload/worker.py": (
        "_attach_packed_table", "patch_ple_offload.py",
    ),
    "models/qwen3_8_flash_next/nvidia/ops/qsa.py": (
        "_qsa_as_fp8", "patch_qsa_fp8_kv.py",
    ),
    "models/qwen3_8_flash_next/nvidia/qsa.py": (
        '"fp8_e4m3"', "patch_qsa_fp8_kv.py",
    ),
}

# Mia's generator file names -> the staged names their code expects.
MIA_STAGE_FILES = {
    "patch_ple_offload.py": {
        # installed path -> staged orig name (relative to the staging dir)
        "model_executor/layers/ple_offload_layer.py": "ple_offload/orig/ple_offload_layer.py",
        "v1/ple_offload/connector.py": "ple_offload/orig/connector.py",
        "v1/ple_offload/worker.py": "ple_offload/orig/worker.py",
        "v1/ple_offload/protocol.py": "ple_offload/orig/protocol.py",
    },
    "patch_qsa_fp8_kv.py": {
        "models/qwen3_8_flash_next/nvidia/ops/qsa.py": "qsa_ops_patched.py.orig",
        "models/qwen3_8_flash_next/nvidia/qsa.py": "qsa_nvidia_patched.py.orig",
    },
}

# Mia's NVFP4 _attach_packed_table block (inserted by her worker patch), from
# "    @staticmethod" through the "    def accept_registrations(" that follows
# it. Replaced by the FP8-aware version below.
MIA_ATTACH_HEAD = "    @staticmethod\n    def _attach_packed_table("
MIA_ATTACH_TAIL = "    def accept_registrations(\n"

FP8_ATTACH = '''    @staticmethod
    def _attach_packed_table(name: str, layer: PleOffloadLayer, path: str) -> None:
        """Replace the shard-loaded table with a read-only memory map."""
        import numpy as np

        meta = json.load(open(path + ".json"))
        rows, width = int(meta["total_rows"]), int(meta["row_width"])
        dtype_name = str(meta.get("dtype", "u8"))
        emb = layer.ngram_embedding
        quant_method = getattr(emb, "quant_method", None)
        expected_width = getattr(quant_method, "packed_row_width", None)
        if expected_width is None:
            weight = getattr(emb, "weight", None)
            if weight is not None and weight.dim() == 2:
                expected_width = weight.shape[-1]
        if expected_width is not None and expected_width != width:
            raise RuntimeError(
                f"PLE {name}: packed table row width {width} != "
                f"expected {expected_width}"
            )
        if emb.weight.shape[0] != rows:
            raise RuntimeError(
                f"PLE {name}: packed table has {rows} rows, model expects "
                f"{emb.weight.shape[0]}"
            )
        if os.path.getsize(path) != rows * width:
            raise RuntimeError(f"PLE {name}: packed table size mismatch")
        mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(rows, width))
        # Tens of GiB of randomly-accessed 160/90-byte rows against a far
        # smaller page cache. Default mmap behaviour faults in a ~64 KiB
        # window per touched row (fault-around); nearly all of it is never
        # read. Declaring the access random makes each fault cost one page.
        try:
            import mmap as _mmap_mod

            mm._mmap.madvise(_mmap_mod.MADV_RANDOM)
            _advice = "MADV_RANDOM"
        except Exception as _exc:  # advisory only, never fatal
            _advice = f"no madvise ({_exc})"
        table = torch.from_numpy(mm)  # zero-copy, file-backed, evictable
        if dtype_name in ("fp8", "fp8_e4m3", "float8_e4m3fn"):
            # FP8 PLE rows: bytes are float8_e4m3fn values; the global scale
            # stays on the GPU and is applied in _dequantize_embeddings.
            table = table.view(torch.float8_e4m3fn)
        elif dtype_name not in ("u8", "nvfp4"):
            raise RuntimeError(
                f"PLE {name}: unknown packed table dtype {dtype_name!r}"
            )
        emb._packed_table = table
        emb._packed_table_mmap = mm
        # Release the never-touched anonymous allocations.
        emb.weight.data = torch.empty(0, dtype=emb.weight.dtype)
        ws = getattr(emb, "weight_scale", None)
        if ws is not None and ws.dim() == 2:
            ws.data = torch.empty(0, dtype=ws.dtype)
        logger.info(
            "PLE %s: mmap table attached (%d rows x %d B = %.2f GiB, %s) [%s]",
            name,
            rows,
            width,
            rows * width / 2**30,
            dtype_name,
            _advice,
        )

    def accept_registrations(
'''

PLE_LAYER_REL = "models/qwen3_8_flash_next/nvidia/ple_layer.py"

# CPU-worker offload fast path: prefer the memory-mapped pre-packed table
# (emb._packed_table, attached by the patched worker) over the shard-loaded
# weight parameter, whose storage is released once the table is attached.
# Taken from Mia's patch_ple_layer.py (offload fast path edit); the final
# fallback is the stock index_select and is byte-identical for FP8 rows
# (row_width == head_dim there). Without it, index_select would run against
# the emptied weight parameter after the mmap table is attached.
PLE_FASTPATH_OLD = '''        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            torch.index_select(
                self.ngram_embedding.weight,
                0,
                ngram_ids.reshape(-1),
                out=output.reshape(-1, self.head_dim),
            )
            return output
        return self.ngram_embedding(ngram_ids).flatten(-2)'''

PLE_FASTPATH_NEW = '''        if output_buffer is not None:
            # Offload fast path. The GPU side expects, per head, the
            # same packed row the on-device lookup produces:
            # NVFP4 -> cat(codes[head_dim/2], scales[head_dim/16]);
            # FP8    -> the full-width fp8 row (row_width == head_dim).
            # Prefer a memory-mapped pre-packed table (built by
            # files/build_ple_packed_table_fp8.py); otherwise assemble
            # the row from the separate code/scale parameters.
            emb = self.ngram_embedding
            ids = ngram_ids.reshape(-1)
            packed = getattr(emb, "_packed_table", None)
            scales = getattr(emb, "weight_scale", None)
            if packed is not None:
                row_width = packed.shape[-1]
                total_width = ngram_ids.shape[-1] * row_width
                output = output_buffer[:num_tokens, :total_width]
                torch.index_select(
                    packed, 0, ids, out=output.reshape(-1, row_width)
                )
                return output
            if scales is not None and scales.dim() == 2:
                codes = emb.weight
                scales_u8 = scales.view(torch.uint8)
                cw, sw = codes.shape[-1], scales_u8.shape[-1]
                row_width = cw + sw
                total_width = ngram_ids.shape[-1] * row_width
                output = output_buffer[:num_tokens, :total_width]
                rows = output.reshape(-1, row_width)
                rows[:, :cw].copy_(codes.index_select(0, ids))
                rows[:, cw:].copy_(scales_u8.index_select(0, ids))
                return output
            weight = emb.weight
            row_width = weight.shape[-1]
            total_width = ngram_ids.shape[-1] * row_width
            output = output_buffer[:num_tokens, :total_width]
            torch.index_select(
                weight, 0, ids, out=output.reshape(-1, row_width)
            )
            return output
        return self.ngram_embedding(ngram_ids).flatten(-2)'''

# Port of upstream vLLM d4d703caf (#54882) onto the image's ple_layer.
PLE_LAYER_EDITS = [
    (
        "from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n",
        "from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n"
        "from vllm.model_executor.layers.quantization.modelopt import (\n"
        "    ModelOptMixedPrecisionConfig,\n"
        ")\n",
    ),
    (
        '    """Select global-scale FP8 only for quantized PLE checkpoint shards."""\n'
        "\n"
        "    if not isinstance(quant_config, Fp8Config):\n",
        '    """Select global-scale FP8 only for quantized PLE checkpoint shards."""\n'
        "\n"
        "    if isinstance(quant_config, ModelOptMixedPrecisionConfig):\n"
        "        if quant_config._resolve_quant_algo(prefix) == \"FP8\":\n"
        "            return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\n"
        "        return None\n"
        "\n"
        "    if not isinstance(quant_config, Fp8Config):\n",
    ),
    (PLE_FASTPATH_OLD, PLE_FASTPATH_NEW),
]

MODELOPT_REL = "model_executor/layers/quantization/modelopt.py"

# 1b. MTP draft MoE dispatch (see module docstring).
MODELOPT_EDITS = [
    (
        "from vllm.model_executor.layers.quantization.kv_cache import "
        "BaseKVCacheMethod\n",
        "from vllm.model_executor.layers.quantization.fp8 import (\n"
        "    Fp8Config,\n"
        "    Fp8MoEMethod,\n"
        ")\n"
        "from vllm.model_executor.layers.quantization.kv_cache import "
        "BaseKVCacheMethod\n",
    ),
    (
        "    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:\n"
        "        candidates = [prefix]\n"
        "\n"
        "        if prefix.endswith(\".lm_head\"):\n",
        "    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:\n"
        "        candidates = [prefix]\n"
        "\n"
        "        # MTP draft layers are constructed at\n"
        "        # `mtp.layers.<num_hidden_layers>` while ModelOpt checkpoints\n"
        "        # list the same tensors under the checkpoint numbering\n"
        "        # `mtp.layers.0`. Add the checkpoint-numbered variant so\n"
        "        # mixed-precision quantized_layers resolve for the draft\n"
        "        # (e.g. the FP8-block MTP experts of the official NVFP4\n"
        "        # checkpoint).\n"
        "        mtp_parts = prefix.split(\".\")\n"
        "        if (\n"
        "            len(mtp_parts) > 3\n"
        "            and mtp_parts[0] == \"mtp\"\n"
        "            and mtp_parts[1] == \"layers\"\n"
        "            and mtp_parts[2].isdigit()\n"
        "            and mtp_parts[2] != \"0\"\n"
        "        ):\n"
        "            candidates.append(\n"
        "                \"mtp.layers.0.\" + \".\".join(mtp_parts[3:])\n"
        "            )\n"
        "\n"
        "        if prefix.endswith(\".lm_head\"):\n",
    ),
    (
        "        if isinstance(layer, RoutedExperts):\n"
        "            if quant_algo == \"FP8\":\n",
        "        if isinstance(layer, RoutedExperts):\n"
        "            if quant_algo == \"FP8_BLOCK_SCALES\":\n"
        "                # ModelOpt emits FP8_BLOCK_SCALES for 128x128\n"
        "                # block-quantized FP8 experts (the official NVFP4\n"
        "                # checkpoint's MTP experts, copied byte-for-byte from\n"
        "                # the FP8 checkpoint). The standard Fp8MoEMethod\n"
        "                # handles RoutedExperts with block scales.\n"
        "                return Fp8MoEMethod(\n"
        "                    Fp8Config(\n"
        "                        is_checkpoint_fp8_serialized=True,\n"
        "                        activation_scheme=\"dynamic\",\n"
        "                        weight_block_size=[128, 128],\n"
        "                    ),\n"
        "                    layer,\n"
        "                )\n"
        "            if quant_algo == \"FP8\":\n",
    ),
]

SHORT_CONV_REL = "v1/attention/backends/short_conv_attn.py"

# 1c. Async PLE MTP metadata transfers — port of upstream vLLM edc0fb7e0
#     (#55054). Per-step CPU tensors in the PleShortConv spec-decode metadata
#     builder move to the GPU asynchronously instead of via blocking
#     .to(device) calls, so the copies overlap GPU work on the decode path.
SHORT_CONV_EDITS = [
    (
        "        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)\n"
        "        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)\n",
        "        spec_req_idx = async_tensor_h2d(\n"
        "            spec_req_idx_cpu, device=query_start_loc.device\n"
        "        )\n"
        "        non_spec_req_idx: torch.Tensor | None = None\n",
    ),
    (
        "            # [decode, prefill] tokens.\n"
        "            req_group = torch.full(\n",
        "            # [decode, prefill] tokens.\n"
        "            non_spec_req_idx = async_tensor_h2d(\n"
        "                non_spec_req_idx_cpu, device=query_start_loc.device\n"
        "            )\n"
        "            decode_req_idx = async_tensor_h2d(\n"
        "                decode_req_idx_cpu, device=query_start_loc.device\n"
        "            )\n"
        "            req_group = torch.full(\n",
    ),
    (
        "            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1\n",
        "            req_group[decode_req_idx] = 1\n",
    ),
    (
        "        num_accepted_tokens = num_accepted_tokens[\n"
        "            spec_req_idx_cpu.to(num_accepted_tokens.device)\n"
        "        ]\n",
        "        num_accepted_tokens = num_accepted_tokens[spec_req_idx]\n",
    ),
    (
        "            if non_spec_req_idx_cpu is not None:\n"
        "                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)\n"
        "                num_computed_tokens = num_computed_tokens[non_spec_req_idx]\n",
        "            assert non_spec_req_idx is not None\n"
        "            num_computed_tokens = num_computed_tokens[non_spec_req_idx]\n",
    ),
]

HC_OPS_REL = "models/qwen3_8_flash_next/nvidia/ops/hc.py"

# 1d. HC combine-norm reuse for MTP input — port of upstream fd4a15126
#     (#54687). Kernels gain an optional injection (unit-weight path) and a
#     1D grid; the draft passes its embedding as a pending combine.
HC_OPS_EDITS = [
    (
        "    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)\n"
        "    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)\n"
        "    res = tl.load(res_ptr + row * stride_res + offs, mask, other=0.0)\n"
        "\n"
        "    # Keeping HC as a broadcast dimension is faster here than four separate\n"
        "    # residual load/store sequences.\n"
        "    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)\n"
        "    out = res.to(tl.float32) + block.to(tl.float32)[None, :] * inj[:, None]\n",
        "    if inj_ptr is not None:\n"
        "        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)\n"
        "    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)\n"
        "    res = tl.load(res_ptr + row * stride_res + offs, mask, other=0.0)\n"
        "\n"
        "    # Keeping HC as a broadcast dimension is faster here than four separate\n"
        "    # residual load/store sequences.\n"
        "    if inj_ptr is not None:\n"
        "        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)\n"
        "        block = block.to(tl.float32)[None, :] * inj[:, None]\n"
        "    out = res.to(tl.float32) + block.to(tl.float32)\n",
    ),
    (
        "def _hc_combine(\n"
        "    residual: torch.Tensor,\n"
        "    block_output: torch.Tensor,\n"
        "    injection_logits: torch.Tensor,\n"
        "    hc_count: int,\n"
        ") -> torch.Tensor:\n"
        "    N, DIM = residual.shape\n"
        "    assert DIM % hc_count == 0\n"
        "    hc_dim = DIM // hc_count\n"
        "    assert block_output.shape == (N, hc_dim)\n"
        "    assert injection_logits.shape == (N, hc_count)\n"
        "    assert residual.stride(1) == 1\n"
        "    assert block_output.stride(1) == 1\n"
        "    assert injection_logits.stride(1) == 1\n",
        "def _hc_combine(\n"
        "    residual: torch.Tensor,\n"
        "    block_output: torch.Tensor,\n"
        "    injection_logits: torch.Tensor | None,\n"
        "    hc_count: int,\n"
        ") -> torch.Tensor:\n"
        "    N, DIM = residual.shape\n"
        "    assert DIM % hc_count == 0\n"
        "    hc_dim = DIM // hc_count\n"
        "    assert block_output.shape == (N, hc_dim)\n"
        "    assert residual.stride(1) == 1\n"
        "    assert block_output.stride(1) == 1\n"
        "    if injection_logits is not None:\n"
        "        assert injection_logits.shape == (N, hc_count)\n"
        "        assert injection_logits.stride(1) == 1\n"
        "\n"
        "    stride_injection = injection_logits.stride(0) if injection_logits is not None else 0\n",
    ),
    (
        "        block_output.stride(0),\n"
        "        residual.stride(0),\n"
        "        injection_logits.stride(0),\n"
        "        out.stride(0),\n"
        "        hc_dim,\n"
        "        hc_count,\n"
        "        BLOCK_SIZE,\n"
        "        launch_pdl=current_platform.is_arch_support_pdl(),\n"
        "    )\n"
        "    return out\n",
        "        block_output.stride(0),\n"
        "        residual.stride(0),\n"
        "        stride_injection,\n"
        "        out.stride(0),\n"
        "        hc_dim,\n"
        "        hc_count,\n"
        "        BLOCK_SIZE,\n"
        "        launch_pdl=current_platform.is_arch_support_pdl(),\n"
        "    )\n"
        "    return out\n",
    ),
    (
        "    row = tl.program_id(0)\n"
        "    stream = tl.program_id(1)\n"
        "    offs_hc = tl.arange(0, HC_PAD)\n"
        "    mask_hc = offs_hc < HC\n"
        "    tile_ids = tl.arange(0, NUM_TILES_PAD)\n",
        "    pid = tl.program_id(0)\n"
        "    row = pid // HC\n"
        "    stream = pid % HC\n"
        "    offs_hc = tl.arange(0, HC_PAD)\n"
        "    mask_hc = offs_hc < HC\n"
        "    tile_ids = tl.arange(0, NUM_TILES_PAD)\n",
    ),
    (
        "    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)\n"
        "    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)\n"
        "    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)\n"
        "    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)\n"
        "    inj = tl.sum(tl.where(offs_hc == stream, inj, 0.0))\n"
        "    # Round the materialized combine result before normalization. This matches\n"
        "    # the unfused combine -> RMSNorm boundary.\n"
        "    out = (res.to(tl.float32) + block.to(tl.float32) * inj).to(out_ptr.dtype.element_ty)\n"
        "    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)\n"
        "\n"
        "    out = out.to(tl.float32)\n"
        "    # Keep the two-axis reduction: flattening the padded tile is ~40% slower\n"
        "    # at decode sizes.\n"
        "    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)\n"
        "    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)\n"
        "\n"
        "    if launch_pdl:\n"
        "        tl.extra.cuda.gdc_launch_dependents()\n"
        "\n"
        "    # Loading the weight earlier helps decode but keeps the tile live across\n"
        "    # the reduction and regresses larger batches, so defer it to the norm.\n"
        "    w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)\n"
        "    y = out * rrms\n",
        "    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)\n"
        "    if inj_ptr is not None:\n"
        "        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)\n"
        "    block = tl.load(\n"
        "        block_ptr + row * stride_block + offs_inner,\n"
        "        mask_inner,\n"
        "        other=0.0,\n"
        "    )\n"
        "    if inj_ptr is not None:\n"
        "        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)\n"
        "        block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))\n"
        "    # Round the materialized combine result before normalization. This matches\n"
        "    # the unfused combine -> RMSNorm boundary.\n"
        "    out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)\n"
        "    if inj_ptr is None:\n"
        "        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)\n"
        "    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)\n"
        "\n"
        "    out = out.to(tl.float32)\n"
        "    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)\n"
        "    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)\n"
        "\n"
        "    if launch_pdl:\n"
        "        tl.extra.cuda.gdc_launch_dependents()\n"
        "\n"
        "    if inj_ptr is not None:\n"
        "        # Loading the weight earlier helps decode but keeps the tile live across\n"
        "        # the reduction and regresses larger batches, so defer it to the norm.\n"
        "        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)\n"
        "    y = out * rrms\n",
    ),
    (
        "def _hc_combine_norm(\n"
        "    residual: torch.Tensor,\n"
        "    block_output: torch.Tensor,\n"
        "    injection_logits: torch.Tensor,\n"
        "    norm_weight: torch.Tensor,\n"
        "    eps: float,\n"
        "    hc_count: int,\n"
        ") -> tuple[torch.Tensor, torch.Tensor]:\n"
        "    N, DIM = residual.shape\n"
        "    assert DIM % hc_count == 0\n"
        "    hc_dim = DIM // hc_count\n"
        "    assert block_output.shape == (N, hc_dim)\n"
        "    assert injection_logits.shape == (N, hc_count)\n"
        "    assert residual.stride(1) == 1\n"
        "    assert block_output.stride(1) == 1\n"
        "    assert injection_logits.stride(1) == 1\n"
        "    assert norm_weight.is_contiguous()\n"
        "    assert norm_weight.numel() in (hc_dim, DIM)\n"
        "\n"
        "    out = residual.new_empty(residual.shape)\n"
        "    y = residual.new_empty(residual.shape)\n"
        "    BLOCK_SIZE = 512\n"
        "    _hc_combine_norm_kernel[(N, hc_count)](\n"
        "        block_output,\n"
        "        residual,\n"
        "        injection_logits,\n"
        "        norm_weight,\n"
        "        out,\n"
        "        y,\n"
        "        block_output.stride(0),\n"
        "        residual.stride(0),\n"
        "        injection_logits.stride(0),\n"
        "        out.stride(0),\n"
        "        y.stride(0),\n",
        "def _hc_combine_norm(\n"
        "    residual: torch.Tensor,\n"
        "    block_output: torch.Tensor,\n"
        "    injection_logits: torch.Tensor | None,\n"
        "    norm_weight: torch.Tensor,\n"
        "    eps: float,\n"
        "    hc_count: int,\n"
        ") -> tuple[torch.Tensor, torch.Tensor]:\n"
        "    N, DIM = residual.shape\n"
        "    assert DIM % hc_count == 0\n"
        "    hc_dim = DIM // hc_count\n"
        "    assert block_output.shape == (N, hc_dim)\n"
        "    assert residual.stride(1) == 1\n"
        "    assert block_output.stride(1) == 1\n"
        "    if injection_logits is not None:\n"
        "        assert injection_logits.shape == (N, hc_count)\n"
        "        assert injection_logits.stride(1) == 1\n"
        "    assert norm_weight.is_contiguous()\n"
        "    assert norm_weight.numel() in (hc_dim, DIM)\n"
        "\n"
        "    stride_injection = injection_logits.stride(0) if injection_logits is not None else 0\n"
        "\n"
        "    out = residual.new_empty(residual.shape)\n"
        "    y = residual.new_empty(residual.shape)\n"
        "    BLOCK_SIZE = 512\n"
        "    _hc_combine_norm_kernel[(N * hc_count,)](\n"
        "        block_output,\n"
        "        residual,\n"
        "        injection_logits,\n"
        "        norm_weight,\n"
        "        out,\n"
        "        y,\n"
        "        block_output.stride(0),\n"
        "        residual.stride(0),\n"
        "        stride_injection,\n"
        "        out.stride(0),\n"
        "        y.stride(0),\n",
    ),
]

HYPERCONN_REL = "models/qwen3_8_flash_next/nvidia/hyperconnection.py"

HYPERCONN_EDITS = [
    (
        "    def combine_and_mix(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "        prev_block_output: torch.Tensor,\n"
        "        prev_injection: torch.Tensor,\n"
        "    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n"
        '        """Consume a pending combine, then prepare the next block input.\n'
        "\n"
        "        ``hidden_states`` is the multi-stream state from before the pending\n"
        "        block's mix. Its combine with ``block_output`` is fused with this\n"
        "        module's input RMSNorm.\n"
        '        """\n',
        "    def combine_and_mix(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "        prev_block_output: torch.Tensor,\n"
        "        prev_injection: torch.Tensor | None,\n"
        "    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n"
        '        """Consume a pending combine, then prepare the next block input.\n'
        "\n"
        "        ``hidden_states`` is the multi-stream state from before the pending\n"
        "        block's mix. Its combine with ``block_output`` is fused with this\n"
        "        module's input RMSNorm. A missing injection applies the block output\n"
        "        to every stream with unit weight.\n"
        '        """\n',
    ),
    (
        "    def combine(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "        block_output: torch.Tensor,\n"
        "        injection: torch.Tensor,\n"
        "    ) -> torch.Tensor:\n"
        "        return hc_combine(hidden_states, block_output, injection, self.hc_count)\n",
        "    def combine(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "        block_output: torch.Tensor,\n"
        "        injection: torch.Tensor | None,\n"
        "    ) -> torch.Tensor:\n"
        "        return hc_combine(hidden_states, block_output, injection, self.hc_count)\n",
    ),
]

MODEL_REL = "models/qwen3_8_flash_next/nvidia/model.py"

MODEL_EDITS = [
    (
        "        attn_hc = self.attn_hyper_connection\n"
        "        if self.ple is not None:\n"
        "            # PLE adds directly to the multi-stream state, so pending HC state\n"
        "            # must be materialized before the addition.\n"
        "            if prev_block_output is not None and prev_injection is not None:\n",
        "        if prev_block_output is None:\n"
        "            assert prev_injection is None\n"
        "        attn_hc = self.attn_hyper_connection\n"
        "        if self.ple is not None:\n"
        "            # PLE adds directly to the multi-stream state, so pending HC state\n"
        "            # must be materialized before the addition.\n"
        "            if prev_block_output is not None:\n",
    ),
    (
        "        # Fuse a pending combine with this HC module's mix when possible.\n"
        "        if prev_block_output is not None and prev_injection is not None:\n"
        "            hidden_states, block_input, injection = attn_hc.combine_and_mix(\n",
        "        # Fuse a pending combine with this HC module's mix when possible.\n"
        "        if prev_block_output is not None:\n"
        "            hidden_states, block_input, injection = attn_hc.combine_and_mix(\n",
    ),
]

MTP_REL = "models/qwen3_8_flash_next/nvidia/mtp.py"

MTP_EDITS = [
    (
        "        hc_count = self.hc_count\n"
        "        hidden_size = self.hidden_size\n"
        "\n"
        "        if get_pp_group().is_first_rank:\n",
        "        hc_count = self.hc_count\n"
        "        hidden_size = self.hidden_size\n"
        "        prev_block_output: torch.Tensor | None = None\n"
        "\n"
        "        if get_pp_group().is_first_rank:\n",
    ),
    (
        "            hidden_states = self.fc_hidden(hidden_states)\n"
        "            # Add the embedding residual to every branch, then fold back\n"
        "            # to [T, hc_count*H] (HC outer, HS inner) for the HC decoder.\n"
        "            hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states\n"
        "            hidden_states = hidden_states.flatten(-2)\n"
        "        else:\n",
        "            hidden_states = self.fc_hidden(hidden_states)\n"
        "            hidden_states = hidden_states.flatten(-2)\n"
        "            # Pass the embedding as a pending HC combine (unit weight,\n"
        "            # no injection); the first draft layer fuses it into its\n"
        "            # combine+norm instead of eagerly broadcasting it here.\n"
        "            prev_block_output = inputs_embeds\n"
        "        else:\n",
    ),
    (
        "        hidden_states, block_output, injection = layer(\n"
        "            hidden_states=hidden_states,\n"
        "            prev_block_output=None,\n"
        "            prev_injection=None,\n",
        "        hidden_states, block_output, injection = layer(\n"
        "            hidden_states=hidden_states,\n"
        "            prev_block_output=prev_block_output,\n"
        "            prev_injection=None,\n",
    ),
]


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def write(path: str, src: str) -> None:
    import ast

    ast.parse(src)  # never write unparseable python
    with open(path, "w") as f:
        f.write(src)


def apply_edits(rel_path: str, edits: list[tuple[str, str]], label: str) -> None:
    path = os.path.join(VLLM_DIR, rel_path)
    if not os.path.isfile(path):
        sys.exit(
            f"[patch-qwen38] ABORT: {path} not found. Is the image "
            "vllm/vllm-openai:qwen38-flash-next?"
        )
    src = read(path)
    applied = 0
    for i, (old, new) in enumerate(edits):
        if new in src:
            continue  # already applied
        count = src.count(old)
        if count != 1:
            sys.exit(
                f"[patch-qwen38] ABORT {label}: anchor {i} matches {count} "
                f"times (expected 1):\n{old[:180]}"
            )
        src = src.replace(old, new)
        applied += 1
    if applied:
        write(path, src)
    print(f"[patch-qwen38] {label}: {applied} edit(s) applied, rest already present")


def swap_fp8_attach(worker_path: str) -> None:
    """Replace Mia's NVFP4 _attach_packed_table with the FP8-aware version."""
    src = read(worker_path)
    if 'meta.get("dtype", "u8")' in src:
        print("[patch-qwen38] worker: FP8-aware _attach_packed_table already present")
        return
    start = src.find(MIA_ATTACH_HEAD)
    end = src.find(MIA_ATTACH_TAIL, start)
    if start < 0 or end < 0:
        sys.exit("[patch-qwen38] ABORT: Mia's _attach_packed_table block not found in worker.py")
    end += len(MIA_ATTACH_TAIL)
    src = src[:start] + FP8_ATTACH + src[end:]
    write(worker_path, src)
    print("[patch-qwen38] worker: swapped in FP8-aware _attach_packed_table")


def run_mia_generator(generator: str, staged: dict[str, str]) -> None:
    """Stage origs from the installed vLLM, run Mia's generator, copy back."""
    tmp = tempfile.mkdtemp(prefix="vllm-patch-qwen38-")
    try:
        gen_src = os.path.join(tmp, generator)
        shutil.copyfile(os.path.join(HERE, generator), gen_src)
        for rel, stage in staged.items():
            dst = os.path.join(tmp, stage)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(VLLM_DIR, rel), dst)
        scope: dict[str, object] = {"__name__": "mia_gen", "__file__": gen_src}
        with open(gen_src) as f:
            exec(compile(f.read(), gen_src, "exec"), scope)
        # Mia's generators write patched copies next to the staged origs.
        produced = {
            "patch_ple_offload.py": [
                ("ple_offload/ple_offload_layer.py", "model_executor/layers/ple_offload_layer.py"),
                ("ple_offload/connector.py", "v1/ple_offload/connector.py"),
                ("ple_offload/worker.py", "v1/ple_offload/worker.py"),
                ("ple_offload/protocol.py", "v1/ple_offload/protocol.py"),
            ],
            "patch_qsa_fp8_kv.py": [
                ("qsa_ops_patched.py", "models/qwen3_8_flash_next/nvidia/ops/qsa.py"),
                ("qsa_nvidia_patched.py", "models/qwen3_8_flash_next/nvidia/qsa.py"),
            ],
        }[generator]
        for stage, rel in produced:
            src = os.path.join(tmp, stage)
            if not os.path.isfile(src):
                sys.exit(f"[patch-qwen38] ABORT: generator did not produce {stage}")
            shutil.copyfile(src, os.path.join(VLLM_DIR, rel))
            print(f"[patch-qwen38] installed patched {rel}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    print("[patch-qwen38] vLLM install:", VLLM_DIR)

    # 1. Mixed-precision PLE dispatch (upstream #54882 port).
    apply_edits(PLE_LAYER_REL, PLE_LAYER_EDITS, "ple_layer: mixed-precision FP8 PLE dispatch")

    # 1b. MTP draft MoE dispatch (FP8_BLOCK_SCALES -> Fp8MoEMethod).
    apply_edits(MODELOPT_REL, MODELOPT_EDITS, "modelopt: mtp FP8-block MoE dispatch")

    # 1c. Async PLE MTP metadata transfers (upstream #55054 port).
    apply_edits(SHORT_CONV_REL, SHORT_CONV_EDITS, "short_conv_attn: async PLE MTP metadata H2D")

    # 1d. HC combine-norm reuse for MTP input (upstream #54687 port).
    apply_edits(HC_OPS_REL, HC_OPS_EDITS, "ops/hc: optional-injection combine kernels")
    apply_edits(HYPERCONN_REL, HYPERCONN_EDITS, "hyperconnection: optional-injection signatures")
    apply_edits(MODEL_REL, MODEL_EDITS, "model: pending-combine without injection")
    apply_edits(MTP_REL, MTP_EDITS, "mtp: embedding as pending HC combine")

    # 2. GB10 offload + mmap packed table + QSA fp8-kv, via Mia's generators.
    for gen in ("patch_ple_offload.py", "patch_qsa_fp8_kv.py"):
        staged = MIA_STAGE_FILES[gen]
        todo = [
            rel
            for rel in staged
            if read(os.path.join(VLLM_DIR, rel)).find(MIA_TARGETS[rel][0]) < 0
        ]
        if not todo:
            print(f"[patch-qwen38] {gen}: all targets already patched")
            continue
        print(f"[patch-qwen38] {gen}: patching {todo}")
        run_mia_generator(gen, staged)

    # 3. Swap Mia's NVFP4 attach for the FP8-aware version.
    swap_fp8_attach(os.path.join(VLLM_DIR, "v1/ple_offload/worker.py"))

    # Post-conditions: every target carries its marker.
    for rel, (marker, _) in MIA_TARGETS.items():
        if read(os.path.join(VLLM_DIR, rel)).find(marker) < 0:
            sys.exit(f"[patch-qwen38] ABORT: {rel} still missing marker {marker!r}")
    print("[patch-qwen38] all patches ready")


if __name__ == "__main__":
    main()
