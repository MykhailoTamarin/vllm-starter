#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Derived from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark
# files/build_ple_packed_table.py (AGPL-3.0-or-later), adapted for the
# OFFICIAL nvidia/Qwen3.8-Flash-Next-NVFP4 checkpoint.
"""Build a packed PLE table file for memory-mapped CPU offload.

Two checkpoint layouts are supported (auto-detected from the index):

- FP8 per-tensor (OFFICIAL nvidia/Qwen3.8-Flash-Next-NVFP4): each of the 128
  row shards is a full-width float8_e4m3 tensor ([rows, head_dim], 160 B/row)
  and the scale is one global BF16 scalar that lives on the GPU. Output:
  one flat [total_rows, row_width] uint8 file (fp8 bytes),
  meta dtype "fp8_e4m3".
- NVFP4 packed (Mia-AiLab/Qwen3.8-Flash-Next-NVFP4): shards carry 4-bit
  codes ([rows, head_dim/2] uint8) plus FP8 block scales
  ([rows, head_dim/16]); each output row is cat(codes, scales.view(u8)),
  90 B/row, meta dtype "u8".

vLLM's PLE lookup returns, per row, exactly this packed row, so the offload
worker can index_select straight out of the memory map. Streams shard-by-
shard with numpy memmaps; peak RAM is well under 1 GiB.

Usage: build_ple_packed_table_fp8.py [snapshot_dir] [out_dir]

Defaults: snapshot dir auto-detected under $HF_HOME (or ~/.cache/huggingface)
for $PLE_MODEL_ID (default nvidia/Qwen3.8-Flash-Next-NVFP4); out dir defaults
to $VLLM_PLE_PACKED_TABLE_DIR (the same var the patched offload worker reads).
"""
import glob
import json
import os
import struct
import sys
import time

import numpy as np

MODEL_ID = os.environ.get("PLE_MODEL_ID", "nvidia/Qwen3.8-Flash-Next-NVFP4")
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def default_snapshot() -> str:
    org, name = MODEL_ID.split("/")
    pattern = os.path.join(
        HF_HOME, "hub", f"models--{org}--{name}", "snapshots", "*", ""
    )
    snaps = sorted(glob.glob(pattern))
    if not snaps:
        sys.exit(
            f"[build-ple] no snapshot for {MODEL_ID} under {HF_HOME}. Download the "
            "checkpoint first (huggingface-cli download " + MODEL_ID + ")"
        )
    snap = snaps[0]
    if not os.path.isfile(os.path.join(snap, "model.safetensors.index.json")):
        sys.exit(f"[build-ple] snapshot incomplete (no index): {snap}")
    return snap


def default_out_dir() -> str:
    out = os.environ.get("VLLM_PLE_PACKED_TABLE_DIR")
    if not out:
        sys.exit(
            "[build-ple] set VLLM_PLE_PACKED_TABLE_DIR (or pass out_dir). The "
            "patched offload worker reads the same variable."
        )
    return out


def main() -> None:
    snap = sys.argv[1] if len(sys.argv) > 1 else default_snapshot()
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_out_dir()
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))[
        "weight_map"
    ]

    prefixes = set()
    for k in idx:
        if ".ngram_embedding.shard_0.weight" in k:
            prefixes.add(k[: k.index(".shard_0.weight")])
    if not prefixes:
        sys.exit("[build-ple] no PLE ngram_embedding shards in index")

    headers: dict[str, tuple[dict, int]] = {}

    def header(fname: str) -> tuple[dict, int]:
        if fname not in headers:
            with open(os.path.join(snap, fname), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                headers[fname] = (json.loads(f.read(n)), 8 + n)
        return headers[fname]

    def view(name: str):
        fname = idx[name]
        h, base = header(fname)
        meta = h[name]
        start, end = meta["data_offsets"]
        mm = np.memmap(
            os.path.join(snap, fname),
            dtype=np.uint8,
            mode="r",
            offset=base + start,
            shape=(end - start,),
        )
        return mm.reshape(meta["shape"]), meta["dtype"]

    os.makedirs(out_dir, exist_ok=True)
    for prefix in sorted(prefixes):
        # vLLM name: strip leading "model." and map language_model -> language_model.model
        vname = prefix
        if vname.startswith("model.language_model."):
            vname = "language_model.model." + vname[len("model.language_model."):]
        shards = sorted(
            {
                int(k[len(prefix) + len(".shard_"):].split(".")[0])
                for k in idx
                if k.startswith(prefix + ".shard_")
            }
        )
        assert shards == list(range(len(shards))), shards
        w0, dt = view(f"{prefix}.shard_0.weight")

        if f"{prefix}.shard_0.weight_scale" in idx:
            # NVFP4 packed layout: per-shard codes + block scales.
            assert dt == "U8", dt
            s0, dt_s = view(f"{prefix}.shard_0.weight_scale")
            assert dt_s == "F8_E4M3", dt_s
            rows, cw = w0.shape
            sw = s0.shape[1]
            width = cw + sw
            dtype_name = "u8"
        else:
            # FP8 per-tensor layout: full-width fp8 rows, one global scale
            # (kept on the GPU; applied in _dequantize_embeddings).
            assert dt == "F8_E4M3", dt
            rows, cw = w0.shape
            sw = 0
            width = cw
            dtype_name = "fp8_e4m3"

        out_name = os.path.join(out_dir, vname + ".packed_u8")
        meta = {
            "rows_per_shard": rows,
            "num_shards": len(shards),
            "row_width": width,
            "codes_width": cw,
            "scales_width": sw,
            "dtype": dtype_name,
            "total_rows": rows * len(shards),
            "snapshot": os.path.basename(os.path.normpath(snap)),
        }
        if os.path.exists(out_name) and os.path.getsize(out_name) == rows * len(shards) * width:
            print("[build-ple] exists:", out_name)
            continue
        print(
            f"[build-ple] building {out_name}: {len(shards)} shards x {rows} rows "
            f"x {width} B ({dtype_name}) = "
            f"{rows * len(shards) * width / 2**30:.2f} GiB",
            flush=True,
        )
        t0 = time.time()
        tmp = out_name + ".tmp"
        CH = 1 << 19
        with open(tmp, "wb") as out:
            for i in shards:
                w, _ = view(f"{prefix}.shard_{i}.weight")
                s = view(f"{prefix}.shard_{i}.weight_scale")[0] if dtype_name == "u8" else None
                assert w.shape == (rows, cw), (i, w.shape)
                if s is not None:
                    assert s.shape == (rows, sw), (i, s.shape)
                for c in range(0, rows, CH):
                    block = w[c:c + CH]
                    if s is not None:
                        block = np.concatenate([block, s[c:c + CH]], axis=1)
                    out.write(np.ascontiguousarray(block).tobytes())
                if i % 8 == 0:
                    print(f"  shard {i}/{len(shards)} {time.time() - t0:.0f}s", flush=True)
        assert os.path.getsize(tmp) == rows * len(shards) * width
        os.rename(tmp, out_name)
        json.dump(meta, open(out_name + ".json", "w"), indent=1)
        print(f"[build-ple] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
