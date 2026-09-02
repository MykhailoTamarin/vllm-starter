# sparkinfer kernel backports

Two fail-closed kernel fixes for the **sparkinfer** package (the
`exl3-trellis-fused` fork of [efeslab/SparkInfer](https://github.com/efeslab/SparkInfer)
that ships the Trellis MoE kernels the EXL3 backend uses). Both originate
from upstream [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)
PRs **#150** and **#228** and were re-anchored onto our pinned tree.

Unlike `vllm-patches/` (unified diffs applied to a source tree before the
wheel build) and `exllamav3-patches/` (git-apply-able `.patch` files on a
cloned tag), these two are **fail-closed Python anchor scripts** applied to
the **installed** sparkinfer package after `pip install`:

```bash
pip install --no-deps git+https://github.com/brandonmmusic-max/b12x.git@669a12dd
python3 sparkinfer-patches/patch_w4a16_expert_counts.py
python3 sparkinfer-patches/patch_tiny_decode_route_clamp.py
```

sparkinfer is pure Python + Triton JIT kernels, so there is no compiled
source tree in our build to patch — the scripts edit the installed files in
place and **fail (`SystemExit(1)`) if an anchor is missing or ambiguous**, so
a drifted upstream can never silently produce an unpatched image.

## Upstream baseline

| Field | Value |
|---|---|
| Repo | `https://github.com/brandonmmusic-max/b12x` (branch `exl3-trellis-fused`) |
| Commit | `669a12ddc7cf3021e91a25f398b1a883b703fd12` |
| `sparkinfer` version | `1.0.1` |
| Backport origin | `local-inference-lab/b12x` PRs #150, #228 |

> The v26 image applied the same changes as in-tree patch scripts at the top
> level of the image dir. They were moved into `sparkinfer-patches/` (parallel
> to `exllamav3-patches/`) and kept as fail-closed anchor scripts — the
> installed-package target has no git tree to apply unified diffs against.

## Patch list

| # | Script | File(s) | What / why |
|---|---|---|---|
| 1 | `patch_w4a16_expert_counts.py` | `sparkinfer/moe/fused_moe/_impl.py` | Backport of b12x #150: preallocate the W4A16 route histogram (`expert_counts`) so the fast count+prefix kernel runs during CUDA-graph capture without allocation. |
| 2 | `patch_tiny_decode_route_clamp.py` | `sparkinfer/moe/_shared/kernels/tiny_decode.py` | Backport of b12x #228: keep graphed tiny-decode routes with inactive expert ids in range (graph padding / non-local EP routes) to avoid out-of-bounds weight reads. |

## Why these specific files?

The EXL3 vLLM overlay drives the MoE hot path through sparkinfer's
`moe/fused_moe` Trellis runner:

- `_impl.py` — the fused MoE implementation. Under **full CUDA graphs**
  (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`) the route-histogram buffer must be
  preallocated, or the count+prefix kernel allocates during graph capture and
  fails.
- `_shared/kernels/tiny_decode.py` — the small-batch (≤ 8 tokens) decode
  kernel. CUDA-graph padding and non-local EP routes can hand it inactive
  expert ids; without the clamp they read out of bounds.

## Regenerating after a bump

1. Move the sparkinfer pin in the Dockerfile (e.g. a newer commit on
   `exl3-trellis-fused` or a new fork head).
2. Re-derive each fix against the new tree:
   - `patch_w4a16_expert_counts.py` — locate the route-histogram allocation
     site in `sparkinfer/moe/fused_moe/_impl.py` and confirm the preallocation
     hunk's anchor text.
   - `patch_tiny_decode_route_clamp.py` — locate the `eid = ...topk_ids[...]`
     and `if cgrp == 0:` guards in `tiny_decode.py` and confirm the clamp
     anchors.
3. Update the `Upstream baseline` table and the image README's patch table.
   If the fixes land in the fork itself, delete the scripts and the
   corresponding `sparkinfer-patches/` rows.