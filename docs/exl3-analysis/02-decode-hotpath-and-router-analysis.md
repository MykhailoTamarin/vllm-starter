# 02 — Decode Hot-Path & Router Analysis

Sources: `overlay/.../exl3.py` (2234 lines), `models/deepseek-v4-flash-0731-exl3-dspark.yaml`, the v0.26.0 DeepSeek-V4 sources in `~/projects/vllm`, and the model's `config.json` (read from HF, no weights).

## 1. Router engagement — fused `dsv4_topk` IS running (40/43 layers)

`fused_topk_bias()` in v0.26.0 dispatches `sqrtsoftplus` to the fused Triton `dsv4_topk` **iff `can_use_dsv4_topk()` is True**:

```python
can_use_dsv4_topk = (is_cuda
    and gating_output.dtype == torch.float32      # ✓ GateLinear(out_dtype=float32)
    and gating_output.ndim == 2
    and gating_output.shape[1] in (216,256,384)   # ✓ after patch_dsv4_topk_k216
    and gating_output.is_contiguous()
    and correction_bias is not None               # ✓ iff not a hash layer
    and correction_bias.dtype == torch.float32
    and correction_bias.shape == (216,)
    and topk == 6                                # ✓ num_experts_per_tok=6
    and renormalize                              # ✓ norm_topk_prob=True
    and indices_dtype in (int32,uint32,int64))
```

Model `config.json` (2026-08-22):
- `n_routed_experts=216`, `num_experts_per_tok=6`, `topk_method=noaux_tc`, `num_hash_layers=3`,
  `norm_topk_prob=True`, `scoring_func=sqrtsoftplus`, `n_shared_experts=1`, `swiglu_limit=10.0`.

`topk_method=noaux_tc` ⇒ `gate.e_score_correction_bias = nn.Parameter(shape (216,), fp32)` is created (model.py:579–582). So:

- **Layers 3–42 (40 of 43): fused `dsv4_topk` kernel — fast path, ~1 kernel/layer.** ✅
- **Layers 0–2 (3 hash layers, `num_hash_layers=3`):** `is_hash_moe ⇒ gate.e_score_correction_bias = None` and routing uses `input_tokens`+`tid2eid` (hash table). `can_use_dsv4_topk` returns False ⇒ falls to the `patch_router_k216` pure-Torch `_topk_softplus_sqrt_torch` (≈5–7 torch launches/layer). **This is inherent** — `dsv4_topk` cannot do hash routing. Cost: ~7% of layers on the slow path. Minor, but a place to micro-optimize if profiling shows it (see `03`).

**Conclusion:** the repo is NOT losing the router race. `patch_dsv4_topk_k216` is doing its job; the pure-Torch fallback is a safety net + the 3-layer hash costjon, not the active decode path.

## 2. Decode hot path — EXL3 Trellis MoE, twice per output token

Per output token the engine runs, through the same `Exl3MoEMethod._apply_rank_sliced`:

1. **Target verify** — 43 layers × 216 experts, EXL3 3.0 bpw, Trellis MoE at `m = 1..6`.
2. **Draft forward** — 43 layers × **K192** experts, same EXL3-Trellis runtime (own scope), run as the speculative draft for the next block(s).

Per-layer flow (decode, `m ∈ [min_trellis_m, max_trellis_m]` = [1,32]):
```python
runtime = _rank_sliced_runtime(layer, x, topk_ids)   # cache-hit after warmup
binding = api.bind(trellis_plan, scratch=trellis_scratch, a=x,
                   weights=layer.exl3_trellis_weights,
                   topk_weights=topk_weights, topk_ids=topk_ids)
output  = api.run(binding=binding).to(x.dtype)
```
This is CUDA-graphed (FULL graph, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`), so per-layer host overhead (contiguous/casts/reshape) is amortized; the GPU-bound kernel efficiency dominates.

### Why decode is where the loss hides

- **Draft ≈ target.** At `DSPARK_DRAFT_EXPERTS=192`, the draft MoE is 192/216 ≈ **89% of the target's weight footprint**. The compact draft's original purpose (freeing unified memory via K64) is effectively gone at K192. Every speculative step therefore issues weight traffic ≈ 2 × (13B-active-equivalent) — the decode is bandwidth-bound **twice**.
- **`block_m=8` at `m=1`.** The decode Trellis plan uses `VLLM_EXL3_TRELLIS_BLOCK_M=8` (default). A single-token decode (`m=1`, capture sizes `1 2 4 6`) runs through a kernel planned for 8-row blocks with tile `(64,256,64,256)` (hidden 4096, intermediate 2048 → divisible by 256). Triton masks rows beyond `m`, but the expert tile fetch/scratch layout is sized for the planned capacity, so single-token efficiency is sublinear vs the theoretical `13B × bpw` memory-read bound. This is the ceiling the repo's own 24–33 t/s reflects — not the HBM-read bound of 13B active.
- **SparkInfer `bind`+`run` per layer.** Even graph-captured, each of the 43×2 layers goes through a bind→run pair; if the SparkInfer implementation re-derives per-step routing descriptors, that is per-step host+jit overhead. (Requires `nsys` to confirm; see `03`.)

### Parity path is dead at runtime
With `VLLM_EXL3_TRELLIS_MIN_M=1`, the eager `exl3_moe` argsort/parity path (`_apply_rank_sliced` tail) is unreachable (`min_trellis_m=1` ⇒ no `m < 1`). It remains as dead safety code (see `05`).

## 3. Prefill path

- `m > max_trellis_m(32)` and ≤ `max_batched_tokens(2048)` → **prefill Trellis plan** (`VLLM_EXL3_PREFILL_TRELLIS=1`, `PREFILL_BLOCK_M=64`, `chunk=128`), arena ~1 GiB.
- Prefill measured ~1290–1320 t/s short-context; degrades to ~1086 t/s at 253k ctx (long-context FFMLA/IO). TTFT is already strong vs the NVFP4 baseline (skill: 62s @64k vs 105s).
- Knobs unevaluated here (see `03` L6).

## 4. Attention path — already healthy

The 3 upstream backports already present:
- `#49486` short-context topk skip (attention.py) — TTFT.
- `#50298` combined-indices workspace (`flashmla.py`, `cache_utils.py`) — removes per-launch `torch.full`.
- `#50365` index-remap atomic drop (`sparse_utils.py`) — removes atomic contention.

These are exactly the attention-path wins the skill flagged; the stack keeps **native DeepSeek-V4 sparse-MLA attention** (FlashMLA, sparse_swa, indexer, compressor), so attention hotfixes transfer. Good.

## 5. Hash-layer detail (for completeness)

`num_hash_layers=3` + `num_nextn_predict_layers=1`; DSPark draft built from `mtp.{0,1,2}` (3 stages). The hash layers route with `input_ids` through `tid2eid` (int32). The pure-Torch fallback correctly handles this. If per-token routing cost on those 3 layers ever shows up in a profile, it cannot use `dsv4_topk` (no hash mode) — the only lever is shaving the torch path's allocations.

## 6. Config knobs in play (from the model YAML)

`gpu-memory-utilization 0.9`, `max-model-len 262144`, `max-num-seqs 2`, `max-num-batched-tokens 2048`, block-size 256, KV fp8, `cudagraph-capture-sizes 1 2 4 6`, DSPark `num_speculative_tokens=5`.
