# DSPark draft acceptance sweep — benchmark tool

`scripts/sweep_dspark_acceptance.py` measures how the **EXL3 DeepSeek-V4-Flash
stack** (REAP-K216 target + compact DSPark speculative draft) behaves across
draft sizes K ∈ {64, 96, 128, 160, 192} for three task types
(coding / chat / text-writing), 3 repeats each.

## What it measures per request (all via v0.26.0 Prometheus counters + SSE timing)

| Metric | Source |
|---|---|
| **Acceptance rate** = accepted / draft tokens | `vllm:spec_decode_num_accepted_tokens_total` / `vllm:spec_decode_num_draft_tokens_total` counter diff |
| **Acceptance by draft position** (pos 0..4) | `vllm:spec_decode_num_accepted_tokens_per_pos_total{position=...}` diff |
| **KV tokens size** | `usage.prompt_tokens + usage.completion_tokens` (per-request sequence KV footprint) |
| **KV cache usage %** | `vllm:kv_cache_usage_perc` gauge (after request) |
| **Generation t/s** | SSE timing: completion_tokens / (last chunk − first token) |
| **TTFT** | SSE first-token latency |

Every request gets a **unique salt** in the prompt, so the prefix cache can
never be reused between repeats — each test runs **cold, ignoring the KV
prefix cache**.

## How it runs

On the DGX Spark host (has docker + the container):

```bash
cd ~/vllm-starters
python3 scripts/sweep_dspark_acceptance.py \
    --model deepseek-v4-flash-0731-exl3-dspark \
    --ks "64 96 128 160 192" --repeats 3 --max-tokens 512
```

For each K it:
1. patches `models/<model>.yaml` (`DSPARK_DRAFT_EXPERTS`, `DRAFT_DIR`, and the
   speculative-config model path) — **original YAML restored at the end**,
2. `VLLM_REMOTE=0 DRY_RUN=false ./vllm-manager.sh start --model ...`
   (stops the current model, starts the new one; the compact draft for K is
   built at startup by `serve_ds4_exl3.sh`, then 99 GiB loads via
   instanttensor) and waits for `GET /health` (up to 40 min),
3. runs tasks × repeats, measuring each request.

## Output

```
models/benchmarks/deepseek-v4-flash-0731-exl3-dspark/draft-acceptance-sweep-<date>/
├── k64.md          # per-K report: per-task averages, per-position acceptance, raw repeats
├── k96.md
├── k128.md
├── k160.md
├── k192.md
└── draft-acceptance-summary.md   # combined: acceptance % / gen t/s / KV tokens / KV% × K
```

Per-K files are the source of truth; the summary is a K×task grid.

## Notes / limitations

- Runs **sequentially** (AGENTS rule: benchmarks never parallel). Full sweep ≈
  1–2 h depending on draft build time per K.
- Acceptance counters are global to the server; the harness diffs them around
  each request, so per-request attribution is exact as long as no other client
  talks to the server during the sweep.
- `max_tokens=512` keeps requests comparable; generation t/s includes
  reasoning tokens (the model serves with the DeepSeek-V4 reasoning preset).
- Startup time includes a from-scratch draft build inside the fresh container
  (draft dirs are not volume-mounted), so per-K restarts are not free.