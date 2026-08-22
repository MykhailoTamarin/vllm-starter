# DSPark acceptance sweep — combined summary

- date: 2026-08-22 22:44
- draft sizes: 64, 96, 128, 160, 192

## Acceptance % by task × K

| K | coding | chat | text |
|---|---|---|---|
| k64 | 35.6 | 29.2 | 35.7 |
| k96 | 39.1 | 29.9 | 34.9 |
| k128 | 40.6 | 30.2 | 38.5 |
| k160 | 45.3 | 35.0 | 39.7 |
| k192 | 43.9 | 39.5 | 38.2 |

## Generation t/s by task × K

| K | coding | chat | text |
|---|---|---|---|
| k64 | 33.3 | 28.4 | 32.6 |
| k96 | 35.1 | 28.7 | 32.2 |
| k128 | 36.2 | 29.0 | 34.2 |
| k160 | 39.3 | 31.9 | 35.0 |
| k192 | 38.2 | 34.5 | 34.2 |

## KV tokens per request (prompt+completion) by task × K

| K | coding | chat | text |
|---|---|---|---|
| k64 | 592 | 574 | 578 |
| k96 | 592 | 574 | 577 |
| k128 | 593 | 574 | 579 |
| k160 | 592 | 575 | 579 |
| k192 | 593 | 574 | 578 |

## KV cache usage % (after request) by task × K

| K | coding | chat | text |
|---|---|---|---|
| k64 | 0.00 | 0.00 | 0.00 |
| k96 | 0.00 | 0.00 | 0.00 |
| k128 | 0.00 | 0.00 | 0.00 |
| k160 | 0.00 | 0.00 | 0.00 |
| k192 | 0.00 | 0.00 | 0.00 |

> Note: `vllm:kv_cache_usage_perc` is a windowed scheduler gauge that decays to
> ~0 between requests (the image runs with `kv_cache_metrics=False`), so the
> post-request sample is not informative. The authoritative per-request KV size
> is **KV tokens (prompt+completion)** above (~574-593 for these prompts).
> To get live peak usage, sample the gauge *during* generation.

## Setup & methodology

- Image `vllm/vllm-openai:v0.26.0` + EXL3 overlay, REAP-K216 target
  (0xSero/deepseek-v4-flash-0731-spark), DSPark compact draft
  `num_speculative_tokens=5`, `draft_sample_method=probabilistic`.
- Tasks: coding (Python module + tests), chat (casual conversation),
  text (5-paragraph article); `max_tokens=512`, temperature 0.6, top_p 0.9.
- 3 repeats per task, every request salted → no prefix-cache reuse (cold KV).
- 2×64-token JIT warmup per K before measuring; acceptance = counter diff of
  `vllm:spec_decode_num_accepted_tokens_total /
  vllm:spec_decode_num_draft_tokens_total` per request.
- Generation t/s includes reasoning tokens (DeepSeek-V4 thinking preset).
