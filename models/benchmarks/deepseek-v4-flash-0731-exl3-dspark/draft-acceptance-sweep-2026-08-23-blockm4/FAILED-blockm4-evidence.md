# A/B: `VLLM_EXL3_TRELLIS_BLOCK_M=4` — **FAILED** (invalid value, engine cannot boot)

- date: 2026-08-23 00:05–00:21 (two attempts: `--health-timeout 2400` then `--health-timeout 600`)
- model: deepseek-v4-flash-0731-exl3-dspark, image `vllm-exl3-v26:latest`, K160 draft
- intent: test the `03` L3 hypothesis that a smaller decode Trellis block (`block_m` 8→4)
  improves single-token (`m=1`) decode efficiency

## Result

`block_m=4` is **not an accepted value** — the model never becomes healthy (`/health`
stays down in both attempts; the APIServer crash-loop-retries the EngineCore forever).
The YAML override was applied via the sweep harness (`--env-extra
VLLM_EXL3_TRELLIS_BLOCK_M=4 --tag blockm4`); the original YAML was restored after each
attempt and the serving config was restored to defaults afterwards.

## Root cause (from container logs, `docker logs` + harness health-timeout dump)

```
(EngineCore pid=131) ERROR [core.py:1330]   File "<string>", line 16, in __init__
(EngineCore pid=131) ERROR [core.py:1330]   File "sparkinfer/moe/trellis_moe/_impl.py", line 146, in __post_init__
(EngineCore pid=131) ERROR [core.py:1330]     raise ValueError(
(EngineCore pid=131) ERROR [core.py:1330] ValueError: block_size_m must be one of (8, 16, 32, 48, 64), got 4
```

SparkInfer's `trellis_moe` planner (`_impl.py:146`, `__post_init__` of the plan object)
only accepts `block_size_m ∈ {8, 16, 32, 48, 64}`.

## Conclusion for the lever table

- **The "reduce decode block to 4/2" sub-lever is closed, empirically**: the current
  served value (`block_m=8`) is the **minimum legal value**. The only legal alternatives
  (16/32/48/64) are strictly *larger* blocks — worse for the `m=1` decode shape the
  spec-decode path actually runs. There is no smaller legal decode block to try.
- The `m=1`-vs-`block_m=8` sublinearity identified in `02`/`03` L3 remains real, but it
  is **not addressable via `VLLM_EXL3_TRELLIS_BLOCK_M`** on this SparkInfer build. Only a
  kernel profile (`nsys`/`ncu`) can quantify it; the config knob cannot.
- Capture-size alignment (`--cudagraph-capture-sizes 1 2 4 6` → `1 2 3 5 6`) was tested
  as the complementary L3 sub-lever and measured **neutral** (see
  `../draft-acceptance-sweep-2026-08-23-capsiz/k160.md`).

## Files

- this note: `FAILED-blockm4-evidence.md` (dir `draft-acceptance-sweep-2026-08-23-blockm4/`)
- harness log (remote, not committed): `/tmp/blockm4_retry.log`, `/tmp/blockm4_retry_wrap.log`