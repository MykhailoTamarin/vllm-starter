# Agents — vLLM Model Manager

Manages vLLM model containers on a DGX Spark. Each model is a YAML config in `models/`, controlled by `vllm-manager.sh`.

## Critical Rules

1. Never commit or push unless explicitly asked.
2. Always work on `develop` — never push to `main`.
3. Always pull `main` before starting work.
4. Always test with `DRY_RUN=true` before committing.
5. **DRY_RUN only** — never run real docker commands or modify the remote system without explicit user approval. Every manager command must use `--local` or `DRY_RUN=true` unless the user says otherwise.
6. **No remote commands** — never run `--remote` commands or SSH operations unless the user explicitly requests it.
7. **No rm/delete** — never run `rm`, `docker rm`, `docker rmi`, `rm -rf`, or any destructive removal command unless the user explicitly asks.
8. **Benchmarks sequential only** — when asked to run benchmarks, never run them in parallel. Run them one after another (wait for each to complete before starting the next). Do NOT check YAML configs — just run the benchmarks as requested and analyze results afterward.

## Git Workflow

```bash
git pull origin main && git switch develop && git pull origin develop
# ... make changes ...
git add -A && git commit -m "your message here" && git push origin develop
```

## Project Structure

```
.
├── vllm-manager.sh          # Main controller
├── llama-bench.sh           # llama-benchy wrapper (auto-saves to models/benchmarks/)
├── .env                     # Config: HF_TOKEN, SSH, DRY_RUN, MODEL
├── models/
│   ├── template.yaml        # Full template (all options documented)
│   └── *.yaml               # One per model (no comments)
├── README.md
├── FOR_AGENTS.md            # Deprecated
└── AGENTS.md
```

## Manager Commands

Model name via `--model <name>` or `.env MODEL`.

| Command                          | Description                            |
| -------------------------------- | -------------------------------------- |
| `start --model <name>`           | Stop all, then start this model        |
| `stop --model <name>`            | Stop & remove container                |
| `stop-all`                       | Stop & remove all                      |
| `restart --model <name>`         | Stop then start                        |
| `logs --model <name> [--follow]` | Last 100 lines; `--follow` local only  |
| `status`                         | docker ps for vllm containers          |
| `list`                           | All models with status                 |
| `delete --model <name>`          | Remove stopped container               |
| `update`                         | Commit, push develop, pull remote      |
| `pull`                           | Pull latest from develop (remote only) |

### Flags

| Flag             | Description                             |
| ---------------- | --------------------------------------- |
| `--remote`       | Force SSH execution                     |
| `--local`        | Force local execution                   |
| `--model <name>` | Model name (falls back to `.env MODEL`) |
| `--follow`       | Live logs (local only)                  |

### Execution Mode (default: DRY_RUN only)

- **Default**: local dry run only (`--local` or `DRY_RUN=true`) — never execute real docker or remote commands
- `--remote` → remote SSH (overrides DRY_RUN) — **only when explicitly asked by the user**
- `--local` → local dry run (no docker)
- `DRY_RUN=true --remote` → remote SSH (not a dry run)

> ⚠️ The agent MUST NOT invoke any `--remote`, `stop`, `stop-all`, `restart`, `delete`, or `update` commands without explicit user instructions. All agent work is local DRY_RUN only.

### Examples

```bash
# Local dry run
./vllm-manager.sh start --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh --local status

# Remote
./vllm-manager.sh --remote start --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh --remote stop-all

# Default model from .env
./vllm-manager.sh start
./vllm-manager.sh --model other start  # override
```

## Benchmarking

`llama-bench.sh` wraps [llama-benchy](https://github.com/eugr/llama-benchy) — auto-builds base-url from `.env SSH_HOST` + `VLLM_API_KEY`, resolves model from YAML config.

**Required:** `llama-benchy` installed (`uvx llama-benchy` or `pip install git+https://github.com/eugr/llama-benchy`).

| Command                         | Description                                                                                    |
| ------------------------------- | ---------------------------------------------------------------------------------------------- |
| `llama-bench.sh --model <name>` | Run benchmark (auto-saves to `models/benchmarks/<name>/benchmark_dd_mm_yy_HH_mm[_c{conc}].md`) |
| `+ --depth 0 4096 8192`         | Context depths to test                                                                         |
| `+ --concurrency 1 2 4`         | Parallel client counts (shows `t/s (total)` vs `t/s (req)`)                                    |
| `+ --latency-mode generation`   | Measure server-side latency (recommended)                                                      |

```bash
# Sequential benchmarks — NEVER run in parallel:
# 1. First run (single concurrency with larger depths)
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 0 4096 8192 16384 32768 65536 131072 --latency-mode generation

# 2. Second run (after first completes): concurrency test
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 0 4096 8192 16384 32768 65536 --concurrency 1 2 --latency-mode generation
```

> ⚠️ **CRITICAL: Benchmarks must NEVER run in parallel.** Always run them sequentially — wait for the first benchmark to complete before starting the second.

Results auto-save to `models/benchmarks/<yaml-name>/benchmark_<timestamp>.md` (gitignored).

### Benchmark file format

Benchmark outputs are Markdown tables saved to `models/benchmarks/<yaml-name>/benchmark_<timestamp>.md`, where `<timestamp>` is `dd_mm_yy_HH_MM` and an optional concurrency suffix `_c{conc}` is appended when concurrency > 1 is used (e.g. `_c1`, `_c1_2_4_6`).

**Filename rules:**
- No concurrency / concurrency 1 only: `benchmark_13_06_26_08_49.md`
- Concurrency flags with only `1`: `benchmark_13_06_26_08_49_c1.md`
- Multiple concurrency levels: `benchmark_13_06_26_08_50_c1_2_3_4_6.md`
- No concurrency flag at all: `benchmark_13_06_26_08_49.md`

**Single-concurrency format** (one concurrency level or no `--concurrency` flag):
```markdown
| model                        |           test |              t/s |       peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
| nvidia/Qwen3.6-35B-A3B-NVFP4 |         pp2048 | 5674.55 ± 736.27 |                |  433.22 ± 50.25 |  367.59 ± 50.25 |  433.22 ± 50.25 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 |           tg32 |   169.94 ± 50.34 | 175.57 ± 52.06 |                 |                 |                 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | pp2048 @ d4096 | 6101.59 ± 286.27 |                | 1074.92 ± 48.71 | 1009.29 ± 48.71 | 1074.92 ± 48.71 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 |   tg32 @ d4096 |   193.66 ± 82.68 | 200.04 ± 85.45 |                 |                 |                 |
```

**Multi-concurrency format** (when `--concurrency N N N` is used):
```markdown
| model                        |                      test |      t/s (total) |        t/s (req) |       peak t/s |   peak t/s (req) |         ttfr (ms) |      est_ppt (ms) |     e2e_ttft (ms) |
| nvidia/Qwen3.6-35B-A3B-NVFP4 |  pp2048 @ d4096 (c1) | 5842.11 ± 170.24 | 5842.11 ± 170.24 |                |                  |   1116.66 ± 30.40 |   1052.67 ± 30.40 |   1116.66 ± 30.40 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 |    tg32 @ d4096 (c1) |   177.62 ± 22.75 |   177.62 ± 22.75 | 183.47 ± 23.53 |   183.47 ± 23.53 |                   |                   |                   |
| nvidia/Qwen3.6-35B-A3B-NVFP4 |  pp2048 @ d4096 (c2) |  6263.17 ± 29.39 |  3238.58 ± 15.35 |                |                  |    1961.42 ± 8.93 |    1897.43 ± 8.93 |    1961.42 ± 8.93 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 |    tg32 @ d4096 (c2) |    219.77 ± 8.78 |   120.80 ± 11.93 |  226.85 ± 9.06 |   124.73 ± 12.32 |                   |                   |                   |
```

**Column key (single-concurrency):**
| Column          | Meaning                                                                                                                                                                    |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test`          | Operation: `pp2048` = prefill throughput (2048 input tokens), `tg32` = generation throughput (32 output tokens), `@ d{N}` = context depth {N} tokens of prior conversation |
| `t/s`           | Throughput in tokens per second (mean ± stddev). For pp rows: prefill throughput. For tg rows: generation throughput (per-request, same as total when concurrency=1).      |
| `peak t/s`      | Peak generation throughput during the test window                                                                                                                          |
| `ttfr (ms)`     | Time per output token (TTFR), in milliseconds. This is the per-token decode time.                                                                                          |
| `est_ppt (ms)`  | Estimated prompt processing time, in milliseconds                                                                                                                          |
| `e2e_ttft (ms)` | End-to-end time to first token, in milliseconds                                                                                                                            |

**Column key (multi-concurrency):**
| Column           | Meaning                                                                       |
| ---------------- | ----------------------------------------------------------------------------- |
| `test`           | Same as single-concurrency, with `(c{N})` suffix indicating concurrency level |
| `t/s (total)`    | Aggregate throughput across all concurrent requests                           |
| `t/s (req)`      | Per-request throughput (total / concurrency)                                  |
| `peak t/s`       | Peak aggregate throughput                                                     |
| `peak t/s (req)` | Peak per-request throughput                                                   |

**Parsing rules:**
- Prefill rows contain `pp` in the test column — e.g. `pp2048` (no depth) or `pp2048 @ d{N}` (with depth N)
- Generation rows contain `tg` in the test column — e.g. `tg32` or `tg32 @ d{N}`
- A row with no depth suffix (e.g. `pp2048`, `tg32`) is effectively depth 0
- Values are always `mean ± stddev` — use the `mean` value

**Filename suffixes:**
- No concurrency / C1 only: `benchmark_13_06_26_08_49.md` or `benchmark_13_06_26_08_49_c1.md`
- Multiple concurrency levels: `benchmark_13_06_26_08_50_c1_2_4.md` or `benchmark_13_06_26_08_50_c1_2_3_4_6.md`

### Aggregating data across multiple benchmark files

When a model has multiple benchmark files (e.g. `benchmark_*.c1.md` and `benchmark_*.c1_2_4.md`):

1. **Prefill/Gen ranges:** Read `pp`/`tg` rows from ALL files. For multi-concurrency files, use only the C1 rows (e.g. `pp2048 (c1)`). Average values at each depth across files, then take min–max of the averaged values. Format: `X–Y t/s` or `X–Yk t/s` if max ≥ 1000.

2. **Concurrency notes:** Read `tg (cN)` rows from multi-concurrency files. Per-request t/s = `t/s (total)` / N. Format as `(CN: ~X–Y req t/s)` — round to nearest integer, use `~` prefix. Include all tested concurrency levels **that operate fine** (per-req t/s ≥ 50% of C1 baseline at same depth, stddev < 30% of mean). If C4 shows severe drop (< 50% of C1 or stddev > 30%), append only C2/C3 and add `(C4: severe drop)` or omit C4 entirely.

3. **TTFT:** Use `e2e_ttft` or `est_ppt` from `pp @ d65536` in any file. If d65536 not tested, use largest depth + `(at Nk)`.

4. **Model size:** From startup log `Checkpoint size: XX.XX GiB` → `XX.XG`.

5. **KV Cache Concurrency:** From log `Maximum concurrency for N tokens per request: Xx` → `Xx`. Should be ≥ `--max-num-seqs` in YAML.


---

## Updating the README Models Table

The **Available Models** table in `README.md` has benchmark results inline. When benchmarking a model, always update the table with the new results.

### Table columns

| Column     | Source                                                                                      | Format                                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model      | YAML filename (without `.yaml`)                                                             | `qwen3.6-35b-a3b-nvfp4-mtp`                                                                                                                      |
| Params     | YAML header `# Key specs` line or `hf models card` total/active params                      | `35B / 3B`, `120B / 12B`, `27B / —`, `—`                                                                                                         |
| Model size | YAML header or `hf models card` "On-disk size"                                              | `21.6G`, `103G`, `—`                                                                                                                              |
| Max Len    | YAML `args: --max-model-len` or HF card "Context length"                                    | `32k`, `128k`, `262k`, `256k`, `—`                                                                                                               |
| Concurrency| Startup log `Maximum concurrency for N tokens per request: Xx`                              | `4.25x`, `—`                                                                                                                                     |
| Prefill    | Benchmark `t/s` from `pp` rows across all context sizes in MD                               | `4.1–6.2k t/s` (range, k suffix for thousands, `—` if untested)                                                                                  |
| Gen t/s    | Single-client `t/s` from `tg` rows across context sizes; concurrency data separate          | `116–197 t/s` (range only — append concurrency notes only if report provided, e.g. `116–197 t/s (C4: 98 @ 8k, ~351 t/s total)`), `—` if untested |
| TTFT @ 64k | Benchmark `e2e_ttft` or `est_ppt` from `pp` row at largest depth in MD, in seconds          | `16.7s` or `16.7s (at 32k)` (convert ms → s; only append `(at {N}k)` if not 64k, `—` if untested)                                                |
| Status     | Whether benchmark has been run                                                              | `✅ **Tested**` or `⬜ Untested`                                                                                                                   |

### YAML name → model name mapping

The table **Model** column always matches the YAML filename (no `.yaml` extension). All model YAML files:

| YAML file                                              | Table Model column                     |
| ------------------------------------------------------ | -------------------------------------- |
| `models/qwen3.6-35b-a3b-nvfp4-mtp.yaml`                | `qwen3.6-35b-a3b-nvfp4-mtp`            |
| `models/nemotron-3-super-120b-a12b-nvfp4-mtp.yaml`     | `nemotron-3-super-120b-a12b-nvfp4-mtp` |
| `models/minimax-m2.7-reap-nvfp4.yaml`                  | `minimax-m2.7-reap-nvfp4`              |
| `models/mistral-small-4-119b-nvfp4.yaml`               | `mistral-small-4-119b-nvfp4`           |
| `models/qwen3.6-27b-nvfp4-mtp.yaml`                    | `qwen3.6-27b-nvfp4-mtp`                |
| `models/qwopus3.5-122b-a10b-kimi-k2.6-nvfp4-mtp.yaml`  | `qwopus3.5-122b-a10b-kimi-k2.6-nvfp4-mtp` |
| `models/qwopus3.6-27b-v2-nvfp4-mtp.yaml`               | `qwopus3.6-27b-v2-nvfp4-mtp`           |
| `models/qwopus3.6-35b-a3b-nvfp4-mtp.yaml`              | `qwopus3.6-35b-a3b-nvfp4-mtp`          |
| `models/step3p7-flash-148b.yaml`                       | `step3p7-flash-148b`                   |
| `models/deepseek-v4-flash-nvfp4-mtp.yaml`              | `deepseek-v4-flash-nvfp4-mtp`          |

### Extracting benchmark data from MD files

Benchmark results are in `models/benchmarks/<yaml-name>/benchmark_*.md`.

**Steps to fill a table row from benchmark MD:**

1. **Prefill**: find rows with `pp` in the `test` column → read the `t/s` value (mean) → collect across all context sizes → format as `min–max` → if max ≥ 1000, use `k` suffix (e.g. `4.1–6.2k t/s`)
2. **Gen t/s**: find rows with `tg` in the `test` column → read the `t/s` value (mean) → collect across all context sizes → format as `min–max t/s` (always < 1000, no k suffix). **Only** append concurrency notes if a concurrency report is provided: `(C<n>: <per-req t/s> @ <depth>, ~<total> t/s total)` — otherwise just the range
3. **TTFT @ 64k**: find the `pp` row at exactly 64k (65536) → read `e2e_ttft` or `est_ppt` → convert ms → s (divide by 1000) → format as `X.Xs`. If 64k is not in the benchmark depths (test column has no `@ d65536`), use the largest available context depth instead, and append `(at {context}k)` — e.g. `16.7s (at 32k)`. If no pp context row exists, use `—`.
4. **Status**: if a benchmark `.md` file exists → `✅ **Tested**`, else → `⬜ Untested`
5. **Params**: from `hf models card` → Total params / Active params (if MoE). Format: `TotalB / ActiveB` or `TotalB / —` if no active params.
6. **Concurrency**: from startup log line `Maximum concurrency for N tokens per request: Xx` → `Xx`. Should be ≥ `--max-num-seqs` in YAML. If not present, use `—`.

**Example:** from `benchmark_13_06_26_08_49_c1.md` for `qwen3.6-35b-a3b-nvfp4-mtp`:

| test column      | t/s (mean) | context |
| ---------------- | ---------- | ------- |
| `pp2048`         | 5674.55    | d0      |
| `pp2048 @ d4096` | 6101.59    | d4096   |
| `pp2048 @ d8192` | 6282.50    | d8192   |
| `tg32`           | 169.94     | d0      |
| `tg32 @ d4096`   | 193.66     | d4096   |
| `tg32 @ d8192`   | 186.46     | d8192   |

→ Prefill: `5.7–6.3k t/s`, Gen t/s: `170–194 t/s`, TTFT @ 8192 from `pp2048 @ d8192` `e2e_ttft`: `1696.22 ms` → `1.7s`

### Adding C2+C4 concurrency notation

When concurrency tests are run (`--concurrency 2` or `--concurrency 2 4`), append C2/C4 notes to the Gen t/s column:

1. Read `tg (cN)` rows from multi-concurrency files at each depth where a C1 row exists.
2. Per-request t/s = `t/s (total)` / N. Include concurrency levels **that operate fine**: per-req t/s ≥ 50% of C1 baseline at same depth, stddev < 30% of mean.
3. If C4 shows severe drop (< 50% of C1 or stddev > 30%), mark as `C4: degraded @ {depth}` or omit C4 entirely.
4. Note: at larger depths (e.g. 32k), C2 and C4 often degrade significantly — include what's stable and add `(C2: degraded @ {depth})` for any level that fails.
5. Concurrency notes use format: `(CN: ~X total @ depth)` — report **total** t/s from `t/s (total)` column, rounded to nearest integer, use `~` prefix for estimates when stddev > 10%.
6. Example: for `qwen3.6-35b-a3b-nvfp4-mtp` where C2 at 4k has ~243 t/s total (121 per req) and C4 at 0 has ~280 t/s total: `125–263 t/s (C2: ~243 @ 4k, C4: ~280 @ 0)`.

### When adding a new model (no benchmark yet)

Fill what you know from the YAML config and HF model card:

```markdown
| minimax-m2.7-reap-nvfp4 | 172B / ~10B | 98.9G | 64k | — | 1.4–2.3k t/s | 16.8–22.8 t/s | 25.7s (at 32k) | ✅ **Tested** |
```

Leave `Prefill`, `Gen t/s`, and `TTFT @ 64k` as `—` (Concurrency and Model size also `—` if unknown).

---

## Adding a New Model

### Step 1: Create config

```bash
cp models/template.yaml models/<name>.yaml
```

### Step 2: Fill in the YAML

Required fields:
- `image:` — Docker image (e.g. `vllm/vllm-openai:v0.23.0`)
- `args:` — At minimum `--model <repo-id>`

Common fields:
- `port:` — Host port (default 8000)
- `env:` — Container env vars
- `volumes:` — Extra host mounts

**Every model must have a header block** above `image:` following this exact format:

```yaml
# ── <Short Title> ─────────────────────────────────────────────────────────
# <One-line description — architecture, model family>
# <Key specs — params, active params, size, quantization>
#
# Recommended for: <use-cases>
# ⚠️ <Warnings if any (nightly required, special tags, etc.)>
#
# Container: vllm-<model-name>
# API:       http://localhost:<port>/v1/chat/completions
# HF:        https://huggingface.co/<owner>/<repo-id>
#
# ──────────────────────────────────────────────────────────────────────────
```

All fields are required except `⚠️` — omit the warning line if none applies.

**YAML structure rules** — critical, these are NOT optional:

- `env:` must be on its **own line** with no inline value. All env vars on subsequent indented lines.
- `args:` must be on its **own line** with no inline value. All `--flags` on subsequent indented lines.
- `volumes:` must be on its **own line** with list items (`- path`) indented below.
- `image:`, `port:`, `hf_cache:` can have values inline (simple scalars).

```
# ✅ CORRECT — env: and args: on their own line, items indented below
env:
  VLLM_ATTENTION_BACKEND=FLASHINFER
  CUDA_VISIBLE_DEVICES=0

args:
  --model qwen/Qwen3-8B
  --tensor-parallel-size 1
  --port 8000

# ❌ BROKEN — inline value after env: or args: breaks YAML parsing
env: VLLM_ATTENTION_BACKEND=FLASHINFER
args: --model qwen/Qwen3-8B
```

**Exception args** — inline on a specific `args:` line, when a parameter must be present for stability or correctness and the reason is non-obvious:

```yaml
# FlashIner JIT autotuner loops infinitely on dense models; disable it
--kernel-config '{"enable_flashinfer_autotune":false}'
```

No other inline comments. Don't comment every arg — only where removal breaks the model.

**When editing an existing model**, always check its [HuggingFace model card](https://huggingface.co) for the latest config, architecture notes, and recommended generation parameters before modifying `args:`.

### Step 3: Test with DRY_RUN

```bash
# .env already has DRY_RUN=true
./vllm-manager.sh start --model <name>
```

Verify the output contains:
- ✅ Correct image tag
- ✅ Correct port mapping (`-p 8000:8000`)
- ✅ `N flags from config` where N > 0
- ✅ Env vars from YAML (`-e KEY=VALUE`)
- ✅ HF cache mount (`-v /home/.../.cache:/root/.cache`)
- ✅ Full docker run command shown

If you see `0 flags from config` or missing env vars, the YAML parser is not reading the config correctly.

### Step 4: Run for real

```bash
# Set DRY_RUN=false in .env (or remove it)
./vllm-manager.sh start --model <name>
```

---

## YAML Format Rules (critical — breakage causes `0 flags from config`)

**`env:` and `args:` must be on their own line with values indented below.**
Inline format (`env: KEY=value` or `args: --model foo`) is **WRONG** — YAML parses it as a scalar string, not a list/mapping, and the manager can't extract the values.

**Correct:**
```yaml
env:
  VLLM_ATTENTION_BACKEND=FLASHINFER
  FLASHINFER_DISABLE_VERSION_CHECK=1
  CUTE_DSL_ARCH=sm_121a

args:
  --model Qwen/Qwen3-8B
  --port 8000
  --tensor-parallel-size 1
```

**WRONG (scalar, not list):**
```yaml
env: VLLM_ATTENTION_BACKEND=FLASHINFER
  FLASHINFER_DISABLE_VERSION_CHECK=1
args: --model Qwen/Qwen3-8B
  --port 8000
```

> ⚠️ **Always verify:** after editing a YAML, run `./vllm-manager.sh --local start --model <name>` and confirm `N flags from config` where `N > 0`. If N=0, the YAML is malformed.

### Minimal config

```yaml
image: vllm/vllm-openai:v0.23.0
args:
  --model Qwen/Qwen3-8B
  --tensor-parallel-size 1
```

### Full config (from template.yaml)

```yaml
image: vllm/vllm-openai:v0.23.0
port: 8000
hf_cache: /path/to/hf/cache          # optional, default: $HOME/.cache/huggingface
volumes:                             # optional extra mounts
  - /data/models:/models

env:
  VLLM_ATTENTION_BACKEND=FLASHINFER

args:
  --model MODEL_NAME_HERE
  --port 8000
  --tensor-parallel-size 1
  --dtype auto
  --max-model-len 32768
  --max-num-seqs 16
  --max-num-batched-tokens 8192
  --enable-chunked-prefill
  --async-scheduling
  --enable-prefix-caching
  --gpu-memory-utilization 0.9
  --trust-remote-code
  --enable-auto-tool-choice
  --disable-access-log-for-endpoints /health,/metrics,/ping
```

### Key args

| Arg                        | Description         | Common values                                   |
| -------------------------- | ------------------- | ----------------------------------------------- |
| `--model`                  | HuggingFace repo ID | `Qwen/Qwen3-8B`, `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| `--tensor-parallel-size`   | GPU count           | `1`, `2`, `4`                                   |
| `--dtype`                  | Data type           | `auto`, `bfloat16`, `float16`                   |
| `--quantization`           | Quant method        | `modelopt`, `fp8`, `awq`, `gptq`                |
| `--max-model-len`          | Context window      | `32768`, `65536`, `131072`                      |
| `--gpu-memory-utilization` | VRAM fraction       | `0.85`, `0.9`, `0.95`                           |
| `--attention-backend`      | Attention engine    | `flashinfer`, `sdpa`, `flash_attn`              |
| `--moe-backend`            | MoE backend         | `marlin`, `triton`                              |
| `--kv-cache-dtype`         | KV cache type       | `fp8`, `fp16`, `auto`                           |

### Docker image

| Tag                           | Use case                                                |
| ----------------------------- | ------------------------------------------------------- |
| `:v0.23.0-aarch64-ubuntu2404` | Stable release (default for Blackwell)                  |
| `:nightly`                    | New features (NVFP4 on older stable, Qwen3.6 MTP, etc.) |

See: https://hub.docker.com/r/vllm/vllm-openai/tags

---

## Model Inspection

**Use the `hf` CLI to inspect models**

**⛔ NEVER use web_fetch for model cards or metadata (not every model has metadata). NEVER run `hf download` command** The `hf` CLI is the only correct source. If `hf` seems broken, verify it's installed (`which hf`) and install if needed — never fall back to web_fetch.

```bash
# Model card (README + YAML frontmatter) — architecture, specs, tags, license
hf models card owner/Model-Name
#→ Markdown with YAML frontmatter (--- ... ---) containing: pipeline_tag,
#  license, tags, base_model, library_name, language, etc.
#  Followed by the full model card markdown with architecture details,
#  benchmarks, usage examples, and warnings.
```

**Extract key fields from `hf models card` output:**

| YAML frontmatter field | Use for                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `pipeline_tag`         | Inference type — `text-generation`, `image-text-to-text`, etc. |
| `license`              | License (e.g. `apache-2.0`, `other`)                           |
| `tags`                 | Quick scan — `nvfp4`, `moe`, `vlm`, `mamba`, `quantized`, etc. |
| `base_model`           | Original model this variant is based on                        |
| `library_name`         | Runtime library — `transformers`, `Model Optimizer`, etc.      |

**Extract architecture/details from the model card body:**

| What to look for             | Where                                                                  |
| ---------------------------- | ---------------------------------------------------------------------- |
| Total params / active params | "Model Details" table — "Total Parameters" or "Number of Parameters"   |
| Architecture                 | "Model Details" table — "Architecture" or card title                   |
| Context length               | "Input" section — "Context length" or "Max Sequence Length"            |
| Quantization method          | "Model Details" table — "Quantization" or card body                    |
| Disk size                    | "Model Details" table — "Size on Disk"                                 |
| Hardware requirements        | "Target Hardware" or "Software Integration" section                    |
| License                      | YAML frontmatter `license` field                                       |
| Warnings / gotchas           | Card body — look for "Gotchas", "Limitations", "⚠️", "⚠"                |
| Usage examples               | "Usage" or "Running on" section — shows `vllm serve` / docker commands |

**Example — parsing a card:**

```bash
hf models card nvidia/Qwen3.6-35B-A3B-NVFP4 2>/dev/null
```

Read the YAML frontmatter (between `---` markers) for license, tags, pipeline_tag. Then read the card body for architecture specs, context length, quantization details, and any warnings. Use this to fill the model YAML header block accurately.

Models are cached under `$HOME/.cache/huggingface` (mounted into every container).

---

## Environment Variables

### Required (.env)

| Variable       | Description                                      |
| -------------- | ------------------------------------------------ |
| `HF_TOKEN`     | HuggingFace auth token                           |
| `VLLM_API_KEY` | API key (default: `vllm`)                        |
| `DRY_RUN`      | `true` to simulate, unset for real docker        |
| `MODEL`        | Default model name (used when `--model` omitted) |

### Optional (.env)

| Variable       | Description                                      |
| -------------- | ------------------------------------------------ |
| `LOKI_URL`     | Loki log forwarding URL                          |
| `SERVICE_NAME` | Loki label (default: `vllm`)                     |
| `SSH_USER`     | Remote SSH username                              |
| `SSH_HOST`     | Remote host IP/hostname                          |
| `SSH_PORT`     | SSH port (default: 22)                           |
| `SSH_KEY`      | SSH private key path                             |
| `SSH_DIR`      | Remote project directory                         |
| `VLLM_REMOTE`  | Set to `0` on remote `.env` to prevent recursion |

---

## KV Cache Concurrency Check

When checking if a model can handle N concurrent requests at full context, **do not calculate token counts or memory sizes** — vLLM already computes this.

Look for these two lines in the startup log for the model:

```
INFO [kv_cache_utils.py:XXXX] GPU KV cache size: 5,XXX,XXX tokens
INFO [kv_cache_utils.py:XXXX] Maximum concurrency for XXX,XXX tokens per request: XX.XXx
```

- The **second line** is the answer: if it says `21.38x` and you want 4 concurrent 262K requests, you're fine.
- If it says `2.5x` for 4 requests, increase `--gpu-memory-utilization` in the YAML (or reduce `--max-model-len`).

Also check — **`--max-num-seqs` in the YAML config is the hard concurrency cap on the vLLM side**, not KV cache. The KV log tells you *what's physically possible*, but `max-num-seqs` tells you *what vLLM will actually allow*. Both must accommodate the desired concurrency.

The **KV log value should be slightly higher than `--max-num-seqs`** — e.g. log says `4.25x` and YAML has `--max-num-seqs 4` is correct. If log says `2.5x` and YAML has `--max-num-seqs 4`, increase `--gpu-memory-utilization` in YAML (or reduce `--max-model-len`).

## Container Naming

Pattern: `vllm-<model-name>`

Example: `vllm-qwen3.6-35b-a3b-nvfp4`

---

## API Access

Once started, the model is available at:

```
http://localhost:<port>/v1/chat/completions
```

For remote models, use the remote host's IP:

```
http://<remote-host>:<port>/v1/chat/completions
```

---

## Update Workflow

When making YAML changes:

1. Edit the model YAML locally
2. Push changes: `./vllm-manager.sh update` (commits, pushes, pulls on remote)
3. Restart: `./vllm-manager.sh --remote restart --model <name>`
