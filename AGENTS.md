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
8. **Benchmarks sequential only** — never run benchmarks in parallel. Wait for each to complete before starting the next.

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
├── llama-bench.sh           # Benchmark wrapper (auto-saves to models/benchmarks/)
├── tools/
│   └── llama-benchy/        # Forked benchy (wait-idle via /metrics, multi-format reports)
├── scripts/
│   └── wait-for-idle.sh     # Wait for vLLM to finish all queued requests
├── .env                     # Config: HF_TOKEN, SSH, DRY_RUN, MODEL
    ├── models/
    │   ├── template.yaml        # Full template (all options documented)
    │   ├── *.yaml               # One per model (no comments)
    │   ├── logs/                 # Startup logs (*.log) — one per model that has been run
    │   ├── benchmarks/           # Benchmark results (see Benchmarking section)
    │   └── files/                # Shared files (chat templates, etc.)
├── README.md                # Benchmark table (update after benchmarking)
└── AGENTS.md
```

## Manager Commands

| Command                          | Description                                   |
| -------------------------------- | --------------------------------------------- |
| `start --model <name>`           | Stop all, then start this model (supports `--remote`) |
| `stop --model <name>`            | Stop & remove container (supports `--remote`) |
| `stop-all`                       | Stop & remove all (supports `--remote`) |
| `restart --model <name>`         | Stop then start (supports `--remote`) |
| `logs --model <name> [--follow]` | Last 100 lines; `--follow` local only (supports `--remote`) |
| `status`                         | docker ps for vllm containers (supports `--remote`) |
| `list`                           | All models with status (supports `--remote`) |
| `update`                         | Commit, push develop, pull remote (supports `--remote`) |
| `pull`                           | Pull latest from develop (remote only) |

### Flags

| Flag             | Description                             |
| ---------------- | --------------------------------------- |
| `--remote`       | Force SSH execution                     |
| `--local`        | Force local execution                   |
| `--model <name>` | Model name (falls back to `.env MODEL`) |
| `--follow`       | Live logs (local only)                  |

### Execution Mode

- **Default** (`DRY_RUN=true`): local dry run only — no docker commands, no SSH
- `--remote`: forces SSH (overrides DRY_RUN)
- `--local`: forces local dry run (even without DRY_RUN)

> ⚠️ The agent MUST NOT invoke `--remote`, `stop`, `stop-all`, `restart`, or `update` without explicit user instructions. All agent work is local DRY_RUN only.

### Examples

```bash
# Local dry run (default)
./vllm-manager.sh start --model qwen3.6-27b-nvfp4-mtp
./vllm-manager.sh --local status

# Remote
./vllm-manager.sh --remote start --model qwen3.6-27b-nvfp4-mtp
./vllm-manager.sh --remote stop-all
./vllm-manager.sh --remote logs --model qwopus3.6-35b-a3b-nvfp4-mtp
./vllm-manager.sh --remote list

# Default model from .env MODEL
./vllm-manager.sh start          # uses .env MODEL value
./vllm-manager.sh --model other start   # override
```

## Benchmarking

`llama-bench.sh` wraps our [forked llama-benchy](https://github.com/eugr/llama-benchy) inside `tools/llama-benchy/`. Adds vLLM idle-check via `/metrics` to prevent test overlap. Auto-builds base-url from `.env SSH_HOST` + `VLLM_API_KEY`, resolves model from YAML config.

**Required:** `llama-benchy` installed (`uvx llama-benchy`).

| Command                           | Description                                |
| --------------------------------- | ------------------------------------------ |
| `llama-bench.sh --model <name>`   | Standard mode (single benchy call)         |
| `llama-bench.sh --model <name> --idle-wait` | Sequential wait-idle mode (vLLM metrics check between tests) |
| `+ --depth <d1> <d2> ...`         | Context depths to test                     |
| `+ --concurrency <c1> <c2> ...`   | Parallel client counts                     |
| `+ --latency-mode generation`     | Measure server-side latency (recommended)  |
| `+ --repeat N`                    | Run the full suite N times (wait-idle mode only) |
| `+ --format <f1>,<f2>...`         | Output formats — `json,md,png` (default: `json,md,png`) |

### Key Behavior: Wait-Idle & Report Generation

Each wait-idle benchmark run:
- **JSON**: `benchmark_<dd_mm_yy_HH_mm>_c<concurrencies>_d<depths>.json` — raw benchmark data (gitignored)
- **MD**: `benchmark_<dd_mm_yy_HH_mm>_c<concurrencies>_d<depths>.md` — parsed summary (tracked, **source of truth**)
- **PNG**: `benchmark_<dd_mm_yy_HH_mm>_c<concurrencies>_d<depths>.png` — visualization graph (gitignored, **NEVER analyze**)

Concurrencies and depths use min-max ranges (e.g., `_c1_d0_256`, `_c1-4_d256-16384`).

**RULES:**
- **ALWAYS use MD files** (e.g. `benchmark_29_06_26_08_37_c1-4_d256-16384.md`) for analysis
- **NEVER analyze PNG graphs** — they are visual artifacts only
- **JSON files are gitignored** — use only when raw data inspection is required
- **For concurrency 1 analysis, use only `_c1_dxxx` files** — never mix in results from multi-concurrency runs (`_c1-4_dxxx`)

### Running Benchmarks

> **Recommended:** Always use `--idle-wait`. The vLLM `/metrics` check between each {C×D} test prevents concurrency overlap that skews results.

#### Benchmark output structure

Each wait-idle benchmark run creates files with the same base name but different extensions:

```
models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_d0_256.json  # Raw data (gitignored)
models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_d0_256.md    # Source of truth (tracked)
models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_d0_256.png   # Visualization graph (gitignored, NEVER analyze)
```

#### Single concurrency, full depth (default workflow)

```bash
# C=1 only, full context: 0, 4k, 8k, 16k, 32k, 64k, 128k — 3 reps each
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --idle-wait --depth 0 4096 8192 16384 32768 65536 131072 --repeat 3
```

Output: `benchmark_<timestamp>_c1_{d0,4096,...}{json,md,png}`

#### Multi-concurrency with idle gates (caps at 16k depth)

```bash
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --idle-wait --depth 1024 2048 4096 8192 16384 --concurrency 1 2 4 --repeat 3
```

Flow:
```
Suite 1: vLLM idle → C=1 d=1024 → vLLM idle → C=2 d=1024 → vLLM idle → C=4 d=1024 → ...
Suite 2: vLLM idle → C=1 d=1024 → vLLM idle → C=2 d=1024 → vLLM idle → C=4 d=1024 → ...
Suite 3: vLLM idle → C=1 d=1024 → vLLM idle → C=2 d=1024 → vLLM idle → C=4 d=1024 → ...
```

Output: `benchmark_<timestamp>_c1-4_d1024_16384_{json,md,png}`

#### Legacy Mode (original behavior)

Single benchy call, no vLLM idle check between tests, no PNG output. For quick single-pass checks only.

```bash
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 0 4096 8192 --latency-mode generation
```

Output: `benchmark_<timestamp>[_c{C}_...]{json,md}`

### Agent Notes — Using Benchmark Results

**RULES:**
- **ALWAYS use MD files** for analysis (e.g. `benchmark_29_06_26_08_37_c1-4_d256-16384.md`)
- **NEVER analyze PNG graphs** — they are visual artifacts only
- **JSON files are gitignored** — use only for raw data inspection when required
- **For concurrency 1 analysis, use only `_c1_dxxx` files** — never mix in results from multi-concurrency runs (`_c1-4_dxxx`)
- **Concurrency rule:** When analyzing C1 results, use ONLY C1-only MD files (e.g., `benchmark_..._c1_d0_256.md`). Never mix C1-only benchmarks with multi-concurrency benchmarks (e.g., `benchmark_..._c1-4_d0_256.md`). Each concurrency suite is independent.

#### Legend (PNG graphs)
- Prefill: circle marker + dashed line
- Generation: square marker + solid line

```markdown
| model                        |               test |    t/s (total) |      t/s (req) |      peak t/s |   peak t/s (req) |     ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
```

Key patterns in the `test` column:

- `pp<tokens>` — prefill throughput (e.g. `pp2048`)
- `tg<tokens>` — generation throughput (e.g. `tg32` or `tg32 (cN)` for multi-concurrency)
- `pp<tokens> @ d<depth>` — prefill at context depth

Values are always formatted as `mean ± stddev` — use the `mean` value.

### Where to find results

Always start with the **Parsed MD** (source of truth for summary stats):

- **Parsed MD** (tracked by git):
  ```
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_d0_1024.md
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1-4_d1024_2048.md
  ```

- **Raw JSONs** (gitignored, use only for deep inspection):
  ```
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_d0_1024.json
  ```

### Updating the README Table

When benchmarking a model, update the **Available Models** table in `README.md`.

| Table Column | Source | Format |
| --- | --- | --- |
| Model | YAML filename (no `.yaml`) | e.g. `qwen3.6-27b-nvfp4-mtp` |
| Params | YAML header or `hf models card` | `35B / 3B`, `27B / —`, `120B / 12B` |
| Model size | YAML header or `hf models card` | `21.9G`, `—` |
| Max Len | YAML `--max-model-len` or HF card | `64k`, `262k`, `—` |
| Max Concurrency | Startup log `Maximum concurrency for N tokens per request: Xx` | `4.25x`, `13.65x`, `—` |
| Prefill | `pp` rows from ALL benchmark files → range of means | `1.0–2.7k t/s` (use `k` suffix if ≥ 1000) |
| Gen t/s | `tg` rows at C1 from ALL benchmark files → range of means | `23–30 t/s` |
| TTFT @ 64k | `e2e_ttft` from `pp` row at `d65536` (from full-depth single-concurrency test) → ms to s | `47.0s` or `17.6s (at 32k)` if no 64k depth |

**Concurrency column:** Only append `(...)` if concurrency tests were run (multi-concurrency `_c1-...` benchmark file exists). Otherwise omit the column entirely (just `Gen t/s` value, no `(...)`). Use `t/s (total)` column from `tg` rows at each concurrency level — **NOT** `t/s (req)`. Group observations by concurrency level separated by semicolons: `(C2: ~X @ d0, ~Y @ d4k, ~Z @ d8k; C4: ~A @ d0, ~B @ d4k, ~C @ d8k)`. List all measured depth points (d0 included). Use d0/d4k/d8k/d16k naming (no leading zeros in depth numbers). Round values with `~` (nearest integer, drop trailing zeros after decimal unless < 5). Prefer the **most recent wait-idle benchmark** for concurrency numbers (legacy runs are less accurate due to concurrency overlap).

**Example row:**
```markdown
| **qwen3.6-35b-a3b-nvfp4-mtp** | 35B / 3B | 21.9G | 256k | 13.38x | 1.7–6.1k t/s | 128–189 t/s (C2: ~182 @ d0, ~193 @ d4k, ~65 @ d8k, ~65 @ d16k; C4: ~317 @ d0, ~65 @ d4k, ~33 @ d8k, ~16 @ d16k) | 16.9s |
```

### Filling a Row from Benchmark MD

1. **Prefill:** collect `pp` rows from all files (use C1 rows from multi-concurrency files) → take min/max of means → format `M–Mk t/s`
2. **Gen t/s (C1):** collect `tg` rows from C1-only files → take min/max of means → format `M–M t/s`
3. **Gen t/s (C>1):** from multi-concurrency file (`_c1-...`), use `t/s (total)` column from `tg` rows at each concurrency level (C2, C4, etc.) → list all measured depths → format `(C2: ~X @ d0, ~Y @ d4k, ~Z @ d8k; C4: ~A @ d0, ...)` → **merge** with C1: `M–M t/s (C2: ~X @ d0, ...)`
4. **TTFT @ 64k:** find `pp` row at `d65536` from full-depth single-concurrency C1 test → read `e2e_ttft` → convert ms÷1000 to seconds → format `X.Xs`. If no 64k depth was tested, use the deepest tested depth: `X.Xs (at <depth>)` (e.g., `17.6s (at 32k)`).
5. **Max Concurrency:** from startup log `Maximum concurrency for N tokens per request: Xx` → `Xx`
6. **Params / Model size:** from YAML header or `hf models card`

## Adding a New Model

### 1. Create config

```bash
cp models/template.yaml models/<name>.yaml
```

### 2. Fill in the YAML

Copy from `models/template.yaml` and edit. Required: `image:`, `args:` with at minimum `--model <repo-id>`.

**Critical YAML rules — `env:` and `args:` must be on their own line with values indented below:**

```yaml
# ✅ CORRECT
env:
  VLLM_ATTENTION_BACKEND=FLASHINFER
args:
  --model Qwen/Qwen3-8B
  --tensor-parallel-size 1

# ❌ BROKEN — inline value breaks YAML parsing
env: VLLM_ATTENTION_BACKEND=FLASHINFER
args: --model Qwen/Qwen3-8B
```

**Every model must have a header block** above `image:`:

```yaml
# ── <Short Title> ─────────────────────────────────────────────────────
# <One-line description — architecture, model family>
# <Key specs — params, active params, size, quantization>
#
# Recommended for: <use-cases>
#
# Container: vllm-<model-name>
# API:       http://localhost:<port>/v1/chat/completions
# HF:        https://huggingface.co/<owner>/<repo-id>
# ─────────────────────────────────────────────────────────────────────
```

### 3. Test with DRY_RUN

```bash
./vllm-manager.sh --local start --model <name>
```

Verify `N flags from config` where N > 0. If N=0, the YAML is malformed (check `env:`/`args:` format).

### 4. Verify with HF

If you edited model args, verify the config against the [HuggingFace model card](https://huggingface.co):

```bash
hf models card owner/Model-Name
```

### 5. Run for real

Set `DRY_RUN=false` in `.env` (or remove it), then:

```bash
./vllm-manager.sh start --model <name>
```

## YAML Key Args Reference

| Arg                        | Description         | Common values                          |
| -------------------------- | ------------------- | -------------------------------------- |
| `--model`                  | HF repo ID          | `nvidia/Qwen3.6-35B-A3B-NVFP4`         |
| `--tensor-parallel-size`   | GPU count           | `1`, `2`, `4`                          |
| `--max-model-len`          | Context window      | `32768`, `65536`, `131072`, `262144`   |
| `--gpu-memory-utilization` | VRAM fraction       | `0.85`, `0.9`, `0.95`                  |
| `--attention-backend`      | Attention engine    | `flashinfer`, `sdpa`, `flash_attn`     |
| `--moe-backend`            | MoE backend         | `marlin`, `triton`                     |

## Model Inspection

**Use `hf` CLI — never web_fetch or `hf download`.**

```bash
# Model card — architecture, specs, tags, license
hf models card owner/Model-Name
```

Fields in YAML frontmatter (between `---` markers): `pipeline_tag`, `license`, `tags`, `base_model`, `library_name`.

Fields in card body: Total/Active params, context length, quantization method, disk size, warnings/gotchas.

## KV Cache Max Concurrency

After starting a model, check these log lines:

```
GPU KV cache size: 5,XXX,XXX tokens
Maximum concurrency for XXX,XXX tokens per request: XX.XXx
```

- The `XX.XXx` value tells you how many concurrent 262K (or full context) requests fit physically.
- **`--max-num-seqs` in YAML is the hard vLLM cap.** KV log value should be slightly higher (e.g. log `4.25x` + YAML `--max-num-seqs 4` → correct).
- If KV log < `--max-num-seqs`, increase `--gpu-memory-utilization` in YAML.

## Remote Model Switching

**This is the most critical pattern — always a single `&&` command.**

After `stop-all`, the current container (and this agent process) dies instantly. The `start` must run in the same shell invocation:

```bash
cd /home/administrator/vllm-starters && ./vllm-manager.sh --remote stop-all && ./vllm-manager.sh --remote start --model <name>
```

**Workflow for switching models:**

1. Check current status: `./vllm-manager.sh --remote status`
2. Pull latest: `git pull`
3. Single command to switch:
   ```bash
   cd <project-path> && ./vllm-manager.sh --remote stop-all && ./vllm-manager.sh --remote start --model <model-name>
   ```
4. Verify: `./vllm-manager.sh --remote status`

## API Access

Once started, the model is available at:

```
http://localhost:<port>/v1/chat/completions
```

## Environment Variables

| Variable       | Description                                |
| -------------- | ------------------------------------------ |
| `HF_TOKEN`     | HuggingFace auth token                     |
| `VLLM_API_KEY` | API key (default: `vllm`)                  |
| `DRY_RUN`      | `true` to simulate, unset for real docker  |
| `MODEL`        | Default model name (used when `--model` omitted) |
| `SSH_USER`     | SSH username for remote execution           |
| `SSH_HOST`     | Remote host IP/hostname                     |
| `SSH_KEY`      | SSH private key path                        |
| `SSH_DIR`      | Remote project directory                    |
| `VLLM_REMOTE`  | Set to `0` on remote .env to prevent recursion |
