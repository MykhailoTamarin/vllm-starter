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

`llama-bench.sh` wraps [llama-benchy](https://github.com/eugr/llama-benchy). Auto-builds base-url from `.env SSH_HOST` + `VLLM_API_KEY`, resolves model from YAML config.

**Required:** `llama-benchy` installed (`uvx llama-benchy`).

| Command                           | Description                                |
| --------------------------------- | ------------------------------------------ |
| `llama-bench.sh --model <name>`   | Standard mode (single benchy call)         |
| `llama-bench.sh --model <name> --wait-idle` | Sequential mode (wait-for-idle between each {C×D} test) |
| `+ --depth <d1> <d2> ...`         | Context depths to test                     |
| `+ --concurrency <c1> <c2> ...`   | Parallel client counts                     |
| `+ --latency-mode generation`     | Measure server-side latency (recommended)  |
| `+ --repeat N`                    | Run the full suite N times (wait-idle mode only) |

### Running Benchmarks

> **Recommended:** Always use `--wait-idle`. It waits for GPU idle between each {C×D} test, preventing concurrency overlap that skews results.

#### Benchmark output structure

Each wait-idle benchmark run creates:
- **Raw JSONs**: `models/benchmarks/<model>/<c<N>_d<D>/` — gitignored, contains detailed benchy JSON per run
  - `<c<N>_d<D>/` uses concurrency and depth from command, e.g., `c1_d0_1024_2048`, `c1_2_d0_1024`
  - Each file: `c<C>_d<D>_r<R>_s<S>.json` (c=concurrency, d=depth, r=run, s=suite)
- **Parsed MD**: `models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c<N>_d<D>.md` — tracked by git
  - Auto-generated at end of each wait-idle run by `scripts/bench-parse.sh`
  - Contains aggregated markdown table with prefill + generation throughput

Manual parse (any folder):
```bash
./scripts/bench-parse.sh -d models/benchmarks/<model>/<c><d>/ -o results.md
```

#### Single concurrency, full depth (default workflow)

```bash
# C=1 only, full context: 0, 4k, 8k, 16k, 32k, 64k, 128k — 3 reps each
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --wait-idle --depth 0 4096 8192 16384 32768 65536 131072 --repeat 3
```

Output folder: `models/benchmarks/<model>/c1_d0_4096_8k_...` (gitignored)
Results MD: `models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_d0_4096_8k_....md` (tracked)

#### Multi-concurrency with idle gates (caps at 16k depth)

```bash
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --wait-idle --depth 1024 2048 4096 8192 16384 --concurrency 1 2 4 --repeat 3
```

Flow:
```
Suite 1: idle → C=1 d=1024 → idle → C=2 d=1024 → idle → C=4 d=1024 → idle → C=1 d=2048 → ...
Suite 2: idle → C=1 d=1024 → idle → C=2 d=1024 → idle → C=4 d=1024 → idle → C=1 d=2048 → ...
Suite 3: idle → C=1 d=1024 → idle → C=2 d=1024 → idle → C=4 d=1024 → idle → C=1 d=2048 → ...
```

Output folder: `models/benchmarks/<model>/c1_2_4_d1024_2048_...` (gitignored)
Results MD: `models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_2_4_d1024_...md` (tracked)

Results are gitignored (raw JSON files). Auto-generated MD is tracked.

#### Legacy Mode (original behavior)

Single benchy call, all tests run together without idle gates. For quick single-pass checks only.

```bash
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 0 4096 8192 --latency-mode generation
```

Results auto-save to `models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>[_c{C}_...].md` (tracked).

### Parsing Benchmark Results

Benchmark MD files contain markdown tables. Key patterns in the `test` column:

- `pp2048` — prefill throughput (2048 tokens input)
- `tg32` — generation throughput (32 tokens output)
- `pp2048 @ d4096` — prefill at 4096 token context depth
- `tg32 (cN)` — generation throughput at concurrency N (multi-concurrency files only)

Values are always formatted as `mean ± stddev` — use the `mean` value.

### Where to find results

Always start with the **Parsed MD** (source of truth for summary stats):

- **Parsed MD** (tracked by git):
  ```
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_d0_1024.md
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_2_4_d1024_2048.md
  ```
  Auto-generated by `scripts/bench-parse.sh` after each wait-idle run. Contains the markdown table from the benchmark output with per-depth metrics.

  Manual parse (if no MD exists):
  ```bash
  ./scripts/bench-parse.sh -d models/benchmarks/<model>/<folder> -o results.md
  ```

- **Raw JSONs** (gitignored, use only for deep inspection):
  ```
  models/benchmarks/<model>/c1_d0_1024_2048/
    c1_d0_r1_s1.json      # C=1, d=0, run=1, suite=1
    c2_d0_r1_s2.json      # C=2, d=0, run=1, suite=2
    ...
  ```
  Each file has: `{benchmarks: [{pp_throughput: {mean, std}, tg_throughput: {mean, std}, ...}]}`

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
| Status | Benchmark exists? | `✅ **Tested**` / `⬜ Untested` |

**Concurrency notes:** Only append if concurrency tests were run. Use the `t/s` column from the parsed benchmark MD (auto-generated in `models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_2_4_<depths>.md`). For multi-concurrency files, look for `tg32 (cN)` rows to get total throughput at concurrency N. Use `~` for approximate values. Skip depth 0 (zero-context) — only include non-zero depths. Prefer the **most recent wait-idle benchmark** for concurrency numbers (legacy runs are less accurate due to concurrency overlap). Format: `(C2: ~190 @ 1k, C4: ~260 @ 1k; C2: ~177 @ 2k, C4: ~191 @ 2k)` — list representative non-zero depth examples showing total throughput at each concurrency level, prioritising low-depth values (1k–2k) where concurrency scales best. Only include depth points where the test completed (all 3 runs).

**Example row:**
```markdown
| **qwopus3.5-122b-a10b-kimi-k2.6-nvfp4-mtp** | 122B / ~10B | 75.9G | 256k | 4.25x | 1.0–2.3k t/s | 24–30 t/s (C2: ~39 @ 4k) | 47.0s | ✅ **Tested** |
```

### Filling a Row from Benchmark MD

1. **Prefill:** collect `pp` rows from all files (use C1 rows from multi-concurrency files) → take min/max of means → format `M–Mk t/s`
2. **Gen t/s:** collect `tg` rows from all files (C1 only) → take min/max of means → format `M–M t/s`
3. **TTFT @ 64k:** find `pp` row at `d65536` → read `e2e_ttft` → convert ms÷1000 to seconds → format `X.Xs` (full-depth single-concurrency test includes this)
4. **Max Concurrency:** from startup log `Maximum concurrency for N tokens per request: Xx` → `Xx`
5. **Params / Model size:** from YAML header or `hf models card`

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
