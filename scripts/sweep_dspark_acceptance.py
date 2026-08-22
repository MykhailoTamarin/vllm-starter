#!/usr/bin/env python3
"""DSPark draft acceptance-rate sweep for the EXL3 DeepSeek-V4-Flash stack.

Runs ON the DGX Spark host (has docker + the model container). For each draft
size K in --ks it:

  1. patches models/<model>.yaml (DSPARK_DRAFT_EXPERTS + DRAFT_DIR +
     speculative-config model path) to K — original YAML restored at the end,
  2. restarts the model via vllm-manager.sh (VLLM_REMOTE=0 DRY_RUN=false
     start), waits for /health,
  3. for each task (coding / chat / text-writing) x --repeats sends a fresh
     *salted* chat request (guarantees no prefix-cache reuse between repeats —
     every test ignores/evades the KV prefix cache),
  4. diffs the v0.26.0 spec-decode Prometheus counters around each request to
     compute the acceptance rate (accepted/draft tokens) and the per-position
     acceptance vector, records sequence KV footprint (prompt+completion) and
     vllm:kv_cache_usage_perc, and measures generation t/s via SSE timing,
  5. writes one MD report per K (k64.md, k96.md, ...) plus a combined summary.

Metrics used (v0.26.0 exact names, from vllm/v1/spec_decode/metrics.py and
vllm/v1/metrics/loggers.py):
  vllm:spec_decode_num_drafts_total
  vllm:spec_decode_num_draft_tokens_total
  vllm:spec_decode_num_accepted_tokens_total
  vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0..4"}
  vllm:kv_cache_usage_perc

Usage (on the DGX, in the repo dir):
  python3 scripts/sweep_dspark_acceptance.py \
      --model deepseek-v4-flash-0731-exl3-dspark \
      --ks "64 96 128 160 192" --repeats 3 --max-tokens 512
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "http://localhost:8000"
TASKS = {
    "coding": (
        "You are a senior Python engineer. Write a production-quality Python module "
        "implementing a thread-safe LRU cache with TTL support and a decorator API. "
        "Include type hints, docstrings, and error handling. Then write 5 pytest unit "
        "tests covering eviction, TTL expiry, and concurrent access.\n\n"
        "[bench-salts: SALT]"
    ),
    "chat": (
        "We are having a casual conversation. Answer like a helpful friend: what is the "
        "most interesting recent development in large language models, and why does it "
        "matter for everyday users? Keep it natural, not academic.\n\n"
        "[bench-salts: SALT]"
    ),
    "text": (
        "Write a well-structured 5-paragraph article (400-500 words) about the impact of "
        "open-source models on scientific research. Include a clear thesis, supporting "
        "arguments, and a conclusion. Use a professional but engaging tone.\n\n"
        "[bench-salts: SALT]"
    ),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def patch_yaml(yaml_path: Path, k: int) -> None:
    text = yaml_path.read_text()
    # deterministic: the draft dir substring appears in DRAFT_DIR and in the
    # speculative-config model path; one replace covers both.
    old_k = re.search(r"dspark-draft-k(\d+)", text)
    if old_k:
        text = text.replace(f"dspark-draft-k{old_k.group(1)}", f"dspark-draft-k{k}")
    new_kv = []
    replaced_exp = False
    for line in text.splitlines(keepends=True):
        m = re.match(r"(\s*DSPARK_DRAFT_EXPERTS=)(\d+)", line)
        if m:
            line = f"{m.group(1)}{k}\n"
            replaced_exp = True
        new_kv.append(line)
    if not replaced_exp:
        raise RuntimeError("DSPARK_DRAFT_EXPERTS line not found in YAML")
    yaml_path.write_text("".join(new_kv))


def manager(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["VLLM_REMOTE"] = "0"
    env["DRY_RUN"] = "false"
    return subprocess.run(
        ["./vllm-manager.sh", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def wait_healthy(timeout_s: int = 2400, poll: int = 15) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{API_BASE}/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(poll)
    return False


_SCRAPE_RE = re.compile(r"^([a-z0-9_:]+)(?:_total)?\{(.*?)\}\s+([0-9.eE+-]+)", re.M)


def scrape_metrics() -> dict:
    """Return {metric_name: float} for the first sample (labels dropped except position)."""
    with urllib.request.urlopen(f"{API_BASE}/metrics", timeout=15) as r:
        text = r.read().decode("utf-8", "replace")
    out: dict = {}
    per_pos: dict = {}
    for name, labels, value in _SCRAPE_RE.findall(text):
        try:
            v = float(value)
        except ValueError:
            continue
        m = re.search(r'position="(\d+)"', labels)
        if m and name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            per_pos[int(m.group(1))] = v
        elif name not in out:
            out[name] = v
    out["_per_pos"] = per_pos
    return out


def snapshot(metrics: dict, key: str) -> float:
    return metrics.get(key, 0.0)


def chat_completion(api_key: str, prompt: str, max_tokens: int, model: str):
    """Stream one completion; return dict with ttft, gen_tps, usage, errors."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.9,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{API_BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    t0 = time.monotonic()
    first_token_t = None
    last_t = t0
    usage = None
    last_err = None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            buf = b""
            while True:
                chunk = r.read(4096)
                if not chunk:
                    break
                buf += chunk
                last_t = time.monotonic()
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices and choices[0].get("delta", {}).get("content"):
                        if first_token_t is None:
                            first_token_t = time.monotonic()
                    if choices and choices[0].get("delta", {}).get("reasoning_content"):
                        if first_token_t is None:
                            first_token_t = time.monotonic()
    except Exception as e:  # noqa: BLE001
        last_err = f"{type(e).__name__}: {e}"
    t1 = last_t
    if first_token_t is None:
        first_token_t = t1
    prompt_tokens = (usage or {}).get("prompt_tokens", 0)
    completion_tokens = (usage or {}).get("completion_tokens", 0)
    ttft = (first_token_t - t0) * 1000.0
    gen_secs = max(t1 - max(first_token_t, t0), 1e-6)
    gen_tps = completion_tokens / gen_secs if completion_tokens else 0.0
    return {
        "ttft_ms": ttft,
        "gen_tps": gen_tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "error": last_err,
    }


def fmt(v) -> str:
    return "—" if v is None else f"{v:.1f}"


def mean_std(vals):
    vals = [
        v
        for v in vals
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    if len(vals) < 2:
        return m, 0.0
    try:
        s = statistics.stdev(vals)
    except Exception:  # noqa: BLE001 — constant/NaN lists (py3.12 quirk)
        s = 0.0
    return m, s


def write_k_report(k: int, rows: list, out_dir: Path, spec: dict) -> None:
    """rows: per-repeat dicts (task, rep, acc, draft, accepted, gen_tps, ttft,
    prompt, completion, kv_tokens, kv_usage_pct, per_pos dict)."""
    md = [
        f"# DSPark acceptance sweep — K{k}",
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- model: {spec['model']}",
        f"- image: {spec['image']}",
        f"- speculative-config: dspark, {spec['n_spec']} tokens, draft_sample_method={spec['sample']}",
        f"- tasks: {', '.join(spec['tasks'])} × {spec['repeats']} repeats, max_tokens={spec['max_tokens']}",
        "- prefix-cache: bypassed (unique salt per request)",
        "",
        "## Per-task averages (3 repeats)",
        "",
        "| task | acceptance % | draft tok | accepted | gen t/s | TTFT ms | prompt tok | completion tok | KV tokens (seq) | KV cache % |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    tasks = sorted({r["task"] for r in rows})
    for t in tasks:
        rs = [r for r in rows if r["task"] == t]
        acc = mean_std([r["acc"] for r in rs])
        draft = mean_std([r["draft"] for r in rs])
        accepted = mean_std([r["accepted"] for r in rs])
        tps = mean_std([r["gen_tps"] for r in rs])
        ttft = mean_std([r["ttft"] for r in rs])
        pr = mean_std([r["prompt"] for r in rs])
        co = mean_std([r["completion"] for r in rs])
        kv = mean_std([r["kv_tokens"] for r in rs])
        kvp = mean_std([r["kv_usage_pct"] for r in rs])
        acc_s = "—" if acc[0] is None else f"{acc[0]*100:.1f} ± {acc[1]*100:.1f}"
        md.append(
            f"| {t} | {acc_s} | {fmt(draft[0])} | {fmt(accepted[0])} | "
            f"{fmt(tps[0])} ± {fmt(tps[1])} | {fmt(ttft[0])} | {fmt(pr[0])} | "
            f"{fmt(co[0])} | {fmt(kv[0])} | {fmt(kvp[0])} |"
        )
    md += [
        "",
        "## Acceptance by draft position (avg over all repeats)",
        "",
        "| task | pos0 | pos1 | pos2 | pos3 | pos4 |",
        "|---|---|---|---|---|---|",
    ]
    for t in tasks:
        rs = [r for r in rows if r["task"] == t]
        pos_means = []
        for p in range(spec["n_spec"]):
            vals = [r["per_pos"].get(p, None) for r in rs]
            vals = [v for v in vals if v is not None]
            pos_means.append(fmt(statistics.mean(vals) if vals else None))
        md.append(f"| {t} | " + " | ".join(pos_means) + " |")
    md += [
        "",
        "## Raw repeats",
        "",
        "| # | task | acc % | draft | accepted | gen t/s | ttft ms | prompt | completion | kv seq | kv % |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        md.append(
            f"| {i} | {r['task']} | {r['acc']*100:.1f} | {r['draft']} | {r['accepted']} | "
            f"{r['gen_tps']:.1f} | {r['ttft']:.0f} | {r['prompt']} | {r['completion']} | "
            f"{r['kv_tokens']} | {r['kv_usage_pct']:.1f} |"
        )
    md.append("")
    (out_dir / f"k{k}.md").write_text("\n".join(md))


def write_summary(k_rows: dict, out_dir: Path, ks: list) -> None:
    md = [
        "# DSPark acceptance sweep — combined summary",
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- draft sizes: {', '.join(str(k) for k in ks)}",
        "",
        "## Acceptance % by task × K",
        "",
        "| K | coding | chat | text |",
        "|---|---|---|---|",
    ]
    for k in ks:
        rows = k_rows[k]
        cells = []
        for t in ["coding", "chat", "text"]:
            rs = [r for r in rows if r["task"] == t]
            m, _ = mean_std([r["acc"] for r in rs])
            cells.append("—" if m is None else f"{m*100:.1f}")
        md.append(f"| k{k} | " + " | ".join(cells) + " |")
    md += [
        "",
        "## Generation t/s by task × K",
        "",
        "| K | coding | chat | text |",
        "|---|---|---|---|",
    ]
    for k in ks:
        rows = k_rows[k]
        cells = []
        for t in ["coding", "chat", "text"]:
            rs = [r for r in rows if r["task"] == t]
            m, _ = mean_std([r["gen_tps"] for r in rs])
            cells.append("—" if m is None else f"{m:.1f}")
        md.append(f"| k{k} | " + " | ".join(cells) + " |")
    md += [
        "",
        "## KV tokens per request (prompt+completion) by task × K",
        "",
        "| K | coding | chat | text |",
        "|---|---|---|---|",
    ]
    for k in ks:
        rows = k_rows[k]
        cells = []
        for t in ["coding", "chat", "text"]:
            rs = [r for r in rows if r["task"] == t]
            m, _ = mean_std([r["kv_tokens"] for r in rs])
            cells.append("—" if m is None else f"{m:.0f}")
        md.append(f"| k{k} | " + " | ".join(cells) + " |")
    md += [
        "",
        "## KV cache usage % (after request) by task × K",
        "",
        "| K | coding | chat | text |",
        "|---|---|---|---|",
    ]
    for k in ks:
        rows = k_rows[k]
        cells = []
        for t in ["coding", "chat", "text"]:
            rs = [r for r in rows if r["task"] == t]
            m, _ = mean_std([r["kv_usage_pct"] for r in rs])
            cells.append("—" if m is None else f"{m:.2f}")
        md.append(f"| k{k} | " + " | ".join(cells) + " |")
    md.append("")
    (out_dir / "draft-acceptance-summary.md").write_text("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash-0731-exl3-dspark")
    ap.add_argument("--ks", default="64 96 128 160 192")
    ap.add_argument("--tasks", default="coding chat text")
    ap.add_argument("--api-model", default=None,
                    help="name the API server knows (--served-model-name); "
                         "default: parsed from the model YAML")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    env = read_env(repo / ".env")
    api_key = env.get("VLLM_API_KEY") or os.environ.get("VLLM_API_KEY", "")
    ks = [int(x) for x in args.ks.split()]
    tasks = args.tasks.split()

    yaml_path = repo / "models" / f"{args.model}.yaml"
    yaml_orig = yaml_path.read_text()

    # served model name (the YAML's --served-model-name) is what the API knows
    api_model = args.api_model
    if not api_model:
        m = re.search(r"--served-model-name\s+(\S+)", yaml_orig)
        if not m:
            raise SystemExit("could not determine --served-model-name from YAML")
        api_model = m.group(1)

    # spec config from current YAML
    n_spec = 5
    m = re.search(r"num_speculative_tokens\":(\d+)", yaml_orig)
    if m:
        n_spec = int(m.group(1))
    sample = "probabilistic"
    m = re.search(r"draft_sample_method\":\"(\w+)\"", yaml_orig)
    if m:
        sample = m.group(1)
    image = "?"
    m = re.search(r"^image:\s*(\S+)", yaml_orig, re.M)
    if m:
        image = m.group(1)

    bench_dir_name = f"draft-acceptance-sweep-{time.strftime('%Y-%m-%d')}"
    out_dir = (
        repo
        / "models"
        / "benchmarks"
        / args.model
        / bench_dir_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "model": args.model,
        "image": image,
        "n_spec": n_spec,
        "sample": sample,
        "tasks": tasks,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
    }

    k_rows: dict[int, list] = {}
    try:
        for k in ks:
            log(f"=== K{k}: patching YAML, restarting model ===")
            patch_yaml(yaml_path, k)
            # vllm-manager's cmd_start does NOT remove an existing container —
            # stop first so `docker run` can't hit a name conflict.
            st = manager(repo, "stop", "--model", args.model)
            log(f"    manager stop rc={st.returncode}")
            res = manager(repo, "start", "--model", args.model)
            log(f"    manager start rc={res.returncode}: {res.stdout.strip()[-400:]}")
            if res.returncode != 0:
                log(f"    STDERR: {res.stderr.strip()[-800:]}")
                raise SystemExit(f"manager start failed for K{k}")
            log("    waiting for /health ...")
            if not wait_healthy():
                log("    HEALTH TIMEOUT — dumping log tail")
                lr = manager(repo, "logs", "--model", args.model, "--tail", "60")
                log(lr.stdout)
                raise SystemExit(f"model did not become healthy for K{k}")
            log("    model up; settling 30s")
            time.sleep(30)

            probe = chat_completion(
                api_key,
                "Reply with the single word ok. probe-salt",
                1,
                api_model,
            )
            if probe["error"] or probe["completion_tokens"] == 0:
                log(f"    API PROBE FAILED: {probe['error']} — aborting K{k}")
                raise SystemExit(
                    f"API not served for K{k} (probe error: {probe['error']})"
                )
            log("    API probe OK")

            rows: list = []
            # zero/re-align each repeat (fresh graph, fresh sequence)
            for rep in range(1, args.repeats + 1):
                for task in tasks:
                    salt = f"k{k}-{task}-{rep}-{os.urandom(4).hex()}"
                    prompt = TASKS[task].replace("SALT", salt)
                    m0 = scrape_metrics()
                    a0 = snapshot(m0, "vllm:spec_decode_num_accepted_tokens")
                    d0 = snapshot(m0, "vllm:spec_decode_num_draft_tokens")
                    kv0 = snapshot(m0, "vllm:kv_cache_usage_perc")
                    pos0 = dict(m0.get("_per_pos", {}))
                    r = chat_completion(api_key, prompt, args.max_tokens, api_model)
                    m1 = scrape_metrics()
                    a1 = snapshot(m1, "vllm:spec_decode_num_accepted_tokens")
                    d1 = snapshot(m1, "vllm:spec_decode_num_draft_tokens")
                    kv1 = snapshot(m1, "vllm:kv_cache_usage_perc")
                    pos1 = dict(m1.get("_per_pos", {}))
                    accepted = max(a1 - a0, 0)
                    draft = max(d1 - d0, 0)
                    acc = accepted / draft if draft > 0 else float("nan")
                    per_pos = {
                        p: (pos1.get(p, 0) - pos0.get(p, 0)) / draft * 100.0
                        if draft > 0
                        else None
                        for p in range(n_spec)
                    }
                    row = {
                        "task": task,
                        "rep": rep,
                        "acc": acc,
                        "draft": int(draft),
                        "accepted": int(accepted),
                        "gen_tps": r["gen_tps"],
                        "ttft": r["ttft_ms"],
                        "prompt": r["prompt_tokens"],
                        "completion": r["completion_tokens"],
                        "kv_tokens": r["total_tokens"],
                        "kv_usage_pct": kv1,
                        "per_pos": per_pos,
                        "error": r["error"],
                    }
                    rows.append(row)
                    log(
                        f"    K{k} {task} rep{rep}: acc={acc*100:.1f}% "
                        f"({int(accepted)}/{int(draft)}) gen={r['gen_tps']:.1f} t/s "
                        f"ttft={r['ttft_ms']:.0f}ms kv_tok={r['total_tokens']} "
                        f"kv%={kv1:.2f}{' ERR:'+row['error'] if row['error'] else ''}"
                    )
            k_rows[k] = rows
            write_k_report(k, rows, out_dir, spec)
            log(f"    K{k} report written -> {out_dir / f'k{k}.md'}")
    finally:
        log("restoring original YAML")
        yaml_path.write_text(yaml_orig)

    write_summary(k_rows, out_dir, ks)
    log(f"DONE. reports in {out_dir}")


if __name__ == "__main__":
    main()