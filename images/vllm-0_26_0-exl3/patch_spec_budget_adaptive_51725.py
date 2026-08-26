"""Backport vLLM PR #51725 onto v0.26.0: adaptive budget for spec-scheduled
input tokens.

Upstream: commit 0914ed2e81 ("[Perf] Adaptive budget for spec scheduled input
          tokens, ~60% better Kimi K3 DSpark TTFT (#51725)")

What it does
------------
Speculative decoding used to reserve draft slots globally: vLLM decreased
max_num_scheduled_tokens by (draft slots x max_num_seqs) up front, and the
scheduler did not separately account for the draft-slot consumption of the
input token budget. With DSPark/DFlash drafts the prefill budget could be
starved by drafting slots.

The fix makes the budget ADAPTIVE at schedule time:
  * input_budget tracks max_num_batched_tokens across running + waiting
    scheduling;
  * each scheduled request consumes (num_new_tokens + draft_slots) from it;
  * preemption restores reclaim both; loops break when input_budget <=
    draft_slots;
  * max_num_scheduled_tokens is no longer pre-reduced (config side).

EXL3 relevance: EXL3 serves DeepSeek-V4-Flash with a compact DSPark K160 draft
(MRV2 forced by DSpark) under --max-num-seqs 2 / --max-num-batched-tokens
2048. The same draft-slot starvation mechanism applies to its prefill/TTFT
path, and the scheduler code in v0.26.0 is line-identical to the upstream
parent.

Scope ported
------------
All hunks of the upstream commit:
  config/vllm.py   — _set_max_num_scheduled_tokens (4 sub-hunks)
  scheduler.py     — Scheduler.schedule (6 sub-hunks A..F)

Faithful vs adapted
-------------------
FAITHFUL. v0.26.0 text for every sub-hunk matches the upstream OLD side exactly
(verified against ``git show v0.26.0:...`` for both files); upstream NEW text
copied verbatim.

Container targets
-----------------
/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py
/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
"""

import os
from pathlib import Path
import py_compile  # noqa: E402

CONFIG = Path(
    os.environ.get("VLLM_CFG", "/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py")
)
SCHED = Path(
    os.environ.get(
        "VLLM_SCHED",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)


def apply(content: str, old: str, new: str, label: str) -> str:
    if new in content:
        print(f"SKIP  {label}: already applied")
        return content
    n = content.count(old)
    if n != 1:
        print(f"FAIL  {label}: expected 1 anchor, found {n}")
        raise SystemExit(1)
    print(f"OK    {label}")
    return content.replace(old, new, 1)


# ── config/vllm.py ──────────────────────────────────────────────────────────
C1_OLD = """        In most cases, the scheduler may schedule a batch with as many tokens as the
        worker is configured to handle. However for some speculative decoding methods,
        the drafter model may insert additional slots into the batch when drafting.
        To account for this, we need to decrease the max_num_scheduled_tokens by an
        upper bound on the number of slots that can be added.
        \"\"\"
        if self.speculative_config is not None:
            scheduled_token_delta = (
                self.speculative_config.max_num_new_slots_for_drafting
                * self.scheduler_config.max_num_seqs
            )
            max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
            if self.scheduler_config.max_num_scheduled_tokens is None:
                self.scheduler_config.max_num_scheduled_tokens = (
                    max_num_batched_tokens - scheduled_token_delta
                )"""

C1_NEW = """        In most cases, the scheduler may schedule a batch with as many tokens as the
        worker is configured to handle.
        \"\"\"
        if self.speculative_config is not None:
            scheduled_token_delta = (
                self.speculative_config.max_num_new_slots_for_drafting
            )
            max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
            if self.scheduler_config.max_num_scheduled_tokens is None:
                self.scheduler_config.max_num_scheduled_tokens = max_num_batched_tokens"""

C2_OLD = """                    \" any tokens to be scheduled. Increase max_num_batched_tokens\"
                    \" to accommodate the additional draft token slots, or decrease\"
                    \" num_speculative_tokens or max_num_seqs.\"
                )"""

C2_NEW = """                    \" any tokens to be scheduled. Increase max_num_batched_tokens\"
                    \" to accommodate the additional draft token slots, or decrease\"
                    \" num_speculative_tokens.\"
                )"""

C3_OLD = """                    \" the speculative decoding settings. This may lead to suboptimal\"
                    \" performance. Consider increasing max_num_batched_tokens to\"
                    \" accommodate the additional draft token slots, or decrease\"
                    \" num_speculative_tokens or max_num_seqs.\",
                )"""

C3_NEW = """                    \" the speculative decoding settings. This may lead to suboptimal\"
                    \" performance. Consider increasing max_num_batched_tokens to\"
                    \" accommodate the additional draft token slots, or decrease\"
                    \" num_speculative_tokens.\",
                )"""

C4_OLD = """            max_num_scheduled_tokens = self.scheduler_config.max_num_scheduled_tokens
            if max_num_batched_tokens < max_num_scheduled_tokens + (
                self.speculative_config.max_num_new_slots_for_drafting
                * self.scheduler_config.max_num_seqs
            ):
                raise ValueError("""

C4_NEW = """            max_num_scheduled_tokens = self.scheduler_config.max_num_scheduled_tokens
            if max_num_batched_tokens <= scheduled_token_delta:
                raise ValueError("""

# ── scheduler.py ────────────────────────────────────────────────────────────
S1_OLD = """        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:"""

S1_NEW = """        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        input_budget = self.scheduler_config.max_num_batched_tokens
        if self._pause_state == PauseState.PAUSED_ALL:"""

S2_OLD = """        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
"""

S2_NEW = """        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if input_budget <= draft_slots:
                break
"""

S3_OLD = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)
"""

S3_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
"""

S4_OLD = """                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
"""

S4_NEW = """                            restored = num_scheduled_tokens.pop(preempted_req_id)
                            token_budget += restored
                            input_budget += restored + draft_slots
"""

S5_OLD = """            token_budget -= num_new_tokens
            req_index += 1
"""

S5_NEW = """            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            req_index += 1
"""

S6_OLD = """            while (self.waiting or self.skipped_waiting) and token_budget > 0:
"""

S6_NEW = """            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if input_budget <= draft_slots:
                    break
"""

cfg = CONFIG.read_text()
for label, (old, new) in {
    "config hunk1 (draft slot delta)": (C1_OLD, C1_NEW),
    "config hunk2 (error msg)": (C2_OLD, C2_NEW),
    "config hunk3 (warning msg)": (C3_OLD, C3_NEW),
    "config hunk4 (tail check)": (C4_OLD, C4_NEW),
}.items():
    cfg = apply(cfg, old, new, label)
CONFIG.write_text(cfg)

sched = SCHED.read_text()
for label, (old, new) in {
    "sched A (input_budget init)": (S1_OLD, S1_NEW),
    "sched B (running break)": (S2_OLD, S2_NEW),
    "sched C (token min)": (S3_OLD, S3_NEW),
    "sched D (preempt restore)": (S4_OLD, S4_NEW),
    "sched E (budget consume)": (S5_OLD, S5_NEW),
    "sched F (waiting break)": (S6_OLD, S6_NEW),
}.items():
    sched = apply(sched, old, new, label)
SCHED.write_text(sched)

py_compile.compile(str(CONFIG), doraise=True)
py_compile.compile(str(SCHED), doraise=True)
print("OK    compile (config + scheduler)")