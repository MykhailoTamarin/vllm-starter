"""Backport of tonyd2wild Patch 5: don't evaluate client stop strings inside
the reasoning segment (fixes null/empty content on think-in-prompt models).

WHY
vLLM's v1 detokenizer matches client `stop` strings against the whole output
stream.  With think-in-prompt templates (DeepSeek V4, Qwen3) generation begins
INSIDE the reasoning segment, and chain-of-thought naturally restates phrases
like "Question:".  Any harness that sends stop sequences (lm-evaluation-harness
sends stop[:4] on every request) decapitates the reasoning mid-think,
``<response>`` never arrives, so the reasoning parser yields content = null.

Reproduced live on vllm-0_26_0-exl3 (hexagon prompt, seed 23): control returns
a full answer; adding stop=["Question"] deterministically returns
``content: None`` with ``finish_reason: stop``.  The guard is backend-agnostic
and reads markers from ``--reasoning-config``, so it adapts to this image's
``<thinking>`` / ``</thinking>`` without being DeepSeek-specific.

WHAT
Per-request guard.  At detokenizer construction, if the request's LAST PROMPT
TOKEN is the reasoning start marker, stop strings stay dormant until the
reasoning end marker appears in the output.  EOS and max_tokens are unaffected.
Non-thinking requests are untouched.  On closing the marker, stop evaluation
advances past it (spec-decode chunks carry up to k+1 tokens, so reasoning that
preceded the close must not fire a stop).

Opt-out: VLLM_SUPPRESS_STOPS_IN_REASONING=0 (process-wide).

Anchored against vLLM v0.26.0 (vllm/v1/engine/detokenizer.py).
"""

import os
import sys
from pathlib import Path

DETOK = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py"
)

content = DETOK.read_text()
results = []

if "VLLM_SUPPRESS_STOPS_IN_REASONING" in content:
    print("SKIP  detokenizer: stop-in-reasoning guard already applied")
    sys.exit(0)

import py_compile  # noqa: E402   (imported late to keep preamble clean)

# --- import os at the top ---
a_import = "from abc import ABC, abstractmethod"
r_import = "import os\nfrom abc import ABC, abstractmethod"
if a_import not in content:
    results.append("FAIL  detokenizer: abc import anchor not found")
    sys.exit(1)
content = content.replace(a_import, r_import, 1)
results.append("OK    detokenizer: added os import")

# --- 1) Rewrite from_new_request to arm the guard after construction ---
a_from = """        if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)"""

r_from = """        if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
            # Fast tokenizer => use tokenizers library DecodeStream.
            detok = FastIncrementalDetokenizer(tokenizer, request)
        else:
            # Fall back to slow python-based incremental detokenization.
            detok = SlowIncrementalDetokenizer(tokenizer, request)
        # PATCH(stop-in-reasoning): arm the guard before returning (see
        # BaseIncrementalDetokenizer.__init__).
        IncrementalDetokenizer._maybe_enable_reasoning_guard(detok, tokenizer, request)
        return detok

    @staticmethod
    def _reasoning_markers() -> tuple[str, str]:
        # PATCH(stop-in-reasoning): prefer the deployment's own
        # --reasoning-config over hardcoded markers, so the guard is general
        # rather than DeepSeek-specific. Falls back to the common defaults.
        start, end = " thinking", " response"
        try:
            from vllm.config import get_current_vllm_config_or_none

            cfg = get_current_vllm_config_or_none()
            rc = getattr(cfg, "reasoning_config", None) if cfg else None
            if rc is not None:
                start = getattr(rc, "reasoning_start_str", "") or start
                end = getattr(rc, "reasoning_end_str", "") or end
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(
                "stop-in-reasoning: no reasoning config (%s); using %r / %r",
                e, start, end,
            )
        return start, end

    @staticmethod
    def _maybe_enable_reasoning_guard(detok, tokenizer, request) -> None:
        # PATCH(stop-in-reasoning): see BaseIncrementalDetokenizer.__init__.
        # NOTE: this opt-out is process-wide, not per-request. A caller who
        # deliberately wants to bound reasoning with a stop string has no
        # escape hatch short of restarting with it set to 0.
        if os.environ.get("VLLM_SUPPRESS_STOPS_IN_REASONING", "1") == "0":
            return
        try:
            stop = getattr(detok, "stop", None)
            ptids = getattr(request, "prompt_token_ids", None)
            try:
                # TEMP diagnostic: surface the real prompt tail for arming
                print(
                    "DBG-ENTRY stop=" + repr(stop) + " ptids_len=" + str(len(ptids) if ptids else 0)
                    + " ptids_tail=" + repr((ptids or [])[-8:]) + " ptids_last=" + repr((ptids or [])[-1:]),
                    file=open("/tmp/guard2.log", "a"),
                )
            except Exception:
                pass
            if not stop or not ptids:
                return
            start_str, end_str = IncrementalDetokenizer._reasoning_markers()
            # The reasoning-start marker is NOT reliably a single token and its
            # tokenization differs between the fast BPE tokenizer and
            # tokenizer_mode='deepseek_v4' (DeepSeek-V4 serves its own vocab:
            # ' thinking' is special id 128821 there vs BPE id 6892 in the HF
            # vocab; '<thinking>' -> [30,77291,32]).  Re-encoding the marker is
            # therefore unreliable, and decoding a multi-token tail can carry
            # trailing byte/special-token artifacts.  Instead, inspect the LAST
            # prompt token: convert its id back to a string and test whether it
            # equals / endswith the reasoning-start marker.  This is robust
            # because convert_ids_to_tokens() maps the special id correctly
            # (verified: 128821 -> ' thinking', 128822 -> ' response').
            candidates = [start_str]
            for extra in (" thinking", "<thinking>", " thinking"):
                if extra not in candidates:
                    candidates.append(extra)
            last_id = ptids[-1]
            try:
                tail_str = tokenizer.decode([last_id])
            except Exception:
                tail_str = ""
            caller = None
            for c in candidates:
                if not c:
                    continue
                if tail_str == c or tail_str.endswith(c) or c.endswith(tail_str.strip()):
                    caller = c
                    break
            if caller is None:
                # Final fallback: strip whitespace on the last token and match
                # against the marker without its leading space.
                stripped = tail_str.strip()
                for c in candidates:
                    if c and stripped in c or c.strip() == stripped:
                        caller = c
                        break
            if caller:
                detok._reasoning_stop_guard = True
                # Determine the end marker(s) to close on. The configured
                # --reasoning-config end can be "" while the served
                # tokenizer_mode='deepseek_v4' actually emits " response"
                # (special id 128822).  Close on whichever of the config marker
                # or the tokenizer-mode marker actually appears in the stream.
                end_candidates = [end_str] if end_str else []
                for end_c in (" response", "</thinking>"):
                    if end_c not in end_candidates:
                        end_candidates.append(end_c)
                detok._reasoning_end_strs = end_candidates if end_candidates else [" response"]
                try:
                    print(
                        "DBG-ARM caller=" + repr(caller) + " end_strs=" + repr(detok._reasoning_end_strs)
                        + " tail=" + repr(tail_str) + " lastid=" + repr(ptids[-1]),
                        file=open("/tmp/guard2.log", "a"),
                    )
                except Exception:
                    pass
                detok._reasoning_end_str = detok._reasoning_end_strs[0]
        except Exception as e:
            # Never swallow silently: a renamed attribute would turn this fix
            # into a no-op, and that failure mode looks exactly like "the
            # patch is not installed".
            logger.debug("stop-in-reasoning: guard not armed (%s)", e)"""

if a_from not in content:
    results.append("FAIL  detokenizer: from_new_request anchor not found")
    sys.exit(1)
content = content.replace(a_from, r_from, 1)
results.append("OK    detokenizer: guard wired into from_new_request")

# --- 2) Add guard state attrs to BaseIncrementalDetokenizer.__init__ ---
a_init = """        self._last_output_text_offset: int = 0

        # Generation data
        self.output_text = \"\""""

r_init = """        self._last_output_text_offset: int = 0

        # PATCH(stop-in-reasoning): when the prompt ends with the reasoning
        # opening token (prompt-side think marker, as DeepSeek-V4/Qwen3
        # templates do), client stop strings must not match inside the
        # reasoning segment -- matching there truncates mid-think and yields
        # content=None. Stops stay dormant until the reasoning end marker
        # appears in the output. Disable with VLLM_SUPPRESS_STOPS_IN_REASONING=0
        # (process-wide).
        self._reasoning_stop_guard: bool = False
        self._reasoning_closed: bool = False
        self._reasoning_end_str: str = "</thinking>"
        # PATCH: list of end markers to close reasoning on -- the configured
        # marker and the tokenizer-mode ones, so the guard closes whichever the
        # served tokenizer actually emits first.
        self._reasoning_end_strs: list[str] = ["</thinking>", " response"]

        # Generation data
        self.output_text = \"\""""

if a_init not in content:
    results.append("FAIL  detokenizer: __init__ anchor not found")
    sys.exit(1)
content = content.replace(a_init, r_init, 1)
results.append("OK    detokenizer: reasoning-guard state added to __init__")

# --- 3) Gate stop evaluation on reasoning state in update() ---
a_upd = """        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:"""

r_upd = """        # 2) Evaluate stop strings.
        # PATCH(stop-in-reasoning): keep stops dormant while reasoning is open.
        if self._reasoning_stop_guard and not self._reasoning_closed:
            try:
                print(
                    "DBG-UP guard=1 closed=0 end=" + repr(getattr(self, "_reasoning_end_strs", None))
                    + " text=" + repr(self.output_text[-40:]),
                    file=open("/tmp/guard2.log", "a"),
                )
            except Exception:
                pass
            end_strs = getattr(self, "_reasoning_end_strs", None) or [
                self._reasoning_end_str
            ]
            closest = None
            for marker in end_strs:
                window = max(0, stop_check_offset - (len(marker) - 1))
                idx = self.output_text.find(marker, window)
                if idx != -1 and (closest is None or idx < closest[0]):
                    closest = (idx, marker)
            if closest is not None:
                idx, marker = closest
                self._reasoning_closed = True
                # Only evaluate stops over what FOLLOWS the marker.  With
                # speculative decoding this same update() carries up to k+1
                # tokens, so the reasoning that preceded the close arrives in
                # this very chunk and must not be able to fire a stop.
                stop_check_offset = max(stop_check_offset, idx + len(marker))
        stop_string = None
        if (
            self.stop
            and self.num_output_tokens() > self.min_tokens
            and (not self._reasoning_stop_guard or self._reasoning_closed)
        ):"""

if a_upd not in content:
    results.append("FAIL  detokenizer: update() anchor not found")
    sys.exit(1)
content = content.replace(a_upd, r_upd, 1)
results.append("OK    detokenizer: stop evaluation gated on reasoning state")

DETOK.write_text(content)

py_compile.compile(str(DETOK), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)
