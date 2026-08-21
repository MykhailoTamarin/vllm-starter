"""Strip spurious 'Skip'-suffix artifacts from DeepSeek-V4 chat responses.

The DSPark speculative decoder occasionally injects single-vocab-token artifacts
mid-reply (corrupt draft tokens that slip past rejection sampling). All known
forms are the punctuation-prefixed 'Skip' tokens that never appear in
legitimate prose:
    )Skip   (id 83480)
    ,Skip   (id 121099)
    .Skip   (id 26104)
    .skip   (id 66477)
    ' Skip' (id 60920, leading space)

Bare 'Skip' / 'skip' / 'skipped' / 'skipping' are legitimate English and are NOT
touched. We remove the artifact strings from client-visible content at the
OpenAI chat serving layer, for BOTH streaming (per-chunk deltas) and
non-streaming (final message content). The reasoning field is untouched.

IMPORTANT: sanitizing is applied on the parser AND non-parser paths. DeepSeek V4
runs with --reasoning-parser deepseek_v4 (and auto tools), so vLLM's serving.py
always has a non-None parser; inserting the strip call only inside the
`parser is None` / auto-tool branches let the artifact through on every real
request.

Repo-convention patch script: run as a Docker build step (COPY + RUN python3)
after vLLM is installed.
"""
from pathlib import Path

SRV = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"
)
results: list[str] = []

if not SRV.exists():
    results.append("FAIL  serving.py not found")
    for r in results:
        print(r)
    raise SystemExit(1)

content = SRV.read_text()

# --- Inject the sanitizer helper before the OpenAIServingChat class ---
HELPER = '''
_SPURIOUS_ARTIFACTS = (
    ")Skip",
    ",Skip",
    ".Skip",
    ".skip",
    " Skip",  # leading-space form (token id 60920)
)


def _strip_spurious_artifacts(text):
    # PATCH(ds4-skip): remove DSPark draft 'Skip'-suffix artifacts that are
    # occasionally accepted mid-reply. Only the punctuation-prefixed forms are
    # stripped (never bare 'skip'/'Skip', which is legitimate English). Collapses
    # the double space the removal can leave behind.
    if not text or text == "":
        return text
    stripped = text
    for art in _SPURIOUS_ARTIFACTS:
        stripped = stripped.replace(art, "")
    import re as _re
    stripped = _re.sub(r" {2,}", " ", stripped)
    return stripped


class OpenAIServingChat'''

a_cls = "\n\n\nclass OpenAIServingChat"
if a_cls in content and "def _strip_spurious_artifacts" not in content:
    content = content.replace(a_cls, HELPER, 1)
    results.append("OK    sanitizer helper added before OpenAIServingChat")
else:
    results.append("FAIL  class anchor not found or helper already present")

# --- Streaming path: sanitize every content delta BEFORE the parser branch ---
# Place the call right after `delta_text = output.text` so it covers both the
# parser (parse_delta) and plain (DeltaMessage) paths. With a reasoning/tool
# parser active, the previous patch's else-branch-only insert never fired.
a_stream = "                    delta_text = output.text"
r_stream = (
    "                    delta_text = output.text\n"
    "                    delta_text = _strip_spurious_artifacts(delta_text)"
)
if a_stream in content:
    content = content.replace(a_stream, r_stream, 1)
    results.append("OK    streaming delta sanitized (parser + plain)")
else:
    results.append("FAIL  streaming anchor not found")

# --- Non-streaming path: sanitize final content AFTER the parser if/else ---
# Covers both branches (parser.parse and plain output.text) before any
# ChatMessage is built, regardless of tool-choice path.
a_plain = "                suppress_metadata = False"
r_plain = (
    "                suppress_metadata = False\n"
    "\n"
    "            # PATCH(ds4-skip): strip Skip-suffix artifacts from content on\n"
    "            # BOTH parser and non-parser paths (with a reasoning/tool parser\n"
    "            # active the branches above bypass the sanitizer).\n"
    "            if isinstance(content, str):\n"
    "                content = _strip_spurious_artifacts(content)"
)
if a_plain in content:
    content = content.replace(a_plain, r_plain, 1)
    results.append("OK    non-streaming content sanitized (parser + plain)")
else:
    results.append("FAIL  non-streaming anchor not found")

SRV.write_text(content)
try:
    import py_compile

    py_compile.compile(str(SRV), doraise=True)
    results.append("OK    compile")
except Exception as e:  # pragma: no cover
    results.append(f"FAIL  compile: {e}")
    for r in results:
        print(r)
    raise SystemExit(1)

for r in results:
    print(r)