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

# --- Streaming path: sanitize each content delta ---
a_stream = "                        delta_message = DeltaMessage(content=delta_text)"
r_stream = (
    "                        delta_text = _strip_spurious_artifacts(delta_text)\n"
    "                        delta_message = DeltaMessage(content=delta_text)"
)
if a_stream in content:
    content = content.replace(a_stream, r_stream, 1)
    results.append("OK    streaming delta sanitized")
else:
    results.append("FAIL  streaming anchor not found")

# --- Non-streaming path: sanitize final content (both branches) ---
a_plain = "                content = output.text"
r_plain = (
    "                content = output.text\n"
    "                content = _strip_spurious_artifacts(content)"
)
if a_plain in content:
    content = content.replace(a_plain, r_plain, 1)
    results.append("OK    non-streaming plain content sanitized")
else:
    results.append("FAIL  non-streaming anchor not found")

a_msg = "                message = ChatMessage(role=role, reasoning=reasoning, content=content)"
r_msg = (
    "                content = _strip_spurious_artifacts(content)\n"
    "                message = ChatMessage(role=role, reasoning=reasoning, content=content)"
)
if a_msg in content:
    content = content.replace(a_msg, r_msg, 1)
    results.append("OK    non-streaming message content sanitized (both branches)")
else:
    results.append("FAIL  non-streaming message anchor not found")

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
