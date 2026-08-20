"""Strip the spurious ')Skip' artifact from DeepSeek-V4 chat responses.

The DSPark speculative decoder occasionally injects the standalone vocab token
')Skip' (id 83480) mid-reply (a corrupt draft token that slips past rejection
sampling).  The artifact is always the same literal string ')Skip' and is never
legitimate prose, so we remove it from client-visible content at the OpenAI chat
serving layer.  Applies to BOTH streaming (per-chunk deltas) and non-streaming
(final message content).  The reasoning field is untouched.

This is a repo-convention patch script: run as a Docker build step
(COPY + RUN python3) after vLLM is installed.
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

# --- Inject the sanitizer helper after the module docstring / before class ---
helper = (
    "\n\n"
    "_SPURIOUS_ARTIFACTS = (')Skip',)\n"
    "\n"
    "def _strip_spurious_artifacts(text):\n"
    "    # PATCH(ds4-skip): remove the DSPark draft artifact ')Skip' that is\n"
    "    # occasionally accepted mid-reply.  Collapses any surrounding\n"
    "    # whitespace left behind (e.g. 'pages)Skip next' -> 'pages next').\n"
    "    if not text or text == '':\n"
    "        return text\n"
    "    stripped = text\n"
    "    for art in _SPURIOUS_ARTIFACTS:\n"
    "        stripped = stripped.replace(art, '')\n"
    "    # Collapse double spaces / space-before-punctuation that the removal\n"
    "    # can leave behind, but keep it conservative (only 2+ spaces).\n"
    "    import re as _re\n"
    "    stripped = _re.sub(r' {2,}', ' ', stripped)\n"
    "    return stripped\n"
    "\n"
    "\n"
    "class OpenAIServingChat"
)

a_cls = "\n\n\nclass OpenAIServingChat"
if a_cls in content and "def _strip_spurious_artifacts" not in content:
    content = content.replace(a_cls, helper, 1)
    results.append("OK    sanitizer helper added before OpenAIServingChat")
else:
    results.append("FAIL  class anchor not found or helper already present")

# --- Streaming path: sanitize each content delta ---
# 'delta_message = DeltaMessage(content=delta_text)'
a_stream = "                        delta_message = DeltaMessage(content=delta_text)"
r_stream = (
    "                        delta_text = _strip_spurious_artifacts(delta_text)\n"
    "                        delta_message = DeltaMessage(content=delta_text)"
)
if a_stream in content:
    content = content.replace(a_stream, r_stream, 1)
    results.append("OK    streaming delta sanitized")
else:
    results.append("FAIL  streaming anchor not found (delta_message = DeltaMessage(content=delta_text))")

# --- Non-streaming path: sanitize final content ---
# The parser branch sets content from parser.parse(); the plain branch sets
# content = output.text.  Sanitize right after both feed `content`, at the
# point where `message = ChatMessage(role=role, reasoning=reasoning, content=content)`.
a_plain = "                content = output.text"
r_plain = (
    "                content = output.text\n"
    "                content = _strip_spurious_artifacts(content)"
)
if a_plain in content:
    content = content.replace(a_plain, r_plain, 1)
    results.append("OK    non-streaming plain content sanitized")
else:
    results.append("FAIL  non-streaming anchor not found (content = output.text)")

# Parser branch: content comes from parser.parse() -- sanitize it too.
a_parser = "                    content = output.text"
# (the parser branch uses parser.parse output.text arg, not assignment; find the
#  assignment of content after parser.parse in full_generator is line 893 content=... )
# Sanitize at the message build to cover both branches in full_generator:
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
