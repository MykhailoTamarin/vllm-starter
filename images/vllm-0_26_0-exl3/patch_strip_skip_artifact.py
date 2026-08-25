"""Strip the spurious ')Skip' artifact from DeepSeek-V4 chat responses.

The DSPark speculative decoder occasionally injects the standalone vocab token
')Skip' (id 83480) mid-reply -- a corrupt draft token that slips past rejection
sampling. It is always the exact literal string ')Skip' and never legitimate
prose, so we remove just that token from client-visible content at the OpenAI
chat serving layer.

SCOPE (deliberately narrow, cf. the earlier wide patch that broke tool output):
  * Only the exact token ')Skip' is removed. Bare 'Skip' / 'skip' / 'skipped'
    (legitimate English) and the other punctuation-prefixed forms (,Skip,
    .Skip, .skip, ' Skip') are deliberately NOT touched -- the wide set was what
    corrupted agent/tool-written content like YAML.
  * Whitespace handling is limited to dropping a single space immediately next
    to the artifact (so 'text )Skip' does not leave a dangling space behind).
    There is NO global multi-space collapse (`re.sub(r" {2,}", ...)`) -- that
    global pass was the other YAML/code breaker. No `re` is used at all so the
    injected helper needs no new import inside serving.py.

Applied on BOTH paths (DeepSeek V4 runs --reasoning-parser deepseek_v4, so vLLM
serving.py always has a non-None parser): streaming per-chunk deltas and
non-streaming final message content. The reasoning field is untouched.

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
# Bare string ops only -- no `re`, so no import is required in serving.py.
HELPER = '''
def _strip_skip_artifact(text):
    # PATCH(ds4-skip): remove the DSPark corrupt-draft token ')Skip' (id 83480).
    # Deliberately narrow: only the exact literal ')Skip', plus a single space
    # immediately next to it (so 'text )Skip' -> 'text', not 'text '). Bare
    # 'Skip'/'skip', other punctuation forms, and unrelated whitespace are all
    # left untouched -- no global space collapsing.
    if not text:
        return text
    text = text.replace(" )Skip", "")
    text = text.replace(")Skip ", "")
    text = text.replace(")Skip", "")
    return text


class OpenAIServingChat'''


a_cls = "\n\n\nclass OpenAIServingChat"
if a_cls in content and "def _strip_skip_artifact" not in content:
    content = content.replace(a_cls, HELPER, 1)
    results.append("OK    sanitizer helper added before OpenAIServingChat")
else:
    results.append("FAIL  class anchor not found or helper already present")

# --- Streaming path: sanitize every content delta BEFORE the parser branch ---
# Covers both the parser (parse_delta) and plain (DeltaMessage) paths.
a_stream = "                    delta_text = output.text"
r_stream = (
    "                    delta_text = output.text\n"
    "                    delta_text = _strip_skip_artifact(delta_text)"
)
if a_stream in content:
    content = content.replace(a_stream, r_stream, 1)
    results.append("OK    streaming delta sanitized (parser + plain)")
else:
    results.append("FAIL  streaming anchor not found")

# --- Non-streaming path: sanitize final content AFTER the parser if/else ---
# Anchor at the tail of the non-parser branch; the inserted block dedents to the
# outer level so it covers BOTH branches before any ChatMessage is built.
a_plain = "                suppress_metadata = False"
r_plain = (
    "                suppress_metadata = False\n"
    "\n"
    "            # PATCH(ds4-skip): strip the ')Skip' artifact from content on\n"
    "            # BOTH parser and non-parser paths.\n"
    "            if isinstance(content, str):\n"
    "                content = _strip_skip_artifact(content)"
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
