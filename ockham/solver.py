"""The detection call: one pack in, one verdict out, the verdict being the first word."""

from openai import APIError, OpenAI

# The balanced prior holds on a 50/50 set only; drop it on an imbalanced one.
SYSTEM_PROMPT = (
    "Decide whether a C/C++ function contains a vulnerability.\n"
    "The pack has a TARGET FUNCTION (the function to judge) and optionally a CONTEXT "
    "section with selected supporting evidence.\n"
    "\n"
    "About half of the functions you are shown are not vulnerable. Answer VULNERABLE "
    "only if you can point to a concrete defect in the target function; answer SAFE "
    "otherwise, including when the code merely looks risky.\n"
    "\n"
    "Answer with a single word as the VERY FIRST token of your reply: "
    "VULNERABLE or SAFE. Output that word first, with no preamble, no punctuation "
    "before it, and nothing else. Do not explain."
)


REQUEST_TIMEOUT_S = 120

_client = None
_client_base_url = None


def _get_client(base_url, api_key):
    global _client, _client_base_url
    if _client is None or _client_base_url != base_url:
        _client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                         timeout=REQUEST_TIMEOUT_S)
        _client_base_url = base_url
    return _client


def extract_verdict(text):
    """1 vulnerable, 0 safe, -1 unreadable: an unreadable reply is never coerced to SAFE."""
    head = text.lstrip()[:12].strip().upper()
    if head.startswith("V"):
        return 1
    if head.startswith("S"):
        return 0
    return -1


def predict(pack_text, model, base_url, api_key=None, max_tokens=16, seed=None):
    """(prediction in {1, 0, -1}, raw reply). Capped short: only the first word is read."""
    client = _get_client(base_url, api_key)
    kwargs = {"seed": seed} if seed is not None else {}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": pack_text}],
            temperature=0, max_tokens=max_tokens, **kwargs,
        )
    except APIError as e:
        return -1, f"[api_error] {e}"
    raw = response.choices[0].message.content or ""
    return extract_verdict(raw), raw
