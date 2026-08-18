"""The detection call: one pack in, one verdict out.

The prompt forces the verdict as the first word of the reply, so reading it back is a
parse of that word rather than of free prose.
"""

from openai import APIError, OpenAI

# The balanced prior is stated on purpose: without it a small model answers VULNERABLE
# to everything. It is true of this arm only, and has to be dropped on an imbalanced set.
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
    """1 vulnerable, 0 safe, -1 unreadable, from the first word of the reply.

    An unreadable reply is never coerced to SAFE. It is a distinct outcome, counted on
    its own; folded into SAFE it would improve or damage a cell's score depending on
    the labels it happened to land on.
    """
    head = text.lstrip()[:12].strip().upper()
    if head.startswith("V"):
        return 1
    if head.startswith("S"):
        return 0
    return -1


def predict(pack_text, model, base_url, api_key=None, max_tokens=16, seed=None):
    """Returns (prediction in {1, 0, -1}, raw reply).

    Only the first word is read, so the reply is capped short: a justification we never
    look at is most of the generation time, and on a thinking model it consumes the whole
    budget before any verdict is emitted.

    `seed` is forwarded when the endpoint accepts one; with temperature 0 it is what
    keeps decoding constant across the cells of a phase. Endpoints that ignore the
    field simply drop it.
    """
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
