"""The detection call: one pack in, a verdict and its probability out, both from token 0."""

import math

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
_TOP_LOGPROBS = 5

_client = None
_client_base_url = None


def _get_client(base_url, api_key):
    global _client, _client_base_url
    if _client is None or _client_base_url != base_url:
        _client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                         timeout=REQUEST_TIMEOUT_S)
        _client_base_url = base_url
    return _client


def _side(token):
    """Which verdict a token starts, or None."""
    t = token.strip().upper()
    if not t:
        return None
    return "V" if t[0] == "V" else "S" if t[0] == "S" else None


def extract_verdict(first_token_logprobs):
    """(prediction, p_vulnerable) from token 0: softmax over the two verdict logprobs.

    Neither verdict among the candidates means unreadable, never SAFE. When only one
    side is there, the missing logprob is floored at the lowest candidate, a lower
    bound that still yields a usable probability for a confident one-sided reply.
    """
    lp_v = lp_s = None
    all_lps = []
    for e in first_token_logprobs:
        all_lps.append(e.logprob)
        side = _side(e.token)
        if side == "V" and (lp_v is None or e.logprob > lp_v):
            lp_v = e.logprob
        elif side == "S" and (lp_s is None or e.logprob > lp_s):
            lp_s = e.logprob
    if lp_v is None and lp_s is None:
        return -1, None
    floor = min(all_lps) if all_lps else -20.0
    lp_v = floor if lp_v is None else lp_v
    lp_s = floor if lp_s is None else lp_s
    return (1 if lp_v >= lp_s else 0), math.exp(lp_v) / (math.exp(lp_v) + math.exp(lp_s))


def _hard_parse(text):
    """Fallback when the server returns no logprobs: the first word only, no probability."""
    side = _side(text.lstrip()[:12])
    return 1 if side == "V" else 0 if side == "S" else -1


def predict(pack_text, model, base_url, api_key=None, max_tokens=8, seed=None):
    """(prediction in {1, 0, -1}, p_vulnerable or None, raw reply), from a single call."""
    client = _get_client(base_url, api_key)
    kwargs = {"seed": seed} if seed is not None else {}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": pack_text}],
            temperature=0, max_tokens=max_tokens,
            logprobs=True, top_logprobs=_TOP_LOGPROBS,
            # a router may fall back to a host that drops logprobs and still
            # answer 200; ignored by endpoints that do not route
            extra_body={"provider": {"require_parameters": True}}, **kwargs,
        )
    except APIError as e:
        return -1, None, f"[api_error] {e}"
    choice = response.choices[0]
    raw = choice.message.content or ""
    content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
    if not content:
        return _hard_parse(raw), None, raw
    prediction, p_vulnerable = extract_verdict(content[0].top_logprobs)
    return prediction, p_vulnerable, raw
