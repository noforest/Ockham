"""The detection call: one pack in, a verdict and its probability out, both from token 0."""

import math
import time

from openai import APIError, OpenAI

# The balanced prior holds on a 50/50 set only; drop it on an imbalanced one.
_V1 = (
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

# v1 buys token-0 logprobs by forbidding any reading; v2 trades p for a short analysis.
_V2 = (
    "Decide whether a C/C++ function contains a vulnerability.\n"
    "The pack has a TARGET FUNCTION (the function to judge) and optionally a CONTEXT "
    "section with selected supporting evidence: functions the target calls, or that "
    "call it.\n"
    "\n"
    "About half of the functions you are shown are not vulnerable. Use the context to "
    "decide what the target's calls actually guarantee -- whether a callee validates "
    "its arguments, bounds a length, or checks a return value. Answer VULNERABLE only "
    "if you can point to a concrete defect in the TARGET function; answer SAFE "
    "otherwise, including when the code merely looks risky.\n"
    "\n"
    "Write at most three short lines of analysis. Never exceed three lines: a reply "
    "cut off before its verdict line is discarded. Then give the verdict on its own "
    "final line, exactly:\n"
    "VERDICT: VULNERABLE\n"
    "or\n"
    "VERDICT: SAFE"
)

SYSTEM_PROMPTS = {"v1": _V1, "v2": _V2}
SYSTEM_PROMPT = _V1          # the default, kept as a name for prompt_tokens_total


DEFAULT_MAX_TOKENS = 8    # v1 reads the first word only; a reasoning model needs far more
V2_MAX_TOKENS = 640       # measured: the model overruns 256 on 9 % of replies
REQUEST_TIMEOUT_S = 120
RETRIES = 3               # a 429 is a queue, not a verdict
RETRY_BACKOFF_S = 4
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


def _tail_parse(text):
    """v2: the verdict is the last VERDICT: line, or failing that the last verdict word."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT"):
            return _verdict_of(line.split(":", 1)[-1])
    return _verdict_of(text)


def _verdict_of(text):
    """Last VULNERABLE / SAFE mentioned, so a trailing verdict wins over one in the prose."""
    up = text.upper()
    v, sf = up.rfind("VULNERABLE"), up.rfind("SAFE")
    if v == sf == -1:
        return -1
    return 1 if v > sf else 0


def predict(pack_text, model, base_url, api_key=None, max_tokens=DEFAULT_MAX_TOKENS,
            seed=None, logprobs=True, reasoning=None, prompt="v1", provider=None):
    """(prediction in {1, 0, -1}, p_vulnerable or None, raw reply, usage), one call.

    usage is what the API billed, not what tiktoken counted locally: a reasoning model
    charges tokens that never appear in the reply, so the local estimate understates it.
    """
    client = _get_client(base_url, api_key)
    kwargs = {"seed": seed} if seed is not None else {}
    extra = {}
    # Under v2 token 0 is analysis, not the verdict, so its logprobs describe nothing.
    logprobs = logprobs and prompt == "v1"
    if logprobs:
        # a router may fall back to a host that drops logprobs and still answer 200;
        # ignored by endpoints that do not route
        kwargs.update(logprobs=True, top_logprobs=_TOP_LOGPROBS)
        extra["provider"] = {"require_parameters": True}
    if reasoning is not None:
        # a thinking model spends max_tokens reasoning and returns empty content
        extra["reasoning"] = ({"enabled": False} if reasoning == "off"
                              else {"effort": reasoning})
    if provider:
        # nine hosts at three quantizations: unpinned, temperature 0 still flips ~7 % of verdicts
        extra.setdefault("provider", {}).update(order=[provider], allow_fallbacks=False)
    if extra:
        kwargs["extra_body"] = extra
    for attempt in range(RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPTS[prompt]},
                          {"role": "user", "content": pack_text}],
                temperature=0, max_tokens=max_tokens, **kwargs,
            )
            break
        except APIError as e:
            # a 429 lands on contiguous stretches, so it must not become a verdict
            retryable = getattr(e, "status_code", None) in (429, 500, 502, 503, 529)
            if not retryable or attempt == RETRIES - 1:
                return -1, None, f"[api_error] {e}", None
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    choice = response.choices[0]
    raw = choice.message.content or ""
    content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
    billed = _usage(response)
    if prompt != "v1":
        return _tail_parse(raw), None, raw, billed
    if content:
        prediction, p_vulnerable = extract_verdict(content[0].top_logprobs)
        if prediction != -1:
            return prediction, p_vulnerable, raw, billed
    # logprobs without top_logprobs: the text still holds the verdict, only p is missing
    return _hard_parse(raw), None, raw, billed


def _usage(response):
    """Billed prompt/completion tokens, plus the reasoning share when the API reports it."""
    u = getattr(response, "usage", None)
    if u is None:
        return None
    details = getattr(u, "completion_tokens_details", None)
    return {"prompt": getattr(u, "prompt_tokens", None),
            "completion": getattr(u, "completion_tokens", None),
            "reasoning": getattr(details, "reasoning_tokens", None) if details else None}
