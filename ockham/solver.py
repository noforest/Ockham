"""The detection call: one pack in, a verdict and its probability out, both from token 0."""

import math
import time

from openai import APIError, OpenAI

# The only prompt: one-shot, argued verdict, "VERDICT: <word>" as the last line. Where each
# clause comes from (papers/ in DEPOT_Contextpack-vuln):
#   "You are a security reviewer" and the fixed last line with nothing after it -- Chen et al.,
#     LLM4FPM, arXiv:2411.03079, Fig. 7: "As an expert in C/C++ code review [...] I should
#     conclude with '@@@ real bug @@@' [...] and include no other output." Form only.
#   "Reason step by step" -- Wei et al., NeurIPS 2022, via LLM4FPM s.IV.B.2.
#   "not necessarily related to it" -- irrelevant context distracts: Parasaram et al.,
#     arXiv:2404.05520; Jia et al., Compressing Code Context for LLM-based Issue Resolution;
#     Shi et al., LongCodeZip.
#   No source: "any kind of defect counts" (CWE-agnostic, where LLM4FPM and PacVD both tailor
#     the prompt per CWE), weighing both verdicts instead of defaulting to one, and the 2560
#     cap around 2000 announced (_tail_parse needs the verdict line; 2048 lost it 31 % of
#     the time).
_V5 = (                                         # 154 tokens (cl100k_base)
    "You are a security reviewer. You are shown one TARGET FUNCTION to judge, and "
    "optionally a CONTEXT section of other functions from the same repository, retrieved "
    "automatically and not necessarily related to it.\n"
    "\n"
    "Reason step by step, in as much detail as the code needs: what it does, what an "
    "input can make it do, and what the context settles about it. Any kind of defect "
    "counts, and no class of defect is more likely than another.\n"
    "\n"
    "Weigh both verdicts on the code you can see, then give the one it supports. Use up "
    "to 2000 tokens for the analysis, then answer: a reply cut off before its verdict is "
    "discarded. The last line must be exactly:\n"
    "VERDICT: VULNERABLE\n"
    "or\n"
    "VERDICT: SAFE"
)

SYSTEM_PROMPTS = {"v5": _V5}
SYSTEM_PROMPT = _V5          # kept as a name for prompt_tokens_total


# The whole completion, reasoning included: 2000 announced for the analysis, the rest is the
# verdict line's margin (at 2048 the model overran on 31 % of replies and lost it).
DEFAULT_MAX_TOKENS = 2560
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


def _tail_parse(text, truncated=False):
    """v2/v3/v4: the last VERDICT: line; failing that the last verdict word, unless the
    reply was cut off at max_tokens -- there the word is mid-analysis, not a conclusion."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip().strip("*#` ")     # a model that bolds the line still parses
        if line.upper().startswith("VERDICT"):
            return _verdict_of(line.split(":", 1)[-1])
    return -1 if truncated else _verdict_of(text)


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
    # Token 0 is analysis, not the verdict, so its logprobs would describe nothing.
    logprobs = False
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
    return _tail_parse(raw, choice.finish_reason == "length"), None, raw, billed


def _usage(response):
    """Billed prompt/completion tokens, plus the reasoning share when the API reports it."""
    u = getattr(response, "usage", None)
    if u is None:
        return None
    details = getattr(u, "completion_tokens_details", None)
    return {"prompt": getattr(u, "prompt_tokens", None),
            "completion": getattr(u, "completion_tokens", None),
            "reasoning": getattr(details, "reasoning_tokens", None) if details else None}
