"""The detection call: one pack in; the verdict, its probability and the CWE out."""

import math
import re
import time

from openai import APIError, OpenAI

REQUEST_TIMEOUT_S = 120
RETRIES = 3               # a 429 is a queue, not a verdict
RETRY_BACKOFF_S = 4
_TOP_LOGPROBS = 5
_MARKUP = " *`#\n"

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
    """(prediction, p_vulnerable) from one token's top logprobs; no verdict there is unreadable."""
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


_VERDICTS = {"VULNERABLE": 1, "SAFE": 0}


def _verdict_line(text, position="last"):
    """The first or last VERDICT: line, read strictly: VULNERABLE or SAFE, anything else -1."""
    # markdown stripped, never words searched: "not vulnerable" must not read as VULNERABLE
    lines = [re.sub(r"[*`#_]", "", line).strip() for line in text.strip().splitlines()]
    for line in (lines if position == "first" else reversed(lines)):
        m = re.match(r"VERDICT\s*:(.*)", line, re.IGNORECASE)
        if m:
            return _VERDICTS.get(m.group(1).strip(" .").upper(), -1)
    return -1


def parse_cwe(text):
    """CWE-<n> or UNKNOWN from the CWE: line; None when the reply has no such line."""
    for line in text.splitlines():
        line = line.strip().strip("*#` ")
        if line.upper().startswith("CWE:"):
            m = re.search(r"\d+", line.split(":", 1)[1])
            return f"CWE-{m.group(0)}" if m else "UNKNOWN"
    return None


def _verdict_token(content, position="first"):
    """Top logprobs of the first real token after the first or last VERDICT: marker."""
    seen, starts = "", []
    for i, tok in enumerate(content):
        seen += tok.token
        if seen.rstrip(_MARKUP).upper().endswith("VERDICT:"):
            starts.append(i + 1)
    if not starts:
        return None
    for tok in content[starts[0] if position == "first" else starts[-1]:]:
        if tok.token.strip(_MARKUP):
            return tok.top_logprobs
    return None


def predict(pack_text, prompt, model, base_url, max_tokens, api_key=None, seed=None,
            logprobs=True, reasoning=None, provider=None):
    """One call: {prediction, p_vulnerable, raw, billed, finish_reason, cwe_pred}."""
    client = _get_client(base_url, api_key)
    kwargs = {"seed": seed} if seed is not None else {}
    extra = {}
    if logprobs:
        kwargs.update(logprobs=True, top_logprobs=_TOP_LOGPROBS)
        if not provider:
            # a router may fall back to a host that drops logprobs and still answer 200
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
                messages=[{"role": "system", "content": prompt.text},
                          {"role": "user", "content": pack_text}],
                temperature=0, max_tokens=max_tokens, **kwargs,
            )
            break
        except APIError as e:
            # a 429 lands on contiguous stretches, so it must not become a verdict
            retryable = getattr(e, "status_code", None) in (429, 500, 502, 503, 529)
            if not retryable or attempt == RETRIES - 1:
                return {"prediction": -1, "p_vulnerable": None, "raw": f"[api_error] {e}",
                        "billed": None, "finish_reason": "error", "cwe_pred": None}
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    choice = response.choices[0]
    raw = choice.message.content or ""
    content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
    top = _verdict_token(content, prompt.verdict) if content else None
    return {"prediction": _verdict_line(raw, prompt.verdict),
            "p_vulnerable": extract_verdict(top)[1] if top else None,
            "raw": raw, "billed": _usage(response), "finish_reason": choice.finish_reason,
            "cwe_pred": parse_cwe(raw) if prompt.ask_cwe else None}


def _usage(response):
    """Billed prompt/completion tokens, plus the reasoning share when the API reports it."""
    u = getattr(response, "usage", None)
    if u is None:
        return None
    details = getattr(u, "completion_tokens_details", None)
    return {"prompt": getattr(u, "prompt_tokens", None),
            "completion": getattr(u, "completion_tokens", None),
            "reasoning": getattr(details, "reasoning_tokens", None) if details else None}
