"""The shared candidate stage: token counting, the candidate pool and the budget.

Every selector ranks the same pool under the same token budget, so a difference in
results is attributable to the selector and not to the pool it drew from.
"""

import re

_FUNC_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


def count_tokens(text):
    """cl100k_base token count, with a word fallback."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def guess_function_name(body):
    """Identifier immediately before the first '(' in the signature."""
    header = body.split("{", 1)[0]
    m = _FUNC_NAME_RE.search(header)
    return m.group(1) if m else None


def enforce_budget(candidates, budget):
    """Keep candidates, in the given order, while their source tokens fit the budget.

    A candidate larger than the whole budget is skipped rather than truncated:
    truncating a body would change the representation, not just the selection.
    """
    kept, used = [], 0
    for c in candidates:
        if used + c.tokens > budget:
            break
        kept.append(c)
        used += c.tokens
    return kept
