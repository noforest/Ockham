"""C0: target function only, the lower bound.

No added context. Returns an empty evidence list, so the representation renders the
target alone. Needs no candidate pool, and therefore no checkout.
"""


def select(target, candidates, budget):
    return []
