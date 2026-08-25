"""S2: dense retrieval, the pool ordered by cosine similarity to the target.

The query is the target source only, never its cve or cwe. The model and its limits
live in embeddings.py; rank() is exposed apart from select() for the same reason as in
s1_bm25.
"""

from .. import candidates as C
from .. import embeddings


def rank(target, candidates):
    """The whole pool ordered by cosine similarity to the target source, best first."""
    if not candidates:
        return []
    matrix = embeddings.pool_matrix(candidates)
    query = embeddings.encode([target.func_body])[0]
    sims = matrix @ query          # both sides are L2-normalised, so this is the cosine
    order = sorted(range(len(candidates)),
                   key=lambda i: (-float(sims[i]), candidates[i].name))
    return [candidates[i] for i in order]


def select(target, candidates, budget):
    return C.enforce_budget(rank(target, candidates), budget)
