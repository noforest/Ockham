"""S3: the lexical and dense rankings fused by Reciprocal Rank Fusion.

RRF combines positions, not scores, so nothing has to be calibrated between BM25 and a
cosine. It reads the full rankings: a candidate one side cuts off keeps the other's credit.
"""

from .. import candidates as C
from . import s1_bm25, s2_dense

# RRF damping constant: rank 1 contributes 1/61, so each list's head dominates.
K = 60


def select(target, candidates, budget):
    score = {}
    for ranking in (s1_bm25.rank(target, candidates), s2_dense.rank(target, candidates)):
        for position, c in enumerate(ranking, 1):
            score[c.name] = score.get(c.name, 0.0) + 1.0 / (K + position)
    fused = sorted(candidates, key=lambda c: (-score.get(c.name, 0.0), c.name))
    return C.enforce_budget(fused, budget)
