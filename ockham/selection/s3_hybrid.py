"""S3: hybrid retrieval, the lexical and dense rankings fused by Reciprocal Rank Fusion.

RRF combines positions, not scores, so nothing has to be calibrated between two scales
that are not comparable: BM25 is unbounded, the cosine lives in [-1, 1].
"""

from .. import candidates as C
from . import s1_bm25, s2_dense

# RRF damping constant: rank 1 contributes 1/61, rank 2 1/62, so the head of each list
# dominates while the tail still nudges the order.
K = 60


def select(target, candidates, budget):
    score = {}
    for ranking in (s1_bm25.select(target, candidates, budget),
                    s2_dense.select(target, candidates, budget)):
        for position, c in enumerate(ranking, 1):
            score[c.name] = score.get(c.name, 0.0) + 1.0 / (K + position)
    fused = sorted(candidates, key=lambda c: (-score.get(c.name, 0.0), c.name))
    return C.enforce_budget(fused, budget)
