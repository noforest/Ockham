"""C2: random functions at equal budget, the negative control."""

import random

from .. import candidates as C

SEED = 0


def select(target, candidates, budget):
    # Seeded per target, so the draw does not depend on the order samples are visited in.
    rng = random.Random(f"{SEED}-{target.sample_id}")
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return C.enforce_budget(shuffled, budget)
