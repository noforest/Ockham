"""C2: random context at equal budget, the negative control.

Functions drawn at random from the repository pool, filling the same budget as the
method under comparison. It separates "the selector picks relevant evidence" from
"the model does better with more code in front of it".
"""

import random

from .. import candidates as C

SEED = 0


def select(target, candidates, budget):
    # Seeded per target, so the draw is fixed for a given sample and independent of the
    # order the cell happens to visit samples in.
    rng = random.Random(f"{SEED}-{target.sample_id}")
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return C.enforce_budget(shuffled, budget)
