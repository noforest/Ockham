"""C2: random context at equal budget, the negative control.

Functions drawn at random from the repository pool, filling the same budget as the
method under comparison. It separates "the selector picks relevant evidence" from
"the model does better with more code in front of it".
"""

import random

from .. import candidates as C

SEED = 0

_rng = random.Random(SEED)


def select(target, candidates, budget):
    shuffled = list(candidates)
    _rng.shuffle(shuffled)
    return C.enforce_budget(shuffled, budget)
