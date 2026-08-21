"""S4: call-graph neighbourhood (draft: "Call-graph neighborhood").

Keep the functions structurally linked to the target, callers and callees, up to a
maximum distance K. Examples from the draft: VulEval, ReposVul. Adjacency only, with no
notion of which edge matters -- that is what will separate it from S5.
"""

from .. import candidates as C
from .. import syntax

MAX_DISTANCE = 2


def _callees(name, source, filename, by_name):
    """Pool functions this body calls. Library calls are not in the pool and drop out."""
    return sorted(n for n in syntax.called_names(name, source, filename) if n in by_name)


def _callers(name, candidates):
    """Pool functions whose body calls `name`.

    The substring test in front of the query is not micro-optimisation: the pool is the
    whole repository, and a body that never mentions the name cannot call it.
    """
    return sorted(c.name for c in candidates
                  if name in c.source and name in syntax.called_names(c.name, c.source, c.file))


def select(target, candidates, budget):
    by_name = {c.name: c for c in candidates}
    target_name = C.guess_function_name(target.func_body)
    if target_name is None:
        return []

    reached = {}                                        # name -> distance
    frontier = [(target_name, target.func_body, target.file_name)]
    for distance in range(1, MAX_DISTANCE + 1):
        nxt = []
        for name, source, filename in frontier:
            neighbours = (_callees(name, source, filename, by_name)
                          + _callers(name, candidates))
            for n in neighbours:
                if n in reached or n == target_name:
                    continue
                reached[n] = distance
                nxt.append((n, by_name[n].source, by_name[n].file))
        frontier = nxt

    ordered = sorted((by_name[n] for n in reached), key=lambda c: c.name)
    return C.enforce_budget(ordered, budget)
