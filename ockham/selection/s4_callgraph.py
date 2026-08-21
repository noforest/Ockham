"""S4: call-graph neighbourhood (draft: "Call-graph neighborhood").

Keep the functions structurally linked to the target: what it calls, and later what
calls it. Adjacency only, with no notion of which edge matters -- that is what will
separate it from S5.
"""

from .. import candidates as C
from .. import syntax


def _callees(name, source, filename, by_name):
    """Pool functions this body calls. Library calls are not in the pool and drop out."""
    return sorted(n for n in syntax.called_names(name, source, filename) if n in by_name)


def select(target, candidates, budget):
    by_name = {c.name: c for c in candidates}
    target_name = C.guess_function_name(target.func_body)
    if target_name is None:
        return []
    reached = _callees(target_name, target.func_body, target.file_name, by_name)
    return C.enforce_budget([by_name[n] for n in reached], budget)
