"""S5: the pool ranked by the target's dependence on it, rather than by adjacency (S4).

Ranking order: a primitive operation on the path, then hops from the target, then how many
identifiers the candidate shares with it, then name. The first two criteria read the call
graph and so change with the selected backend; the third does not, and stays identifier
overlap rather than a real use-def chain, which is the honest limit of this selector.
"""

from .. import candidates as C
from .. import syntax

DEPTH = 3   # same propagation depth as the primitive-API walk


def _callee_depths(target_name, target_source, target_file, by_name):
    """{callee name: hops from the target}, following calls up to DEPTH, shallowest wins."""
    depth = {}
    frontier = [(target_name, target_source, target_file)]
    for hop in range(1, DEPTH + 1):
        nxt = []
        for name, source, filename in frontier:
            for n in sorted(syntax.called_names(name, source, filename)):
                if n in by_name and n not in depth:
                    depth[n] = hop
                    nxt.append((n, by_name[n].source, by_name[n].file))
        frontier = nxt
    return depth


def _reaches_primitive(name, by_name, budget, seen, memo):
    """Is a primitive operation reachable from this function within `budget` local hops?

    Memoised on (name, budget): an answer found with two hops left says nothing about the
    same function with one hop left.
    """
    if budget == 0 or name in seen or name not in by_name:
        return False
    if (name, budget) in memo:
        return memo[(name, budget)]
    c = by_name[name]
    called = syntax.called_names(c.name, c.source, c.file)
    hit = bool(called & syntax.PRIMITIVE_APIS) or any(
        _reaches_primitive(n, by_name, budget - 1, seen | {name}, memo) for n in sorted(called))
    memo[(name, budget)] = hit
    return hit


def select(target, candidates, budget):
    by_name = {c.name: c for c in candidates}
    target_name = C.guess_function_name(target.func_body)
    depth = _callee_depths(target_name, target.func_body, target.file_name, by_name)
    if not depth:
        return []

    target_ids = syntax.identifiers(target_name, target.func_body, target.file_name)
    memo = {}
    ordered = sorted(
        (by_name[n] for n in depth),
        key=lambda c: (
            0 if _reaches_primitive(c.name, by_name, DEPTH, frozenset(), memo) else 1,
            depth[c.name],
            -len(syntax.identifiers(c.name, c.source, c.file) & target_ids),
            c.name,
        ),
    )
    return C.enforce_budget(ordered, budget)
