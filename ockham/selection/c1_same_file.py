"""C1: same-file proximity, the heuristic baseline.

Keep the other functions of the target's own file, nearest first by line distance:
previous, next, second previous, second next, and so on. Preceding wins ties.
"""

from .. import candidates as C


def select(target, candidates, budget):
    same_file = [c for c in candidates if c.file == target.file_name]
    t_line = C.target_line(target)
    if t_line is None:
        ordered = same_file      # no resolved line: keep file order, still same-file only
    else:
        ordered = sorted(
            same_file,
            key=lambda c: (abs(c.line - t_line), 0 if c.line < t_line else 1),
        )
    return C.enforce_budget(ordered, budget)
