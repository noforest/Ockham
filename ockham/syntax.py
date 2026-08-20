"""Program queries about a function, routed by name to the backend currently selected.

What does this function call and under which conditions, which identifiers does it
mention, which of its lines carry a construct -- every such question goes through
`candidates.backend()`, asked by name. That is what makes --backend a real experimental
variable: switching it switches the call graph the selectors walk, not merely where the
pool came from.

A name the backend cannot answer for falls through to parsing the source it was handed.
A backend is free to refuse that fallback (Joern cannot parse a detached body at
reasonable cost); the miss is then counted and the answer is empty, so one arm never
silently borrows the other's parser.

Known limit, the same in both arms: resolution is by name, so two static functions
sharing a name collapse to whichever the symbol table kept.
"""

from . import candidates as C

_misses = {}   # name -> how many queries the current backend could not answer


def n_symbol_misses():
    return sum(_misses.values())


def missed_names():
    """The names the backend could not answer for, most frequent first."""
    return sorted(_misses, key=lambda n: (-_misses[n], n))


def reset_symbol_misses():
    _misses.clear()


# What the backend calls each answer, against what ParsedSource calls it.
_FIELD = {"get_calls": "calls", "identifiers": "identifiers", "keep_lines": "keep_lines"}


def _query(what, name, source, filename):
    """Ask the backend about `name`, else parse `source`. None when neither can answer."""
    if name:
        hit = getattr(C.backend(), what)(name)
        if hit is not None:
            return hit
    parsed = C.backend().parse_source(source, filename) if source else None
    if parsed is None:
        key = name or "<unnamed>"
        _misses[key] = _misses.get(key, 0) + 1
        return None
    return getattr(parsed, _FIELD[what])


def calls(name, source=None, filename=None):
    """[CallSite] for the body, each with its enclosing conditions outermost first."""
    hit = _query("get_calls", name, source, filename)
    return list(hit) if hit is not None else []


def called_names(name, source=None, filename=None):
    """The names this body calls."""
    return frozenset(c.name for c in calls(name, source, filename))


def identifiers(name, source=None, filename=None):
    """Every identifier in the body: variables, fields, types, callees."""
    hit = _query("identifiers", name, source, filename)
    return hit if hit is not None else frozenset()


def keep_lines(name, source=None, filename=None):
    """Rows carrying a construct worth keeping, 0-indexed from the start of the body."""
    hit = _query("keep_lines", name, source, filename)
    return hit if hit is not None else frozenset()
