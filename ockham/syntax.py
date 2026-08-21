"""Program queries about a function, asked BY NAME of the selected backend.

That is what makes --backend an experimental variable: switching it switches the call
graph the selectors walk, not merely where the pool came from. A name the backend cannot
resolve falls through to parsing the source we were handed, and a backend may refuse that
fallback -- the miss is then counted, never answered by the other parser.
"""

from . import candidates as C

_misses = {}    # name -> how many queries the current backend could not answer


def n_symbol_misses():
    return sum(_misses.values())


def missed_names():
    """The names the backend could not answer for, most frequent first."""
    return sorted(_misses, key=lambda n: (-_misses[n], n))


def reset_symbol_misses():
    _misses.clear()


# Backend method -> the ParsedSource field holding the same answer.
_FIELD = {"get_calls": "calls", "identifiers": "identifiers", "keep_lines": "keep_lines"}


def _query(what, name, source, filename):
    """Ask the backend about `name`, else parse `source`. None if neither can answer."""
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
    """[CallSite], each carrying its guarding conditions outermost first."""
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
    """Rows R1 keeps, 0-indexed from the start of the body."""
    hit = _query("keep_lines", name, source, filename)
    return hit if hit is not None else frozenset()
