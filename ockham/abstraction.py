"""The vocabulary every backend builds.

Both backends answer the same questions about a function, so the shapes they answer
with live here rather than in either one of them.
"""

from dataclasses import dataclass, field


@dataclass
class CallSite:
    """One call site, with every control-structure condition enclosing it."""
    name: str
    conditions: list = field(default_factory=list)   # outermost first
    args: str = ""


@dataclass
class ParsedSource:
    """What a backend extracts from a function body given as a string.

    Returned by `backend.parse_source()`, the path taken when a name is absent from the
    symbol table. A backend that cannot parse a detached body returns None instead, so
    the miss is counted rather than answered by another parser.
    """
    calls: list = field(default_factory=list)
    identifiers: frozenset = frozenset()
    keep_lines: frozenset = frozenset()   # rows 0-indexed from the start of the body
