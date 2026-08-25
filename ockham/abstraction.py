"""The shapes both backends answer with, kept here rather than in either one of them."""

from dataclasses import dataclass, field


@dataclass
class CallSite:
    """One call site, with every control-structure condition enclosing it."""
    name: str
    conditions: list = field(default_factory=list)   # outermost first
    args: str = ""


@dataclass
class ParsedSource:
    """What a backend extracts from a body given as a string, when the name is unknown."""
    calls: list = field(default_factory=list)
    identifiers: frozenset = frozenset()
    keep_lines: frozenset = frozenset()   # rows 0-indexed from the start of the body
