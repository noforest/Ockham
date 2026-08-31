"""The shapes both backends answer with, and the fixed list of primitive operations.

The list is one CWE-agnostic union applied to every sample alike. Choosing it per
ground-truth weakness class would leak the label into the pack.
"""

from dataclasses import dataclass, field

PRIMITIVE_APIS = frozenset({
    "malloc", "calloc", "realloc", "free",
    "memcpy", "memmove", "memset", "memcmp",
    "strcpy", "strncpy", "strcat", "strncat", "strlen", "strcmp", "strncmp",
    "sprintf", "snprintf", "vsprintf", "sscanf", "scanf", "fscanf",
    "gets", "fgets", "fread", "fwrite",
    "read", "write", "open", "close", "fopen", "fclose",
    "system", "exec", "execve", "popen",
})

DEPTH = 3   # Li et al.: mean 2.8 layers of propagation, ~75% of weakness types by depth 3


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
