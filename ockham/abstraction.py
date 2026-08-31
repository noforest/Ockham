"""The shapes both backends answer with, and the walk from a function to what it reaches.

The list of primitive operations is one CWE-agnostic union applied to every sample alike:
choosing it per ground-truth weakness class would leak the label into the pack. The walk
speaks only the backend convention get_function(name) / get_calls(name), so it never
branches on which backend is selected -- if it needed to, the interface would have leaked.
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


@dataclass
class ApiInfo:
    """One primitive operation as reached from a target: how often, under what conditions."""
    count: int = 0
    branches: list = field(default_factory=list)


def _compose(outer, inner):
    """The two conditions joined by '&&', or 'unconditionally' when neither holds."""
    parts = [p for p in (outer, inner) if p and p != "unconditionally"]
    return " && ".join(parts) if parts else "unconditionally"


def _analyze(backend, fn_name, depth, visiting, memo, depth_counts):
    """{api: ApiInfo} for this function, recursing into the callees the backend resolves."""
    if depth == 0 or fn_name in visiting:
        return {}
    if fn_name in memo:
        return memo[fn_name]
    hop = DEPTH - depth + 1

    result = {}
    for call in backend.get_calls(fn_name):
        cond = " && ".join(call.conditions) if call.conditions else "unconditionally"
        if call.name in PRIMITIVE_APIS:
            info = result.setdefault(call.name, ApiInfo())
            info.count += 1
            info.branches.append(cond)
            depth_counts[hop] = depth_counts.get(hop, 0) + 1
        elif backend.get_function(call.name) is not None:
            sub = _analyze(backend, call.name, depth - 1, visiting | {fn_name}, memo,
                           depth_counts)
            for api, sub_info in sub.items():
                info = result.setdefault(api, ApiInfo())
                info.count += sub_info.count
                # The caller's own condition guards everything the callee does.
                info.branches.extend(_compose(cond, b) for b in sub_info.branches)

    memo[fn_name] = result
    return result


def primitive_api_a3(backend, fn_name):
    """({api: ApiInfo}, {hop: primitive call sites at that hop}) over DEPTH hops."""
    depth_counts = {}
    apis = _analyze(backend, fn_name, DEPTH, frozenset(), {}, depth_counts)
    return apis, depth_counts


def signature(code):
    """Everything before the first '{', whitespace collapsed, else the first line."""
    sig = " ".join(code.split("{", 1)[0].split())
    return sig or code.splitlines()[0].strip()


def render_apis(apis):
    """{api: ApiInfo} as lines, each distinct branch condition listed once with its count."""
    lines = []
    for api, info in apis.items():
        counts, order = {}, []
        for b in info.branches:
            if b not in counts:
                order.append(b)
            counts[b] = counts.get(b, 0) + 1
        lines.append(f"{api} ({info.count}x):")
        for b in order:
            suffix = f" (x{counts[b]})" if counts[b] > 1 else ""
            lines.append(f"  {b}{suffix}")
    return lines


def callee_breakdown(backend, fn_name):
    """Same walk as primitive_api_a3, but findings stay grouped under each direct callee.

    A pack rendered from one flattened dict says which operations are reached; grouped, it
    also says through which callee, which is what makes the abstraction readable.
    """
    depth_counts = {}
    direct = {}
    by_callee = {}   # callee name -> [signature, {api: ApiInfo}]
    memo = {}
    for call in backend.get_calls(fn_name):
        cond = " && ".join(call.conditions) if call.conditions else "unconditionally"
        if call.name in PRIMITIVE_APIS:
            info = direct.setdefault(call.name, ApiInfo())
            info.count += 1
            info.branches.append(cond)
            depth_counts[1] = depth_counts.get(1, 0) + 1
        elif backend.get_function(call.name) is not None:
            sub = _analyze(backend, call.name, DEPTH - 1, frozenset({fn_name}), memo,
                           depth_counts)
            if not sub:
                continue
            if call.name not in by_callee:
                by_callee[call.name] = [signature(backend.get_function(call.name)), {}]
            bucket = by_callee[call.name][1]
            for api, sub_info in sub.items():
                info = bucket.setdefault(api, ApiInfo())
                info.count += sub_info.count
                info.branches.extend(_compose(cond, b) for b in sub_info.branches)
    return direct, by_callee, depth_counts
