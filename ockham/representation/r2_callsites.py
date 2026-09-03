"""R2: R0 plus the call sites tying each evidence function back to the target."""

from .. import syntax
from .r0_raw import _assemble

HEADER = "=== CONTEXT (selected functions, with how the target reaches them) ==="


def render(target_source, evidence, target_last=False, target_file=None):
    sites = _sites_by_name(target_source, target_file)
    blocks = [_block(i, c, sites.get(c.name, [])) for i, c in enumerate(evidence, 1)]
    return _assemble(target_source, blocks, HEADER, target_last)


def _sites_by_name(target_source, target_file):
    """{callee name: [guard list, ...]} for every call the target makes."""
    out = {}
    for call in syntax.calls(None, target_source, target_file):
        out.setdefault(call.name, []).append(list(call.conditions))
    return out


def _link(name, guards):
    """One line per call site saying under which conditions the target reaches it."""
    if not guards:
        return "reached-from-target: not called directly (caller, or deeper in the chain)"
    return "\n".join(
        f"reached-from-target: {name}() called when " + " && ".join(g) if g
        else f"reached-from-target: {name}() called unconditionally" for g in guards)


def _block(i, c, guards):
    header = (f"[EVIDENCE E{i}]\nfile: {c.file}\nfunction: {c.name}\nline: {c.line}\n"
              f"{_link(c.name, guards)}")
    return f"{header}\n\n{c.source.strip()}\n[/EVIDENCE]"
