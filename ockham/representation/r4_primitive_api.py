"""R4: the evidence replaced by the primitive operations it performs, and under which guards."""

from .. import abstraction
from .. import candidates as C
from .. import syntax
from .r0_raw import _assemble

HEADER = "=== CONTEXT (selected functions, primitive-API abstraction) ==="


def render(target_source, evidence, target_last=False, target_file=None):
    if not evidence:
        return _assemble(target_source, [], HEADER, target_last)
    target_name = C.guess_function_name(target_source) or "target"
    backend = syntax.EvidenceBackend(target_name, target_source, evidence)
    direct, by_callee, _ = abstraction.callee_breakdown(backend, target_name)
    where = {c.name: (c.file, c.line) for c in evidence}
    return _assemble(target_source, [_body(direct, by_callee, where, len(evidence))],
                     HEADER, target_last)


def _body(direct, by_callee, where, n_evidence):
    lines = []
    if direct:
        lines.append("(called directly by the target)")
        lines.extend(abstraction.render_apis(direct))
        lines.append("")
    for name, (signature, apis) in by_callee.items():
        lines.append(f"{signature}{_where(name, where)}")
        lines.extend(abstraction.render_apis(apis))
        lines.append("")
    if not lines:
        # "nothing reachable" is an answer; an empty section reads as a failed build
        return ("No primitive-API operation is reachable from the selected evidence "
                f"({n_evidence} function(s) selected).")
    return "\n".join(lines).rstrip()


def _where(name, where):
    if name not in where:
        return ""
    return f"   ({where[name][0]}:{where[name][1]})"
