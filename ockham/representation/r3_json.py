"""R3: the selected code restated as facts, each one checkable against a file line."""

import json

from .. import candidates as C
from .. import syntax
from .r0_raw import _assemble

HEADER = "=== CONTEXT (selected functions, as structured facts) ==="


def render(target_source, evidence, target_last=False, target_file=None):
    if not evidence:
        return _assemble(target_source, [], HEADER, target_last)
    target_name = C.guess_function_name(target_source) or "target"
    facts = {
        "target": {
            "function": target_name,
            "signature": syntax.signature(target_source),
            "calls": _call_facts(target_name, target_source, target_file),
        },
        "evidence": [_evidence_facts(f"E{i}", c) for i, c in enumerate(evidence, 1)],
        "relations": _relations(target_name, target_source, target_file, evidence),
    }
    return _assemble(target_source, [json.dumps(facts, indent=2)], HEADER, target_last)


def _call_facts(name, source, filename):
    """One entry per distinct (callee, guards) pair, with how often it occurs."""
    folded = {}
    for call in syntax.calls(name, source, filename):
        key = (call.name, tuple(call.conditions))
        folded[key] = folded.get(key, 0) + 1
    facts = []
    for (callee, guards), count in folded.items():
        fact = {"callee": callee, "primitive": callee in syntax.PRIMITIVE_APIS,
                "guards": list(guards) or ["unconditionally"]}
        if count > 1:
            fact["occurrences"] = count
        facts.append(fact)
    return facts


def _evidence_facts(eid, c):
    return {
        "id": eid,
        "function": c.name,
        "file": c.file,
        "line": c.line,
        "signature": syntax.signature(c.source),
        "primitive_operations": sorted(syntax.primitives(c.name, c.source, c.file)),
        "calls": _call_facts(c.name, c.source, c.file),
    }


def _relations(target_name, target_source, target_file, evidence):
    """Call edges inside the pack only: an edge to something unselected is not checkable."""
    names = {c.name for c in evidence}
    edges = [{"from": target_name, "calls": n}
             for n in sorted(syntax.called_names(target_name, target_source, target_file)
                             & names)]
    for c in evidence:
        edges.extend({"from": c.name, "calls": n}
                     for n in sorted(syntax.called_names(c.name, c.source, c.file) & names)
                     if n != c.name)
    return edges
