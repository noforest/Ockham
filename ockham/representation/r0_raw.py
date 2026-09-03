"""R0: selected functions inserted verbatim, the control the others are measured against."""


def render(target_source, evidence, target_last=False, target_file=None):
    blocks = [_evidence_block(i, c) for i, c in enumerate(evidence, 1)]
    return _assemble(target_source, blocks,
                     "=== CONTEXT (selected functions, raw source) ===", target_last)


def _evidence_block(i, c):
    header = f"[EVIDENCE E{i}]\nfile: {c.file}\nfunction: {c.name}\nline: {c.line}"
    return f"{header}\n\n{c.source.strip()}\n[/EVIDENCE]"


ANCHOR = "Now judge ONLY the target function above. The context is background."


def _assemble(target_source, blocks, header, target_last):
    """Target first (the control), or context first with the target last and anchored."""
    target = ["=== TARGET FUNCTION ===", target_source.strip()]
    if not blocks:
        return "\n".join(target)
    context = [header] + blocks
    if not target_last:
        return "\n".join(target + ["\n" + context[0]] + context[1:])
    return "\n".join(context + ["\n" + target[0], target[1], "\n" + ANCHOR])
