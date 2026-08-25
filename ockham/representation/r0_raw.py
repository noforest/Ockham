"""R0: selected functions inserted verbatim, the control the others are measured against."""


def render(target_source, evidence):
    parts = ["=== TARGET FUNCTION ===", target_source.strip()]
    if evidence:
        parts.append("\n=== CONTEXT (selected functions, raw source) ===")
        for i, c in enumerate(evidence, 1):
            parts.append(_evidence_block(i, c))
    return "\n".join(parts)


def _evidence_block(i, c):
    header = f"[EVIDENCE E{i}]\nfile: {c.file}\nfunction: {c.name}\nline: {c.line}"
    return f"{header}\n\n{c.source.strip()}\n[/EVIDENCE]"
