"""R1: each selected function reduced to the head line of what it declares, branches,
calls and returns, keeping the original numbers. The target itself is never reduced."""

from .. import syntax


def render(target_source, evidence):
    parts = ["=== TARGET FUNCTION ===", target_source.strip()]
    if evidence:
        parts.append("\n=== CONTEXT (selected functions, reduced to their relevant lines) ===")
        parts.append("Original line numbers are preserved; '...' marks elided lines.")
        for i, c in enumerate(evidence, 1):
            parts.append(_snippet_block(i, c))
    return "\n".join(parts)


def _snippet_block(i, c):
    header = f"[EVIDENCE E{i}]\nfile: {c.file}\nfunction: {c.name}\nline: {c.line}"
    return f"{header}\n\n{_snippet(c)}\n[/EVIDENCE]"


def _kept_rows(c, n_lines):
    """Body-relative rows to keep: signature, closing line, construct heads."""
    rows = syntax.keep_lines(c.name, c.source, c.file)
    if not rows:
        return set()
    return {0, n_lines - 1} | set(rows)


def _snippet(c):
    lines = c.source.splitlines()
    rows = _kept_rows(c, len(lines))
    if not rows:
        return c.source.strip()      # unreadable body: better verbatim than empty
    out, previous = [], None
    for row in sorted(rows):
        if not 0 <= row < len(lines):
            continue
        if previous is not None and row > previous + 1:
            out.append("      ...")
        out.append(f"{c.line + row:6d}| {lines[row].rstrip()}")
        previous = row
    return "\n".join(out)
