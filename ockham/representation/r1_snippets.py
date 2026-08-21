"""R1: syntax-preserving code snippets (draft: "Syntax-preserving code snippets").

Each selected function is reduced to the lines that carry meaning for the analysis: its
signature, what it declares, what guards it, what it calls, what it returns. Original
line numbers are kept so the model can still name a location. Examples from the draft:
LLMxCPG, LongCodeZip.

The target is never reduced: it is what the model has to judge.

Limit to state: the draft specifies R1 at statement grain, and this works at function
grain, because the shared pool offers functions. R1 reduces INSIDE each selected
function; it cannot keep half a function that selection did not choose.
"""

from .. import syntax


def render(target_source, evidence):
    parts = ["=== TARGET FUNCTION ===", target_source.strip()]
    if evidence:
        parts.append("\n=== CONTEXT (selected functions, reduced to their relevant lines) ===")
        parts.append("Original line numbers are preserved.")
        for i, c in enumerate(evidence, 1):
            parts.append(_snippet_block(i, c))
    return "\n".join(parts)


def _snippet_block(i, c):
    header = f"[EVIDENCE E{i}]\nfile: {c.file}\nfunction: {c.name}\nline: {c.line}"
    return f"{header}\n\n{_snippet(c)}\n[/EVIDENCE]"


def _kept_rows(c, n_lines):
    """Rows worth keeping: the signature, the closing line, and the head of every construct.

    Rows are offsets into the body the backend was given, whichever backend answered, so
    they index c.source directly. An empty answer means it could not read this body, and
    _snippet then falls back to verbatim.
    """
    rows = syntax.keep_lines(c.name, c.source, c.file)
    if not rows:
        return set()
    return {0, n_lines - 1} | set(rows)


def _snippet(c):
    lines = c.source.splitlines()
    rows = _kept_rows(c, len(lines))
    if not rows:
        return c.source.strip()      # unreadable body: better verbatim than empty
    out = []
    for row in sorted(rows):
        if not 0 <= row < len(lines):
            continue
        out.append(f"{c.line + row:6d}| {lines[row].rstrip()}")
    return "\n".join(out)
