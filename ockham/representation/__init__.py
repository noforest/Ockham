"""Context representation stage.

Every representation is one module exposing the same function:

    render(target_source, evidence) -> str

Two invariants each of them keeps: the target is always rendered in full, because
reducing it would change the question rather than the context; and an empty evidence
list yields a target-only pack, so C0 means the same thing whichever representation
it is crossed with.
"""
