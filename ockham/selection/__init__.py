"""Context selection stage.

Every selector is one module exposing the same function:

    select(target, candidates, budget) -> list[Candidate]

It ranks the shared candidate pool and returns an ordered subset that fits the budget
(via candidates.enforce_budget). A selector queries the target source only, never its
cve or cwe: those are the label, and reading them would leak it into the pack.
"""
