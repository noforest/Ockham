"""Context selection stage: select(target, candidates, budget) -> list[Candidate].

Each selector ranks the shared pool and cuts it to the budget with enforce_budget. It
reads the target source only: its cve and cwe are the label, and would leak it.
"""
