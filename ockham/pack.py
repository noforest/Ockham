"""The atomic unit the selection stage ranks and the budget counts."""

from dataclasses import dataclass


@dataclass
class Candidate:
    """One repository function offered to the selection stage."""
    name: str
    file: str                 # repo-relative
    line: int                 # 1-indexed definition line
    source: str
    kind: str = "function"
    tokens: int = 0
