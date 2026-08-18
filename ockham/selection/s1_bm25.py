"""S1: lexical retrieval with BM25.

Rank the pool by token and identifier overlap with the target. The query is the target
source and nothing else: never its cve or cwe, which would leak the label.

rank() is exposed apart from select() because a hybrid selector fuses full rankings,
not budget-cut results: a candidate the budget drops here can still be pulled back in
by its other rank.
"""

import re

from rank_bm25 import BM25Okapi

from .. import candidates as C

_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(code):
    """Identifier-aware tokens: split on non-alphanumerics, then on camelCase humps.

    read_buffer, readBuffer and ReadBuffer have to share the tokens "read" and "buffer".
    A whitespace tokenizer scores three spellings of one concept as unrelated, which on
    C code is most of the vocabulary.
    """
    words = []
    for raw in _SPLIT.split(code):
        if raw:
            words.extend(part.lower() for part in _CAMEL.sub(" ", raw).split())
    return words


def rank(target, candidates):
    """The whole pool ordered by BM25 score against the target source, best first."""
    if not candidates:
        return []
    corpus = [tokenize(c.source) for c in candidates]
    query = tokenize(target.func_body)
    if not query or not any(corpus):
        return list(candidates)
    scores = BM25Okapi(corpus).get_scores(query)
    # Name breaks ties. Many candidates share a score exactly, and a stable sort would
    # then echo the pool order, which is the backend's traversal order.
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].name))
    return [candidates[i] for i in order]


def select(target, candidates, budget):
    return C.enforce_budget(rank(target, candidates), budget)
