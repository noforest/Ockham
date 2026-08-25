"""The sentence embedding model behind the dense selector, loaded once per process.

Embedding a whole repository pool is the expensive part of S2, so the vectors are kept
across samples, keyed per candidate rather than per pool: two samples of one project sit
at two commits, so their pools rarely match whole but share almost every function.

Two limits of the model are worth recording next to any S2 result: it is general-purpose
rather than code-trained, and it truncates at 256 word pieces, so long bodies are
compared on their opening lines only.
"""

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CACHED_VECTORS = 200_000   # ~300 MB at 384 dimensions, several repositories' worth

_model = None
_vectors = {}   # (name, file, source length) -> one embedding row


def model():
    """Load the model on first use, so a cell that never runs S2 never pays for it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts):
    """L2-normalised embeddings, one row per text, so a dot product is a cosine."""
    return model().encode(list(texts), batch_size=64, show_progress_bar=False,
                          normalize_embeddings=True, convert_to_numpy=True)


def _key(c):
    return (c.name, c.file, len(c.source))


def pool_matrix(candidates):
    """Embeddings of the candidate sources, one row per candidate, in pool order."""
    missing = [c for c in candidates if _key(c) not in _vectors]
    if missing:
        if len(_vectors) + len(missing) > MAX_CACHED_VECTORS:
            _vectors.clear()
        for c, row in zip(missing, encode([c.source for c in missing])):
            _vectors[_key(c)] = row
    return np.vstack([_vectors[_key(c)] for c in candidates])
