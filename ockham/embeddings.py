"""The sentence embedding model behind the dense selector, loaded once per process.

Embedding a whole repository pool is the expensive part of S2, so the vectors are kept
across samples. Two limits of the model are worth recording next to any S2 result: it is
general-purpose rather than code-trained, and it truncates at 256 word pieces, so long
bodies are compared on their opening lines only.
"""

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_pools = {}


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


def pool_matrix(candidates):
    """Embeddings of the candidate sources, one row per candidate, in pool order."""
    key = tuple(c.name for c in candidates)
    if key not in _pools:
        _pools[key] = encode([c.source for c in candidates])
    return _pools[key]
