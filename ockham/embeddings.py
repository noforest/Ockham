"""The sentence embedding model behind S2, cached per candidate and persisted to disk."""

import atexit
import os
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CACHED_VECTORS = 200_000   # ~300 MB at 384 dimensions, several repositories' worth

CACHE = (Path(__file__).resolve().parent.parent / "workspace" / "embed_cache" /
         f"{MODEL_NAME.split('/')[-1]}.npz")

_model = None
_vectors = {}   # (name, file, source length) -> one embedding row
_loaded = False
_dirty = False


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


def _load_cache():
    """Read the vectors earlier runs wrote; a damaged file is ignored."""
    global _loaded
    _loaded = True
    if not CACHE.exists():
        return
    try:
        with np.load(CACHE, allow_pickle=False) as z:
            keys, rows = z["keys"], z["rows"]
    except (OSError, ValueError, KeyError):
        return
    for key, row in zip(keys, rows):
        name, file, length = str(key).split("\x00")
        _vectors[(name, file, int(length))] = row


def _save_cache():
    """Write the vectors back atomically if this process computed new ones."""
    if not _dirty or not _vectors:
        return
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array(["\x00".join((n, f, str(length))) for n, f, length in _vectors])
    tmp = CACHE.with_name(CACHE.name + ".partial.npz")
    np.savez(tmp, keys=keys, rows=np.vstack(list(_vectors.values())))
    os.replace(tmp, CACHE)


atexit.register(_save_cache)


def pool_matrix(candidates):
    """Embeddings of the candidate sources, one row per candidate, in pool order."""
    global _dirty
    if not _loaded:
        _load_cache()
    missing = [c for c in candidates if _key(c) not in _vectors]
    if missing:
        if len(_vectors) + len(missing) > MAX_CACHED_VECTORS:
            _vectors.clear()      # the clear drops the hits too, so nothing is cached now
            missing = list(candidates)
        for c, row in zip(missing, encode([c.source for c in missing])):
            _vectors[_key(c)] = row
        _dirty = True
    return np.vstack([_vectors[_key(c)] for c in candidates])
