"""The frozen sample set: drawn once, reused identically by every cell of a phase.

A seeded draw is not enough, since it changes silently with the requested size or the
pair file, so the ids go to disk with the seed and a hash of that file.
"""

import hashlib
import json
from pathlib import Path

from .data import stratified_subsample

FORMAT = 1


def draw(samples, n_pairs, seed=0):
    """Stratified draw by project and primary cwe, at a fixed seed."""
    return stratified_subsample(samples, n_pairs=n_pairs, seed=seed)


def sample_set_id(sample_ids):
    """Short stable hash of the set, recorded in every row."""
    joined = "\n".join(sorted(sample_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def dataset_hash(path):
    """Hash of the pair file, so a changed dataset is detected rather than absorbed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save(path, sample_ids, seed, dataset_path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "sample_set_id": sample_set_id(sample_ids),
        "seed": seed,
        "dataset": str(dataset_path),
        "dataset_hash": dataset_hash(dataset_path),
        "n_samples": len(sample_ids),
        "sample_ids": list(sample_ids),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[samples] froze {len(sample_ids)} samples "
          f"(set {payload['sample_set_id']}) -> {path}", flush=True)
    return payload


def load(path, dataset_path=None):
    """Load a frozen set. Raises if the pair file changed since it was frozen."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if dataset_path is not None:
        current = dataset_hash(dataset_path)
        if current != payload.get("dataset_hash"):
            raise ValueError(
                f"pair file changed since the set was frozen "
                f"({payload.get('dataset_hash')} -> {current}). The cells would no longer "
                f"be comparable; re-freeze the set or restore the original file."
            )
    print(f"[samples] loaded set {payload['sample_set_id']} "
          f"({payload['n_samples']} samples) from {path}", flush=True)
    return payload


def apply(samples, sample_ids):
    """Keep only the frozen ids, in the frozen order. Raises if any is missing."""
    by_id = {s.sample_id: s for s in samples}
    missing = [i for i in sample_ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} frozen sample ids absent from the pair file, "
                         f"first few: {missing[:5]}")
    return [by_id[i] for i in sample_ids]
