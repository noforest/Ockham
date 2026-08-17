"""The sample set a cell runs on.

Every cell of a phase must see the same functions, or a method can look better simply
because it was handed easier ones.
"""

import hashlib

from .data import stratified_subsample


def draw(samples, n_pairs, seed=0):
    """Stratified draw by project and primary cwe, at a fixed seed."""
    return stratified_subsample(samples, n_pairs=n_pairs, seed=seed)


def sample_set_id(sample_ids):
    """Short stable hash of the set, recorded in every row so cells can be checked later."""
    joined = "\n".join(sorted(sample_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
