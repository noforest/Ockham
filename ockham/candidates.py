"""The shared candidate stage: token counting, the candidate pool and the budget.

Every selector ranks the same pool under the same token budget, so a difference in
results is attributable to the selector and not to the pool it drew from.

The backend is an experimental variable, not a fidelity choice. Selecting one swaps
the whole program-query layer -- symbols, call edges, enclosing conditions -- so a
cell can be run twice and the gap between the two measured.
"""

import importlib
import re
import time
from pathlib import Path

from .pack import Candidate

_FUNC_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")

BACKENDS = ("ts", "joern")

_backend_name = "ts"
_BACKEND = None
_indexed = set()      # (backend, worktree) pairs already indexed
_index_times = {}     # (backend, worktree) -> cost of its first index build, in seconds
_index_counts = {}    # (backend, worktree) -> symbols found; 0 means the backend failed


def set_backend(which):
    """Select the backend every later query goes through."""
    global _backend_name, _BACKEND
    if which not in BACKENDS:
        raise ValueError(f"unknown backend {which!r}, expected one of {list(BACKENDS)}")
    _backend_name = which
    _BACKEND = None
    _indexed.clear()
    return backend()


def backend():
    """The backend module currently selected."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = importlib.import_module(f".backend_{_backend_name}", __package__)
    return _BACKEND


def backend_name():
    return _backend_name


def count_tokens(text):
    """cl100k_base token count, with a word fallback."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def guess_function_name(body):
    """Identifier immediately before the first '(' in the signature."""
    header = body.split("{", 1)[0]
    m = _FUNC_NAME_RE.search(header)
    return m.group(1) if m else None


def ensure_indexed(repo_dir):
    """Index a checkout unless the current backend has already indexed it.

    Returns (index_time_s, n_symbols). A count of 0 means the backend failed on this
    repository -- a parser timeout is a result to record, not an empty repository.
    Only the first build of a worktree is timed, so a later re-index does not inflate
    the reported cost.
    """
    key = (_backend_name, str(repo_dir))
    if key not in _indexed:
        t0 = time.time()
        n = backend().index(repo_dir)
        _indexed.add(key)
        _index_times.setdefault(key, time.time() - t0)
        _index_counts[key] = n
    return _index_times.get(key, 0.0), _index_counts.get(key, 0)


def build_candidate_pool(sample, repo_dir):
    """Enumerate the indexed functions into a fixed pool. Call ensure_indexed first.

    Everything except the target itself, each candidate carrying its source and token
    count so budget enforcement never has to go back to the backend. The pool comes
    from this sample's own commit, so the patched sibling is not in it.
    """
    target_name = guess_function_name(sample.func_body)
    pool = []
    for name, (path, line) in backend().symbols().items():
        if name == target_name:
            continue
        source = backend().get_function(name)
        if not source:
            continue
        pool.append(Candidate(
            name=name, file=_relative(path, repo_dir), line=int(line) if line else 0,
            source=source, tokens=count_tokens(source),
        ))
    return pool


def _relative(path, repo_dir):
    """Repo-relative path. A backend may report either an absolute or a relative one."""
    try:
        return str(Path(path).resolve().relative_to(Path(repo_dir).resolve()))
    except (ValueError, OSError):
        return str(path)


def target_line(sample):
    """The target function's definition line in the indexed repo, or None."""
    name = guess_function_name(sample.func_body)
    entry = backend().symbols().get(name)
    return int(entry[1]) if entry and entry[1] else None


def enforce_budget(candidates, budget):
    """Greedily fill the budget with candidates, in the given order.

    A fill, not a prefix cut: the loop goes past a candidate that does not fit, so a
    small low-ranked one can still enter after a large higher-ranked one was skipped.
    Leaving budget unspent would understate every selector. A candidate larger than
    the whole budget is skipped rather than truncated, which would change the
    representation and not just the selection.
    """
    kept, used = [], 0
    for c in candidates:
        if used + c.tokens > budget:
            continue
        kept.append(c)
        used += c.tokens
    return kept
