"""The shared candidate stage: token counting, the candidate pool and the budget."""

import importlib
import json
import re
import time
from pathlib import Path

from .pack import Candidate

_FUNC_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")

BACKENDS = ("ts", "joern")

_backend_name = "ts"
_BACKEND = None
_indexed = set()      # (backend, worktree) pairs already indexed
_index_times = {}     # -> seconds spent on the first index build
_index_counts = {}    # -> symbols found; 0 means the backend failed


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
    """Index a checkout once per backend: (index_time_s, n_symbols), 0 meaning it failed."""
    key = (_backend_name, str(repo_dir))
    if key not in _indexed:
        t0 = time.time()
        n = backend().index(repo_dir)
        _indexed.add(key)
        _index_times.setdefault(key, time.time() - t0)
        _index_counts[key] = n
    return _index_times.get(key, 0.0), _index_counts.get(key, 0)


def build_candidate_pool(sample, repo_dir):
    """Every indexed function but the target, cached per worktree since only the checkout matters."""
    pool = _cached_pool(repo_dir)
    target_name = guess_function_name(sample.func_body)
    return [c for c in pool if c.name != target_name]


def _pool_cache_path(repo_dir):
    return (Path(repo_dir).parent.parent / "pool_cache" /
            f"{backend_name()}-{Path(repo_dir).name}.json")


def _cached_pool(repo_dir):
    """The whole checkout's functions, from disk when already walked."""
    path = _pool_cache_path(repo_dir)
    if path.exists():
        try:
            return [Candidate(**d) for d in json.loads(path.read_text())]
        except (json.JSONDecodeError, TypeError, OSError):
            pass          # a half-written cache is rebuilt
    pool = []
    for name, (file_path, line) in backend().symbols().items():
        source = backend().get_function(name)
        if not source:
            continue
        pool.append(Candidate(
            name=name, file=_relative(file_path, repo_dir), line=int(line) if line else 0,
            source=source, tokens=count_tokens(source),
        ))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps([c.__dict__ for c in pool]))
    tmp.replace(path)     # a killed run must not leave a truncated cache
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
    """Greedily fill the budget with candidates, in the given order."""
    kept, used = [], 0
    for c in candidates:
        if used + c.tokens > budget:
            continue
        kept.append(c)
        used += c.tokens
    return kept
