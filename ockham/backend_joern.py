"""CPG backend: one joern-parse plus one dump.sc per checkout, same convention as backend_ts."""

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .abstraction import CallSite

DUMP_SC = Path(__file__).parent / "dump.sc"
PARSE_TIMEOUT_S = 900
QUERY_TIMEOUT_S = 600

_symbols = {}       # name -> (filename, line), first definition wins as in backend_ts
_ranges = {}        # name -> (filename, first line, last line), 1-indexed and inclusive
_calls = {}         # name -> [CallSite]
_identifiers = {}   # name -> frozenset[str]
_keep = {}          # name -> frozenset[int], rows relative to the method's first line
_repo_dir = None    # the checkout the tables describe, for resolving filenames
_file_lines = {}    # resolved path -> the file's lines, read once
_from_cache = False # did the last index() read the disk cache instead of running joern?


def _cache_path(repo_dir):
    """<workspace>/joern_cache/<worktree>-<hash>.json; the dump only depends on the checkout."""
    repo_dir = Path(repo_dir).resolve()
    digest = hashlib.sha256(str(repo_dir).encode("utf-8")).hexdigest()[:12]
    return repo_dir.parent.parent / "joern_cache" / f"{repo_dir.name}-{digest}.json"


def _load(methods, repo_dir=None):
    """Fill the tables from the dump payload."""
    global _symbols, _ranges, _calls, _identifiers, _keep, _repo_dir, _file_lines
    _symbols, _ranges, _calls, _identifiers, _keep = {}, {}, {}, {}, {}
    _repo_dir = Path(repo_dir).resolve() if repo_dir else None
    _file_lines = {}
    for m in methods:
        name = m["name"]
        if name in _symbols:
            continue
        filename = m.get("filename", "")
        start = int(m.get("lineNumber") or 0)
        _symbols[name] = (filename, start)
        _ranges[name] = (filename, start, int(m.get("lineNumberEnd") or start))
        _calls[name] = [CallSite(c["name"], list(c.get("conditions", [])))
                        for c in m.get("calls", [])]
        _identifiers[name] = frozenset(m.get("identifiers", []))
        _keep[name] = frozenset(int(x) for x in m.get("keepLines", []))
    return len(_symbols)


def _run(cmd, cwd, timeout_s):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return r if r.returncode == 0 else None


def _build_dump(repo_dir):
    """joern-parse then dump.sc; None when either fails, which is a result and not a crash."""
    repo_dir = Path(repo_dir).resolve()      # joern-parse runs from the work directory
    workdir = Path(tempfile.mkdtemp(prefix="joern_"))
    cpg_path, out_path = workdir / "cpg.bin", workdir / "dump.json"
    t0 = time.time()
    try:
        # --nooverlays skips the dataflow fixpoint: nothing here reads reaching-def edges.
        if _run(["joern-parse", str(repo_dir), "-o", str(cpg_path), "--nooverlays"],
                workdir, PARSE_TIMEOUT_S) is None or not cpg_path.exists():
            print(f"[joern] joern-parse failed ({time.time() - t0:.0f}s)", flush=True)
            return None
        if _run(["joern", "--script", str(DUMP_SC), "--param", f"cpgPath={cpg_path}",
                 "--param", f"outPath={out_path}"],
                workdir, QUERY_TIMEOUT_S) is None or not out_path.exists():
            print(f"[joern] dump.sc failed ({time.time() - t0:.0f}s)", flush=True)
            return None
        return json.loads(out_path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, ValueError) as e:
        print(f"[joern] dump unreadable: {e}", flush=True)
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def index(repo_dir):
    """Build the tables for this checkout, from disk when already dumped; 0 means it failed."""
    global _from_cache
    _from_cache = False
    cache = _cache_path(repo_dir)
    if cache.exists():
        try:
            n = _load(json.loads(cache.read_text(encoding="utf-8")), repo_dir)
            _from_cache = True
            print(f"[joern] {n} methods from cache ({cache.name})", flush=True)
            return n
        except (OSError, ValueError):
            pass                                   # a corrupt cache is rebuilt

    t0 = time.time()
    methods = _build_dump(repo_dir)
    if methods is None:
        return _load([], repo_dir)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".json.partial")
        tmp.write_text(json.dumps(methods), encoding="utf-8")
        tmp.replace(cache)
    except OSError:
        pass                                       # an unwritable cache must not fail the run
    n = _load(methods, repo_dir)
    print(f"[joern] indexed {n} methods ({time.time() - t0:.0f}s)", flush=True)
    return n


def index_from_cache():
    """Did the last index() read the disk cache? Recorded per row, cold cost is the s3 number."""
    return _from_cache


def symbols():
    """name -> (file path, 1-indexed line)."""
    return dict(_symbols)


def _lines_of(filename):
    """The file's lines, read once; a Joern filename may be relative or absolute."""
    path = Path(filename)
    if not path.is_absolute() and _repo_dir is not None:
        path = _repo_dir / path
    key = str(path)
    if key not in _file_lines:
        try:
            _file_lines[key] = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            _file_lines[key] = []
    return _file_lines[key]


def get_function(name):
    """The source sliced out of the file by the CPG line range, never the truncated code."""
    entry = _ranges.get(name)
    if entry is None:
        return None
    filename, start, end = entry
    lines = _lines_of(filename)
    if not lines or start <= 0:
        return None
    return "\n".join(lines[start - 1:max(end, start)]) or None


def get_calls(name):
    return _calls.get(name)          # None when the dump does not hold the name: a counted miss


def identifiers(name):
    return _identifiers.get(name)


def keep_lines(name):
    return _keep.get(name)


def parse_source(source, filename=None):
    """Always None: parsing a detached body would mean one JVM per string, so the arm stays pure."""
    return None
