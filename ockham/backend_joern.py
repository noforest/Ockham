"""CPG backend: one joern-parse plus one dump.sc per checkout, same convention as backend_ts."""

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
_source = {}        # name -> the method's source
_calls = {}         # name -> [CallSite]
_identifiers = {}   # name -> frozenset[str]
_keep = {}          # name -> frozenset[int], rows relative to the method's first line


def _load(methods):
    """Fill the tables from the dump payload."""
    global _symbols, _source, _calls, _identifiers, _keep
    _symbols, _source, _calls, _identifiers, _keep = {}, {}, {}, {}, {}
    for m in methods:
        name = m["name"]
        if name in _symbols:
            continue
        _symbols[name] = (m.get("filename", ""), int(m.get("lineNumber") or 0))
        _source[name] = m.get("code") or ""
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
    """Build the tables for this checkout; 0 methods means the backend failed on it."""
    t0 = time.time()
    methods = _build_dump(repo_dir)
    if methods is None:
        return _load([])
    n = _load(methods)
    print(f"[joern] indexed {n} methods ({time.time() - t0:.0f}s)", flush=True)
    return n


def symbols():
    """name -> (file path, 1-indexed line)."""
    return dict(_symbols)


def get_function(name):
    return _source.get(name) or None


def get_calls(name):
    return _calls.get(name)          # None when the dump does not hold the name: a counted miss


def identifiers(name):
    return _identifiers.get(name)


def keep_lines(name):
    return _keep.get(name)


def parse_source(source, filename=None):
    """Always None: parsing a detached body would mean one JVM per string, so the arm stays pure."""
    return None
