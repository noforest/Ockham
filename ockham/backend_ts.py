"""tree-sitter + ctags backend.

ctags resolves a name to (file, line); tree-sitter parses that file and walks the AST.
Module-level state, reset by each index() call -- one checkout at a time.
"""

import json
import subprocess
from pathlib import Path

import tree_sitter as ts
import tree_sitter_c
import tree_sitter_cpp

_C = ts.Language(tree_sitter_c.language())
_CPP = ts.Language(tree_sitter_cpp.language())
_PARSER_C = ts.Parser(_C)
_PARSER_CPP = ts.Parser(_CPP)

_CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}

_symbols = {}   # name -> (Path, line)
_trees = {}     # Path -> (source bytes, root node)


def _run_ctags(repo_dir):
    """[[name, path, line], ...] for every function tag in the checkout."""
    out = subprocess.run(
        ["ctags", "-R", "--languages=C,C++", "--fields=+n",
         "--output-format=json", "--sort=no", str(repo_dir)],
        capture_output=True, text=True,
    ).stdout
    entries = []
    for line in out.splitlines():
        try:
            tag = json.loads(line)
        except json.JSONDecodeError:
            continue
        if tag.get("_type") == "tag" and tag.get("kind") == "function":
            entries.append([tag["name"], tag["path"], tag.get("line", 0)])
    return entries


def index(repo_dir):
    """Build name -> (file, line) for a checkout. Returns the symbol count."""
    global _symbols, _trees
    _symbols = {}
    _trees = {}
    for name, path, line in _run_ctags(repo_dir):
        _symbols[name] = (Path(path), line)
    return len(_symbols)


def symbols():
    """name -> (file path as a string, 1-indexed line). Read-only view for the pool."""
    return {name: (str(path), int(line) if line else 0)
            for name, (path, line) in _symbols.items()}


def _tree(path):
    """Parse a file once and keep its root node."""
    if path not in _trees:
        src = path.read_bytes()
        parser = _PARSER_CPP if path.suffix in _CPP_SUFFIXES else _PARSER_C
        _trees[path] = (src, parser.parse(src).root_node)
    return _trees[path]


def _find(node, type_):
    """First node of the given type, depth-first, or None."""
    if node.type == type_:
        return node
    for child in node.children:
        hit = _find(child, type_)
        if hit is not None:
            return hit
    return None
