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
        _symbols.setdefault(name, (Path(path), line))   # first definition wins
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


def _func_name(fd):
    """Name identifier of a function_definition node, or None."""
    decl = _find(fd, "function_declarator")
    if decl is None:
        return None
    name = decl.child_by_field_name("declarator")
    while name is not None and name.type not in ("identifier", "field_identifier"):
        if name.type == "qualified_identifier":
            # C++ Class::method: the name is in the 'name' field. The leftmost child is
            # the class or namespace scope, which would return the wrong identifier.
            inner = name.child_by_field_name("name")
        else:
            inner = name.child_by_field_name("declarator")
        name = inner if inner is not None else (name.children[0] if name.children else None)
    return name.text.decode("utf-8", "ignore") if name is not None else None


def _line_start_col(src, row):
    """0-indexed column of the first non-whitespace byte on a 0-indexed row."""
    lines = src.split(b"\n")
    if row >= len(lines):
        return 0
    text = lines[row]
    return len(text) - len(text.lstrip())


def _func_node(name):
    """The function_definition node a symbol resolves to, or None.

    Descends to the ctags line and climbs back up. A function wrapped in #ifdef,
    extern "C" or a namespace is not a direct child of the translation unit, so
    scanning the root's children misses it while ctags resolves it fine.
    """
    entry = _symbols.get(name)
    if entry is None:
        return None
    path, line = entry
    if not path.exists():
        return None
    src, root = _tree(path)
    row = line - 1
    col = _line_start_col(src, row)
    node = root.descendant_for_point_range((row, col), (row, col))
    while node is not None and node.type != "function_definition":
        node = node.parent
    if node is not None and _func_name(node) == name:
        return node
    return None


def get_function(name):
    """Source of an indexed function, or None if this backend cannot locate it."""
    node = _func_node(name)
    return node.text.decode("utf-8", "ignore") if node is not None else None
