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

from .abstraction import CallSite

_C = ts.Language(tree_sitter_c.language())
_CPP = ts.Language(tree_sitter_cpp.language())
_PARSER_C = ts.Parser(_C)
_PARSER_CPP = ts.Parser(_CPP)

_CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}
_CONTROL = {"if_statement", "while_statement", "for_statement", "switch_statement",
            "do_statement"}
_IDENTIFIER_TYPES = {"identifier", "field_identifier", "type_identifier"}

# Node types R1 keeps: what a function declares, branches on, calls and returns. The
# head line only -- keeping an if_statement entire would keep its body with it, and R1
# would stop being a reduction.
_KEEP = {"declaration", "if_statement", "while_statement", "for_statement",
         "switch_statement", "do_statement", "return_statement", "goto_statement",
         "case_statement", "labeled_statement", "call_expression"}

_symbols = {}   # name -> (Path, line)
_trees = {}     # Path -> (source bytes, root node)
_walked = {}    # name -> (calls, identifiers, keep_lines)


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


def _condition_text(ctrl):
    """The condition of a control structure, whitespace-collapsed and unparenthesised."""
    cond = ctrl.child_by_field_name("condition")
    if cond is None:
        return None
    text = " ".join(cond.text.decode("utf-8", "ignore").split())
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text or None


def _enclosing_conditions(call, stop):
    """Conditions of the control structures between a call and its function, outermost first."""
    conds = []
    node = call.parent
    while node is not None and node is not stop:
        if node.type in _CONTROL:
            text = _condition_text(node)
            if text:
                conds.append(text)
        node = node.parent
    conds.reverse()
    return conds


def _collect_calls(node, fn_node, out):
    """Append a CallSite for every call in the subtree."""
    if node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn is not None and fn.type in ("identifier", "field_identifier"):
            out.append(CallSite(
                name=fn.text.decode("utf-8", "ignore"),
                conditions=_enclosing_conditions(node, fn_node),
            ))
    for child in node.children:
        _collect_calls(child, fn_node, out)


def _walk_node(fn_node):
    """(calls, identifiers, keep_lines) for one function_definition node.

    keep_lines rows are 0-indexed from the start of the function, so a caller can index
    into the source this backend returned without knowing where in the file it came from.
    """
    calls = []
    _collect_calls(fn_node, fn_node, calls)

    base = fn_node.start_point[0]
    idents, rows = set(), set()
    stack = [fn_node]
    while stack:
        n = stack.pop()
        if n.type in _IDENTIFIER_TYPES:
            idents.add(n.text.decode("utf-8", "ignore"))
        if n.type in _KEEP:
            rows.add(n.start_point[0] - base)
        stack.extend(n.children)
    return calls, frozenset(idents), frozenset(rows)


def _analyzed(name):
    """Memoised walk of an indexed function, or None if this backend cannot locate it.

    S4 asks the whole pool whether it calls a given name, once per sample, so the same
    bodies are walked over and over within a checkout.
    """
    if name not in _walked:
        node = _func_node(name)
        _walked[name] = None if node is None else _walk_node(node)
    return _walked[name]


def get_calls(name):
    hit = _analyzed(name)
    return None if hit is None else hit[0]


def identifiers(name):
    hit = _analyzed(name)
    return None if hit is None else hit[1]


def keep_lines(name):
    hit = _analyzed(name)
    return None if hit is None else hit[2]
