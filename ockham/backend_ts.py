"""tree-sitter + ctags backend.

ctags resolves a name to (file, line); tree-sitter parses that file and walks the AST.
Module-level state, reset by each index() call -- one checkout at a time.
"""

import tree_sitter as ts
import tree_sitter_c
import tree_sitter_cpp

_C = ts.Language(tree_sitter_c.language())
_CPP = ts.Language(tree_sitter_cpp.language())
_PARSER_C = ts.Parser(_C)
_PARSER_CPP = ts.Parser(_CPP)

_CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}

_trees = {}     # Path -> (source bytes, root node)


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
