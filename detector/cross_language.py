"""
One smell detector, four languages: Python, Java, C++ and C.

Why not PMD and clang-tidy
--------------------------
The obvious route to Java and C++ is to bolt on the established tool for each --
PMD or Checkstyle for Java, clang-tidy or cppcheck for C/C++. That gives three
tools with three different ideas of what "long method" means: PMD's default is 100
lines, this project's Python detector uses 15, and cppcheck has no such check at
all. Cross-language numbers produced that way cannot be compared, which would
undercut the only question the cross-language phase exists to answer.

So the definitions live here once and are evaluated against tree-sitter syntax
trees, which exist for all four languages. Thresholds are shared with
detector/smell_detector.py so the Python results stay continuous with what has
already been measured.

What C can and cannot have
--------------------------
Eight of the 25 targeted smells are defined over classes and C has none. This
module reports per-language applicability rather than silently returning zero,
because a smell that cannot fire looks identical to a smell that never occurs, and
C would appear cleaner than the other languages for a purely mechanical reason.

    from detector.cross_language import detect, LANGUAGES
    detect(source_bytes, "java")
"""

from collections import defaultdict

from tree_sitter import Language, Parser

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_java
import tree_sitter_python

# --- thresholds, shared with detector/smell_detector.py ----------------------
MAX_METHOD_LINES = 15
MAX_PARAMS = 3
MAX_NESTING_DEPTH = 4
MAX_CLASS_METHODS = 10
SWITCH_MIN_BRANCHES = 4
ALLOWED_NUMBERS = {"0", "1", "2", "-1", "0.0", "1.0", "100", "0L", "1L", "0.0f", "1.0f"}

# Node type names differ per grammar; everything else about the checks does not.
SPEC = {
    "python": {
        "module": tree_sitter_python,
        "function": {"function_definition"},
        "params": {"parameters", "lambda_parameters"},
        "param_item": {"identifier", "typed_parameter", "default_parameter",
                       "typed_default_parameter", "list_splat_pattern",
                       "dictionary_splat_pattern"},
        "class": {"class_definition"},
        "control": {"if_statement", "for_statement", "while_statement",
                    "with_statement", "try_statement"},
        "number": {"integer", "float"},
        "switch": {"match_statement"},
        "switch_case": {"case_clause"},
        "self_params": {"self", "cls"},
    },
    "java": {
        "module": tree_sitter_java,
        "function": {"method_declaration", "constructor_declaration"},
        "params": {"formal_parameters"},
        "param_item": {"formal_parameter", "spread_parameter"},
        "class": {"class_declaration", "interface_declaration", "record_declaration"},
        "control": {"if_statement", "for_statement", "enhanced_for_statement",
                    "while_statement", "do_statement", "try_statement",
                    "switch_expression"},
        "number": {"decimal_integer_literal", "hex_integer_literal",
                   "decimal_floating_point_literal", "octal_integer_literal"},
        "switch": {"switch_expression", "switch_statement"},
        "switch_case": {"switch_block_statement_group", "switch_rule"},
        "self_params": set(),
    },
    "cpp": {
        "module": tree_sitter_cpp,
        "function": {"function_definition"},
        "params": {"parameter_list"},
        "param_item": {"parameter_declaration", "optional_parameter_declaration",
                       "variadic_parameter_declaration"},
        "class": {"class_specifier", "struct_specifier"},
        "control": {"if_statement", "for_statement", "for_range_loop",
                    "while_statement", "do_statement", "try_statement",
                    "switch_statement"},
        "number": {"number_literal"},
        "switch": {"switch_statement"},
        "switch_case": {"case_statement"},
        "self_params": set(),
    },
    "c": {
        "module": tree_sitter_c,
        "function": {"function_definition"},
        "params": {"parameter_list"},
        "param_item": {"parameter_declaration", "variadic_parameter"},
        "class": set(),                       # C has no classes. See module docstring.
        "control": {"if_statement", "for_statement", "while_statement",
                    "do_statement", "switch_statement"},
        "number": {"number_literal"},
        "switch": {"switch_statement"},
        "switch_case": {"case_statement"},
        "self_params": set(),
    },
}

LANGUAGES = tuple(SPEC)

# Which smells this module can decide, per language. A smell absent here is not
# "zero occurrences" -- it is "cannot be asked", and the caller must not average
# the two together.
APPLICABLE = {
    "Long Method":          set(LANGUAGES),
    "Long Parameter List":  set(LANGUAGES),
    "Deep Nesting":         set(LANGUAGES),
    "Magic Numbers/Strings": set(LANGUAGES),
    "Switch Statements":    set(LANGUAGES),
    "God Class / Large Class": {"python", "java", "cpp"},
}

_parsers = {}


def parser_for(lang):
    if lang not in _parsers:
        if lang not in SPEC:
            raise ValueError(f"unsupported language {lang!r}; have {LANGUAGES}")
        _parsers[lang] = Parser(Language(SPEC[lang]["module"].language()))
    return _parsers[lang]


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _lines(node):
    return node.end_point[0] - node.start_point[0] + 1


def _params_of(node, spec):
    """Count declared parameters, discounting Python's implicit self/cls."""
    for child in _walk(node):
        if child.type in spec["params"]:
            items = [c for c in child.named_children if c.type in spec["param_item"]]
            n = len(items)
            if spec["self_params"]:
                for it in items:
                    if it.text.decode("utf8", "replace").strip() in spec["self_params"]:
                        n -= 1
            return n
    return 0


def _nesting(node, spec):
    best = 0

    def rec(n, depth):
        nonlocal best
        for c in n.children:
            if c.type in spec["control"]:
                best = max(best, depth + 1)
                rec(c, depth + 1)
            else:
                rec(c, depth)

    rec(node, 0)
    return best


def detect(source, lang):
    """Return a list of smell dicts. `source` may be str or bytes."""
    if isinstance(source, str):
        source = source.encode("utf8", "replace")
    spec = SPEC[lang]
    tree = parser_for(lang).parse(source)
    root = tree.root_node
    out = []

    for node in _walk(root):
        if node.type in spec["function"]:
            n_lines = _lines(node)
            if n_lines > MAX_METHOD_LINES:
                out.append({"smell": "Long Method", "lines": n_lines,
                            "line_number": node.start_point[0] + 1,
                            "threshold": MAX_METHOD_LINES})
            n_params = _params_of(node, spec)
            if n_params > MAX_PARAMS:
                out.append({"smell": "Long Parameter List", "params": n_params,
                            "line_number": node.start_point[0] + 1,
                            "threshold": MAX_PARAMS})
            depth = _nesting(node, spec)
            if depth >= MAX_NESTING_DEPTH:
                out.append({"smell": "Deep Nesting", "depth": depth,
                            "line_number": node.start_point[0] + 1,
                            "threshold": MAX_NESTING_DEPTH})

        elif node.type in spec["class"]:
            methods = [c for c in _walk(node) if c.type in spec["function"]]
            if len(methods) > MAX_CLASS_METHODS:
                out.append({"smell": "God Class / Large Class",
                            "methods": len(methods),
                            "line_number": node.start_point[0] + 1,
                            "threshold": MAX_CLASS_METHODS})

        elif node.type in spec["switch"]:
            cases = [c for c in _walk(node) if c.type in spec["switch_case"]]
            if len(cases) >= SWITCH_MIN_BRANCHES:
                out.append({"smell": "Switch Statements", "branches": len(cases),
                            "line_number": node.start_point[0] + 1,
                            "threshold": SWITCH_MIN_BRANCHES})

        elif node.type in spec["number"]:
            text = node.text.decode("utf8", "replace")
            if text not in ALLOWED_NUMBERS:
                out.append({"smell": "Magic Numbers/Strings", "value": text,
                            "line_number": node.start_point[0] + 1})

    # An if/else-if ladder is the same design problem as a switch, and Python has
    # no switch before 3.10, so it would otherwise be unmeasurable there.
    out.extend(_ladders(root, spec))
    return out


def _ladders(root, spec):
    """Count if / else-if chains, which the grammars shape differently.

    Python hangs every `elif` off one if_statement as siblings, so the chain is
    flat. C, C++ and Java have no `elif`: `else if` is an else_clause containing a
    fresh if_statement, so the chain is a right-leaning nest. Walking only one of
    those shapes silently reports zero for the other languages.
    """
    out, seen = [], set()
    for node in _walk(root):
        if node.type != "if_statement" or id(node) in seen:
            continue
        seen.add(id(node))

        # flat form: elif siblings
        branches = 1 + sum(1 for c in node.children if c.type == "elif_clause")
        if any(c.type == "else_clause" for c in node.children) and branches > 1:
            branches += 1

        # nested form: else { if ... }
        cur = node
        while True:
            nxt = None
            for c in cur.children:
                if c.type == "else_clause":
                    inner = [g for g in _walk(c) if g.type == "if_statement"]
                    if inner:
                        nxt = inner[0]
                    break
            if nxt is None or id(nxt) in seen:
                break
            seen.add(id(nxt))
            branches += 1
            cur = nxt

        if branches >= SWITCH_MIN_BRANCHES:
            out.append({"smell": "Switch Statements", "branches": branches,
                        "line_number": node.start_point[0] + 1,
                        "form": "if/else ladder",
                        "threshold": SWITCH_MIN_BRANCHES})
    return out


def summary(source, lang):
    found = detect(source, lang)
    by = defaultdict(int)
    for s in found:
        by[s["smell"]] += 1
    return {"total": len(found), "by_smell": dict(by),
            "types": sorted(by), "applicable": sorted(
                s for s, langs in APPLICABLE.items() if lang in langs)}
