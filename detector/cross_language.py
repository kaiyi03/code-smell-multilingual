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

import re
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
CHAIN_MIN_LINKS = 3       # a.b.c.d() -- three hops past the root
LAZY_MAX_METHODS = 1
# No line threshold here, deliberately. detector/extended_smells.py pairs the
# method count with "<= 5 lines", which works for Python and does not transfer:
# lines per class are a property of how verbose a language is, not of how little
# the class does. Measured on the lazy_class prompts, the classes the models wrote
# have a median of 1 method in both Python and C++ -- the smell, in both -- but a
# median of 6 lines in Python against 8 in C++, so the shared 5-line rule admitted
# 36 of 92 Python classes and 1 of 56 C++ ones. Mean file length across the corpus
# runs 31 lines for Python against 80 for C++, so any shared line threshold means
# something different in each language. The method count is what carries the smell.
COMMENT_DENSITY = 0.4
COMMENT_MIN = 5
DATA_CLASS_MIN_FIELDS = 2
# Duplication is measured on the shape of the syntax tree, not on characters.
# detector/smell_detector.py compares sets of characters, which cannot carry a
# shared threshold across these four languages: C++ is dense with {};<>:: and
# Python is not, so the same 0.75 would mean something different per language.
# Node-type trigrams also ignore identifier renaming, which is the case a
# character measure is weakest on.
DUP_MIN_SIMILARITY = 0.75
DUP_MIN_TRIGRAMS = 8      # below this a body is too small to call duplicated
# Code inside a comment, in any of the four languages: a keyword, or something
# that looks like an assignment or a call. Applied to the comment's contents, so
# it does not need to know whether the marker was #, // or /* */.
CODE_IN_COMMENT = re.compile(
    r"\b(?:def|class|if|for|while|return|import|from|try|except|catch|with"
    r"|elif|else|switch|printf|cout|public|private|int|void)\b"
    r"|System\.out"
    r"|[A-Za-z_][A-Za-z0-9_]*\s*(?:=[^=]|\()")

ALLOWED_NUMBERS = {"0", "1", "2", "-1", "0.0", "1.0", "100", "0L", "1L", "0.0f", "1.0f"}

# Node type names differ per grammar; everything else about the checks does not.
SPEC = {
    "python": {
        "module": tree_sitter_python,
        "behaviour_class": {"class_definition"},
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
        "comment": {"comment"},
        "chain": {"attribute"},
        "field_decl": set(),          # no field declarations; fields are self.x
        "assign": {"assignment"},
        "terminator": {"return_statement", "raise_statement", "break_statement",
                       "continue_statement"},
        "const_marker": set(),        # ALL_CAPS is the convention, handled below
    },
    "java": {
        "module": tree_sitter_java,
        "behaviour_class": {"class_declaration", "record_declaration"},
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
        "comment": {"line_comment", "block_comment"},
        "chain": {"method_invocation"},
        "field_decl": {"field_declaration"},
        "assign": {"field_declaration", "local_variable_declaration"},
        "terminator": {"return_statement", "throw_statement", "break_statement",
                       "continue_statement"},
        "const_marker": {"final"},
    },
    "cpp": {
        "module": tree_sitter_cpp,
        "behaviour_class": {"class_specifier"},
        "function": {"function_definition"},
        "params": {"parameter_list"},
        "param_item": {"parameter_declaration", "optional_parameter_declaration",
                       "variadic_parameter_declaration"},
        "class": {"class_specifier", "struct_specifier"},
        # C++ declares members in the class and defines them elsewhere; see
        # _methods_of. Only field_declarations that declare a function count.
        "member_decl": {"field_declaration"},
        "control": {"if_statement", "for_statement", "for_range_loop",
                    "while_statement", "do_statement", "try_statement",
                    "switch_statement"},
        "number": {"number_literal"},
        "switch": {"switch_statement"},
        "switch_case": {"case_statement"},
        "self_params": set(),
        "comment": {"comment"},
        "chain": {"field_expression"},
        "field_decl": {"field_declaration"},
        "assign": {"declaration"},
        "terminator": {"return_statement", "throw_statement", "break_statement",
                       "continue_statement", "goto_statement"},
        "const_marker": {"const", "constexpr"},
    },
    "c": {
        "module": tree_sitter_c,
        "behaviour_class": set(),
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
        "comment": {"comment"},
        "chain": {"field_expression"},
        "field_decl": {"field_declaration"},
        "assign": {"declaration"},
        "terminator": {"return_statement", "break_statement", "continue_statement",
                       "goto_statement"},
        "const_marker": {"const"},
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
    # --- added in the second pass. Each is decidable from the tree alone; the
    # fifteen still missing are listed in OUT_OF_REACH below with the reason.
    "Duplicated Code":      set(LANGUAGES),
    "Global State":         set(LANGUAGES),
    "Message Chains":       set(LANGUAGES),
    "Comments (as smell indicator)": set(LANGUAGES),
    # A class with fields and no behaviour. C has no classes, and a bare struct is
    # ordinary C rather than a smell, so asking the question of C would count
    # idiomatic code as a defect.
    "Data Class":           {"python", "java", "cpp"},
    "Lazy Class":           {"python", "java", "cpp"},
}

# What this module still cannot decide, and why -- recorded here so the gap is a
# stated limit rather than something a reader has to infer from absence.
OUT_OF_REACH = {
    # Tried and withdrawn rather than never attempted. Both halves of the usual
    # definition were implemented: statements after a return, which models almost
    # never write, and internal functions nothing calls, which fires about as often
    # on files that asked for some other smell. Lift came out at +1.4, +0.6, -3.9
    # and -0.3 across the four languages -- a base rate wearing the smell's name.
    # The Python ast detector reaches +7.3 on the same smell, so two independent
    # instruments both fail to separate it, which is evidence about the smell rather
    # than about either detector.
    "Dead Code": "fires as often on code that was not asked for it; see git history",
    "Feature Envy": "needs to resolve which object each field access belongs to",
    "Middle Man": "needs to tell a delegating call from a working one",
    "Inappropriate Intimacy": "needs per-class field ownership across two classes",
    "Temporary Field": "needs to know a field is unset on most paths",
    "Data Clumps": "needs parameter identity matched across signatures and types",
    "Refused Bequest": "needs the parent's members, usually in another file",
    "Parallel Inheritance Hierarchies": "needs two hierarchies compared by name",
    "Speculative Generality": "needs to know an abstraction has one user",
    "Primitive Obsession": "a judgement about domain modelling, not a count",
    "Shotgun Surgery": "needs change history, not a syntax tree",
    "Incomplete Library Class": "needs the library's intent",
    "Alternative Classes with Different Interfaces": "needs semantic equivalence",
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


def _text(node):
    return node.text.decode("utf8", "replace")


def _is_accessor(name):
    """get_x, getX, set_total, to_dict, from_json -- a field door, not behaviour."""
    bare = name.lstrip("_").lower().replace("_", "")
    return bare.startswith(("get", "set", "to", "from", "is", "has"))


def _chain_depth(node, spec):
    """How many hops a member-access chain makes past its root.

    The three grammars nest this differently but all nest it the same way: each
    hop wraps the one before, so the depth is how deep the chain nodes stack.
    Python nests `attribute`, Java `method_invocation`, C and C++ `field_expression`
    (which covers both `a.b` and `p->b`).
    """
    n, cur = 0, node
    while cur is not None:
        nxt = None
        for c in cur.named_children:
            if c.type in spec["chain"]:
                nxt = c
                break
            # a.b().c() puts a call between the two member accesses
            if c.type in ("call", "call_expression", "argument_list"):
                for g in c.named_children:
                    if g.type in spec["chain"]:
                        nxt = g
                        break
            if nxt is not None:
                break
        if nxt is None:
            break
        n += 1
        cur = nxt
    return n + 1


def _message_chains(root, spec):
    """Chains the caller walks through someone else's object graph."""
    out, inner = [], set()
    chains = [n for n in _walk(root) if n.type in spec["chain"]]
    for n in chains:
        for c in _walk(n):
            if c.id != n.id and c.type in spec["chain"]:
                inner.add(c.id)
    for n in chains:
        if n.id in inner:                      # report the outermost hop only
            continue
        depth = _chain_depth(n, spec)
        if depth >= CHAIN_MIN_LINKS:
            out.append({"smell": "Message Chains", "depth": depth,
                        "line_number": n.start_point[0] + 1,
                        "threshold": CHAIN_MIN_LINKS})
    return out[:5]


def _global_state(root, spec, lang):
    """Mutable state at file scope, which any function can reach and change.

    Each language spells it differently and the difference is not cosmetic: a
    Java field is only global if it is `static`, a C or C++ file-scope variable
    is global by being there at all, and Python's convention is that ALL_CAPS
    means constant. Constants are excluded in every language -- a named constant
    is the fix for Magic Numbers, not a smell of its own.
    """
    out = []
    if lang == "python":
        for node in root.named_children:
            if node.type != "expression_statement":
                continue
            for a in node.named_children:
                if a.type not in spec["assign"]:
                    continue
                tgt = a.child_by_field_name("left")
                if tgt is not None and tgt.type == "identifier":
                    name = _text(tgt)
                    if not name.isupper():
                        out.append({"smell": "Global State", "variable": name,
                                    "line_number": a.start_point[0] + 1})
    elif lang == "java":
        for node in _walk(root):
            if node.type != "field_declaration":
                continue
            mods = next((c for c in node.named_children if c.type == "modifiers"), None)
            if mods is None:
                continue
            words = {_text(c) for c in mods.children}
            if "static" in words and not (words & spec["const_marker"]):
                out.append({"smell": "Global State",
                            "variable": _text(node).strip().rstrip(";"),
                            "line_number": node.start_point[0] + 1})
    else:                                       # c, cpp: file-scope declarations
        for node in root.named_children:
            if node.type not in spec["assign"]:
                continue
            words = {_text(c) for c in node.children}
            if words & spec["const_marker"]:
                continue
            out.append({"smell": "Global State",
                        "variable": _text(node).strip().rstrip(";"),
                        "line_number": node.start_point[0] + 1})
    return out


def _dead_code(root, spec, lang):
    """A statement standing after return, throw, break or continue in one block.

    Only the same block counts. A return inside an if, followed by code after the
    if, is ordinary control flow rather than dead code, and treating it otherwise
    would fire on most well-written functions.
    """
    out = []
    for node in _walk(root):
        # A declaration is as dead as a statement, and only Python spells one as a
        # statement: `int x = 2;` is a local_variable_declaration in Java and a
        # declaration in C and C++. Matching only *_statement silently made the
        # terminator the last item in every brace-language block, so nothing ever
        # fired outside Python.
        kids = [c for c in node.named_children
                if c.type.endswith(("statement", "declaration"))
                or c.type in spec["terminator"]]
        for i, stmt in enumerate(kids[:-1]):
            if stmt.type in spec["terminator"]:
                nxt = kids[i + 1]
                if nxt.type in ("comment",) or nxt.type in spec["comment"]:
                    continue
                out.append({"smell": "Dead Code", "kind": "unreachable",
                            "line_number": nxt.start_point[0] + 1})
                break
    out.extend(_uncalled(root, spec, lang))
    return out


def _fn_name(fn, spec):
    """The declared name of a function, however the language attaches it."""
    n = fn.child_by_field_name("name")
    if n is not None:
        return _text(n)
    dec = fn.child_by_field_name("declarator")
    while dec is not None:
        ident = next((c for c in dec.named_children
                      if c.type in ("identifier", "field_identifier",
                                    "qualified_identifier")), None)
        if ident is not None:
            return _text(ident).split("::")[-1]
        dec = dec.child_by_field_name("declarator")
    return ""


def _called_names(root):
    """Every name that appears in a call position anywhere in the file."""
    out = set()
    for n in _walk(root):
        if n.type in ("call", "call_expression"):
            f = n.child_by_field_name("function")
            if f is None:
                continue
            if f.type in ("identifier",):
                out.add(_text(f))
            else:                       # a.b() / p->b() / Obj::b()
                last = None
                for c in _walk(f):
                    if c.type in ("identifier", "field_identifier"):
                        last = c
                if last is not None:
                    out.add(_text(last))
        elif n.type == "method_invocation":          # java
            nm = n.child_by_field_name("name")
            if nm is not None:
                out.add(_text(nm))
    return out


def _is_internal(fn, spec, lang):
    """Can anything outside this file reach it?

    Only internal definitions count as dead. A public function with no caller here
    may simply be called from elsewhere -- and in this corpus the prompt usually
    asks for exactly one public function, so counting those would fire on nearly
    every file and measure the prompt format rather than the smell.
    """
    if lang == "python":
        return _fn_name(fn, spec).startswith("_")
    words = set()
    cur = fn
    for _ in range(3):                  # the modifier may sit on an enclosing node
        if cur is None:
            break
        words |= {_text(c) for c in cur.children if not c.is_named}
        # Java puts the keyword in a modifiers node; C and C++ wrap it in a
        # storage_class_specifier or access_specifier, so neither is a bare token.
        words |= {_text(g) for c in cur.named_children if c.type == "modifiers"
                  for g in c.children}
        words |= {_text(c).strip(": ") for c in cur.named_children
                  if c.type in ("storage_class_specifier", "access_specifier")}
        cur = cur.parent
    return bool(words & {"private", "static"})


def _uncalled(root, spec, lang):
    """Internal functions that nothing in the file ever calls."""
    called = _called_names(root)
    out = []
    for fn in _walk(root):
        if fn.type not in spec["function"]:
            continue
        name = _fn_name(fn, spec)
        if not name or name in called:
            continue
        if name.startswith("__") or name in ("main", "Main"):
            continue
        if not _is_internal(fn, spec, lang):
            continue
        out.append({"smell": "Dead Code", "kind": "uncalled", "function": name,
                    "line_number": fn.start_point[0] + 1})
    return out[:5]


def _comments(root, spec, src):
    """Commented-out code, and comment volume standing in for clarity."""
    out = []
    comments = [n for n in _walk(root) if n.type in spec["comment"]]
    if not comments:
        return out

    # Commented-out code, judged on what is inside the comment rather than on a
    # per-language regex over the raw line.
    code_like = 0
    first = None
    for c in comments:
        body = _text(c).lstrip("/#* \t").rstrip("*/ \t")
        if CODE_IN_COMMENT.search(body):
            code_like += 1
            first = first or c.start_point[0] + 1
    if code_like:
        out.append({"smell": "Comments (as smell indicator)",
                    "kind": "commented-out code", "count": code_like,
                    "line_number": first})

    comment_lines = sum(_text(c).count("\n") + 1 for c in comments)
    total = len([l for l in src.splitlines() if l.strip()])
    code_lines = max(total - comment_lines, 0)
    if comment_lines >= COMMENT_MIN and code_lines and \
            comment_lines / code_lines >= COMMENT_DENSITY:
        out.append({"smell": "Comments (as smell indicator)", "kind": "density",
                    "comments": comment_lines, "code_lines": code_lines,
                    "threshold": COMMENT_DENSITY,
                    "line_number": comments[0].start_point[0] + 1})
    return out


def _fields_of(cls, spec, lang):
    """The data a class holds, by whichever route the language declares it."""
    if lang == "python":
        names = set()
        for n in _walk(cls):
            if n.type not in spec["assign"]:
                continue
            tgt = n.child_by_field_name("left")
            if tgt is not None and tgt.type == "attribute":
                t = _text(tgt)
                if t.startswith("self."):
                    names.add(t)
        return names
    out = set()
    for n in _walk(cls):
        if n.type in spec["field_decl"] and not any(
                g.type == "function_declarator" for g in _walk(n)):
            ident = next((g for g in _walk(n) if g.type == "field_identifier"
                          or g.type == "identifier"), None)
            if ident is not None:
                out.add(_text(ident))
    return out


def _class_smells(cls, spec, lang):
    """Data Class and Lazy Class, both decided by counting members."""
    out = []
    methods = _methods_of(cls, spec)
    named = []
    for m in methods:
        ident = next((g for g in _walk(m) if g.type in ("identifier",
                                                        "field_identifier")), None)
        named.append(_text(ident) if ident is not None else "")
    # Constructors and dunders are plumbing, not behaviour, in every language here.
    cls_ident = next((c for c in cls.named_children if c.type in
                      ("identifier", "type_identifier")), None)
    cls_name = _text(cls_ident) if cls_ident is not None else ""
    behaviour = [n for n in named
                 if n and n != cls_name and not n.startswith("__")
                 and not _is_accessor(n)]
    fields = _fields_of(cls, spec, lang)

    if len(fields) >= DATA_CLASS_MIN_FIELDS and not behaviour:
        out.append({"smell": "Data Class", "class": cls_name,
                    "fields": len(fields), "behaviour_methods": 0,
                    "line_number": cls.start_point[0] + 1})

    lines = _lines(cls)
    real = [n for n in named if n and n != cls_name and not n.startswith("__")]
    # Only a construct the language expects to carry behaviour can be lazy. A C++
    # struct is a data holder by convention, so counting it here made every plain
    # struct in the corpus a Lazy Class.
    if cls.type in spec["behaviour_class"] and len(real) <= LAZY_MAX_METHODS:
        out.append({"smell": "Lazy Class", "class": cls_name,
                    "methods": len(real), "lines": lines,
                    "line_number": cls.start_point[0] + 1,
                    "threshold": LAZY_MAX_METHODS})
    return out


def _shape(fn, spec):
    """A function body reduced to the shape of its tree.

    Node types only: identifiers and literals are dropped, so two bodies that
    differ only in naming compare as identical. Trigrams of that sequence are
    what the similarity is computed over, which keeps some ordering information
    that a bag of node types would lose.
    """
    seq = [n.type for n in _walk(fn) if n.is_named
           and n.type not in ("identifier", "field_identifier", "type_identifier",
                              "comment", "string_content")]
    return {tuple(seq[i:i + 3]) for i in range(len(seq) - 2)}


def _duplicated(root, spec):
    """Function bodies with near-identical structure."""
    fns = [n for n in _walk(root) if n.type in spec["function"]]
    shapes = [(n, _shape(n, spec)) for n in fns]
    shapes = [(n, s) for n, s in shapes if len(s) >= DUP_MIN_TRIGRAMS]
    out = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            a, b = shapes[i][1], shapes[j][1]
            union = a | b
            if not union:
                continue
            sim = len(a & b) / len(union)
            if sim >= DUP_MIN_SIMILARITY:
                out.append({"smell": "Duplicated Code",
                            "similarity": round(sim, 3),
                            "threshold": DUP_MIN_SIMILARITY,
                            "line_number": shapes[i][0].start_point[0] + 1})
    return out[:5]


def _methods_of(node, spec):
    """The methods belonging to a class, counting the shapes each language uses.

    Python and Java put the body in the class, so a function node inside it is the
    whole story. C++ usually does not: the idiomatic form declares members in the
    class and defines them out of line as `void G::m() {...}`, which leaves nothing
    but a field_declaration inside the class body. Counting only definitions saw
    fourteen methods as zero, and C++ scored 12.5% on God Class against Java's
    33.3% for what is the same class written the way the language expects.

    Declarations are counted rather than the out-of-line definitions they pair
    with, so a class cannot be counted twice for one method.
    """
    out = []
    for c in _walk(node):
        if c.type in spec["function"]:
            out.append(c)
        elif c.type in spec.get("member_decl", ()):
            # A field_declaration is a method only when it declares a function;
            # otherwise it is a data member and counting it would make any class
            # with eleven fields a God Class.
            if any(g.type == "function_declarator" for g in _walk(c)):
                out.append(c)
    return out


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
            out.extend(_class_smells(node, spec, lang))
            methods = _methods_of(node, spec)
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
    out.extend(_message_chains(root, spec))
    out.extend(_global_state(root, spec, lang))
    out.extend(_dead_code(root, spec, lang))
    out.extend(_comments(root, spec, source.decode("utf8", "replace")))
    out.extend(_duplicated(root, spec))
    return out


def _alternatives(node):
    """The else/elif parts of an if_statement, in whichever shape the grammar uses.

    All three shapes are reachable through the `alternative` field, which is why
    that is what this reads rather than child node types:

        python   if_statement -> elif_clause, elif_clause, else_clause  (siblings)
        c, cpp   if_statement -> else_clause -> if_statement            (nested)
        java     if_statement -> if_statement                           (no wrapper)

    Java is the one that has no wrapper node at all, so a check written against
    `else_clause` finds nothing there and silently reports a chain length of one.
    """
    alts = [c for c in node.children if c.type in ("elif_clause", "else_clause")]
    return alts or [c for c in node.children_by_field_name("alternative")]


def _chain_length(node):
    """Branches in the if/else-if chain rooted at this if_statement.

    A trailing plain `else` counts as a branch in every language, which is what
    the previous version got wrong: Python counted it and the brace languages did
    not, so the same four-branch ladder scored 4 in Python and 3 in C and C++ --
    one short of the threshold, in the one place where one short matters.
    """
    branches, cur, guard = 1, node, 0
    while cur is not None and guard < 500:       # guard: generated code nests hard
        guard += 1
        nxt = None
        for alt in _alternatives(cur):
            if alt.type == "elif_clause":        # python's flat continuation
                branches += 1
                continue
            # else_clause wrapping a nested if (c, cpp), the nested if itself
            # (java), or a plain trailing else in any of them. Only DIRECT
            # children count: `else { if (...) }` is a nested if inside a block,
            # not another rung of this ladder.
            inner = (alt if alt.type == "if_statement" else
                     next((g for g in alt.named_children
                           if g.type == "if_statement"), None))
            branches += 1
            if inner is not None:
                nxt = inner
        cur = nxt
    return branches


def _ladders(root, spec):
    """Count if / else-if chains, which the grammars shape differently.

    An if/else-if ladder is the same design problem as a switch, and Python had no
    switch before 3.10, so without this the smell would be unmeasurable there. It
    has to be counted identically in all four languages or the comparison measures
    the grammar rather than the code.
    """
    # An if_statement reachable as another if's alternative is a rung, not a new
    # ladder. Collecting them first is clearer than mutating a seen-set mid-walk,
    # and it cannot leave a rung counted twice.
    rungs = set()
    for node in _walk(root):
        if node.type != "if_statement":
            continue
        for alt in _alternatives(node):
            inner = (alt if alt.type == "if_statement" else
                     next((g for g in alt.named_children
                           if g.type == "if_statement"), None))
            if inner is not None:
                rungs.add(inner.id)

    out = []
    for node in _walk(root):
        if node.type != "if_statement" or node.id in rungs:
            continue
        branches = _chain_length(node)
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
