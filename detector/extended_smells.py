"""
Additional smell detectors, extending detector/smell_detector.py.

smell_detector.py covers eight of the twenty-five smells the prompt set targets:
Long Method, Long Parameter List, Duplicated Code, Deep Nesting, God Class, Magic
Numbers, Global State, Data Class. Those eight are the ones decidable by counting
things inside a single file. The rest were never written, which capped every
induction figure at roughly a third of the corpus.

This module adds thirteen more, taking coverage to 21 of 25. Each is a heuristic
with a stated threshold -- these are structural approximations of design
judgements, not definitions, and pipeline/run_analysis.py reports a discrimination
check (how often the detector fires on files that were NOT asked for that smell)
so a check that fires on everything is visible as noise rather than counted as a
finding.

Four targeted smells are deliberately absent because no single-file rule
approximates them honestly:

  Shotgun Surgery                          needs change history across files
  Incomplete Library Class                 needs to know the library's intent
  Alternative Classes / Different Interfaces  needs semantic equivalence of methods
  Primitive Obsession                      needs a judgement about what deserves a type

Smell labels match dataset/prompts_core.json exactly so run_analysis.py can join
on them.
"""

import ast
import re
from collections import defaultdict

# --- thresholds, all adjustable and all reported in the output ---------------
CLUMP_MIN_SIZE = 3        # a clump is >= 3 values travelling together
CLUMP_MIN_USES = 2        # ... in >= 2 signatures. Fowler says "repeatedly"; at 3
                          # the check missed clumps split across a pair of methods
                          # while its false-positive rate was only 2.4%.
CHAIN_MIN_LINKS = 3       # a.b.c.d() -- three hops past the root
ENVY_MIN_FOREIGN = 3      # >= 3 accesses on another object
MIDDLE_MAN_RATIO = 0.5    # half the methods do nothing but delegate
LAZY_MAX_METHODS = 1      # a class earning its keep does more than this
LAZY_MAX_LINES = 5     # a class with one short method and a
                       # constructor is small, not lazy
SWITCH_MIN_BRANCHES = 4   # if / elif / elif / elif on one subject
COMMENT_DENSITY = 0.4
COMMENT_MIN = 5


def _methods(node):
    return [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _param_names(fn):
    a = fn.args
    names = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
    return [n for n in names if n not in ("self", "cls")]


def _body_is_stub(fn):
    """Body is only pass / ... / docstring / raise NotImplementedError."""
    real = [s for s in fn.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                    and isinstance(s.value.value, str))]
    if not real:
        return True
    for s in real:
        if isinstance(s, ast.Pass):
            continue
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis:
            continue
        if isinstance(s, ast.Raise):
            exc = s.exc
            name = getattr(exc, "id", None) or getattr(getattr(exc, "func", None), "id", None)
            if name == "NotImplementedError":
                continue
        return False
    return True


def check_data_clumps(tree):
    """The same group of values passed around together again and again."""
    groups = defaultdict(int)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = _param_names(fn)
        if len(names) >= CLUMP_MIN_SIZE:
            groups[frozenset(names)] += 1

    # A clump is a set of >= 3 names common to several signatures. Compare each
    # pair of signatures and keep intersections that recur.
    inter = defaultdict(int)
    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            common = a & b
            if len(common) >= CLUMP_MIN_SIZE:
                inter[frozenset(common)] += groups[a] + groups[b]
    for k, n in groups.items():          # identical signature repeated counts too
        if n >= CLUMP_MIN_USES:
            inter[k] = max(inter[k], n)

    out = []
    for names, n in inter.items():
        if n >= CLUMP_MIN_USES:
            out.append({"smell": "Data Clumps", "fields": sorted(names),
                        "occurrences": n,
                        "thresholds": {"min_size": CLUMP_MIN_SIZE, "min_uses": CLUMP_MIN_USES}})
    return out[:5]


def _chain_len(node):
    n = 0
    while True:
        if isinstance(node, ast.Attribute):
            n += 1
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return n


def check_message_chains(tree):
    """a.getB().getC().getD() -- the caller navigating someone else's graph."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Call)):
            depth = _chain_len(node)
            if depth >= CHAIN_MIN_LINKS:
                out.append({"smell": "Message Chains", "depth": depth,
                            "line_number": getattr(node, "lineno", 0),
                            "thresholds": {"min_links": CHAIN_MIN_LINKS}})
    # de-duplicate nested reports of the same chain by line
    seen, uniq = set(), []
    for r in sorted(out, key=lambda r: -r["depth"]):
        if r["line_number"] in seen:
            continue
        seen.add(r["line_number"])
        uniq.append(r)
    return uniq[:5]


def check_feature_envy(tree):
    """A method more interested in another object's data than its own."""
    out = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in _methods(cls):
            own, foreign = 0, defaultdict(int)
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id in ("self", "cls"):
                        own += 1
                    else:
                        foreign[node.value.id] += 1
            if not foreign:
                continue
            target, n = max(foreign.items(), key=lambda kv: kv[1])
            if n >= ENVY_MIN_FOREIGN and n > own:
                out.append({"smell": "Feature Envy", "method": f"{cls.name}.{fn.name}",
                            "envies": target, "foreign_accesses": n, "own_accesses": own,
                            "line_number": fn.lineno,
                            "thresholds": {"min_foreign": ENVY_MIN_FOREIGN}})
    return out


def check_middle_man(tree):
    """A class whose methods only forward to something else."""
    out = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        methods = [m for m in _methods(cls) if not m.name.startswith("__")]
        if len(methods) < 2:
            continue
        delegating = 0
        for fn in methods:
            body = [s for s in fn.body
                    if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                            and isinstance(s.value.value, str))]
            if len(body) != 1:
                continue
            stmt = body[0]
            val = stmt.value if isinstance(stmt, (ast.Return, ast.Expr)) else None
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute):
                root = val.func.value
                # self._thing.method(...) -- forwarding to a held collaborator
                if isinstance(root, ast.Attribute) and isinstance(root.value, ast.Name) \
                        and root.value.id == "self":
                    delegating += 1
        if delegating >= 2 and delegating / len(methods) >= MIDDLE_MAN_RATIO:
            out.append({"smell": "Middle Man", "class": cls.name,
                        "delegating_methods": delegating, "total_methods": len(methods),
                        "line_number": cls.lineno,
                        "thresholds": {"ratio": MIDDLE_MAN_RATIO}})
    return out


def check_data_class_pure(tree):
    """A class that is only fields.

    smell_detector.check_data_class requires >= 2 methods of which most are named
    get_/set_/to_/from_, so it finds the getter-and-setter form and misses the
    purer one -- a constructor assigning attributes and no behaviour at all, which
    is what a model usually writes when asked for a Data Class.
    """
    out = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        methods = [m for m in _methods(cls) if not m.name.startswith("__")]
        attrs = {n.attr for fn in _methods(cls) for n in ast.walk(fn)
                 if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                 and n.value.id == "self" and isinstance(n.ctx, ast.Store)}
        accessors = [m for m in methods
                     if m.name.startswith(("get_", "set_", "to_", "from_"))]
        if len(attrs) >= 2 and len(methods) == len(accessors):
            out.append({"smell": "Data Class", "class": cls.name,
                        "attributes": sorted(attrs), "behaviour_methods": 0,
                        "line_number": cls.lineno})
    return out


def check_lazy_class(tree):
    """A class too small to justify existing."""
    out = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        methods = [m for m in _methods(cls) if not m.name.startswith("__")]
        end = max((getattr(n, "end_lineno", n.lineno) for n in ast.walk(cls)
                   if hasattr(n, "lineno")), default=cls.lineno)
        lines = end - cls.lineno + 1
        if len(methods) == 0 or (len(methods) <= LAZY_MAX_METHODS
                                 and lines <= LAZY_MAX_LINES):
            out.append({"smell": "Lazy Class", "class": cls.name,
                        "methods": len(methods), "lines": lines,
                        "line_number": cls.lineno,
                        "thresholds": {"max_methods": LAZY_MAX_METHODS,
                                       "max_lines": LAZY_MAX_LINES}})
    return out


def _subject(test):
    """The thing an if-test is branching on, if it is a simple comparison."""
    if isinstance(test, ast.Compare) and isinstance(test.left, (ast.Name, ast.Attribute)):
        return ast.dump(test.left)
    return None


def check_switch_statements(tree):
    """A long if/elif ladder on one value, where polymorphism belongs."""
    out = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or id(node) in seen:
            continue
        subjects, n, cur = [_subject(node.test)], 1, node
        seen.add(id(node))
        while len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
            cur = cur.orelse[0]
            seen.add(id(cur))
            subjects.append(_subject(cur.test))
            n += 1
        if cur.orelse:
            n += 1
        real = [s for s in subjects if s]
        if n >= SWITCH_MIN_BRANCHES and real and len(set(real)) == 1:
            out.append({"smell": "Switch Statements", "branches": n,
                        "line_number": node.lineno,
                        "thresholds": {"min_branches": SWITCH_MIN_BRANCHES}})
    for node in ast.walk(tree):
        if isinstance(node, getattr(ast, "Match", ())) and len(node.cases) >= SWITCH_MIN_BRANCHES:
            out.append({"smell": "Switch Statements", "branches": len(node.cases),
                        "line_number": node.lineno,
                        "thresholds": {"min_branches": SWITCH_MIN_BRANCHES}})
    return out


def check_dead_code(tree):
    """Statements that can never run, and functions nobody calls."""
    out = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body[:-1]):
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                out.append({"smell": "Dead Code", "kind": "unreachable",
                            "line_number": body[i + 1].lineno})
                break

    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    called |= {n.func.attr for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    # Only private helpers count as uncalled. A public module-level function with
    # no call site is the normal shape of a generated snippet -- it IS the answer
    # to the prompt -- so flagging it would fire on almost every file in the corpus.
    for fn in tree.body:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and fn.name.startswith("_") and not fn.name.startswith("__") \
                and fn.name not in called and fn.name not in referenced:
            out.append({"smell": "Dead Code", "kind": "uncalled private helper",
                        "name": fn.name, "line_number": fn.lineno})

    # Imported and never used.
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name != "*":
                    imported[(a.asname or a.name).split(".")[0]] = node.lineno
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for name, line in imported.items():
        if name not in referenced and name not in attrs:
            out.append({"smell": "Dead Code", "kind": "unused import",
                        "name": name, "line_number": line})

    # Assigned and never read, inside a function body.
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        stored, loaded = {}, set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Store):
                    stored.setdefault(n.id, n.lineno)
                else:
                    loaded.add(n.id)
        for name, line in stored.items():
            if name not in loaded and not name.startswith("_"):
                out.append({"smell": "Dead Code", "kind": "unused variable",
                            "name": name, "line_number": line})

    # Branches that can never be taken.
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)) and isinstance(node.test, ast.Constant) \
                and not node.test.value:
            out.append({"smell": "Dead Code", "kind": "constant-false branch",
                        "line_number": node.lineno})
    return out[:8]


def _is_null_literal(node):
    """None -- "no value yet", as opposed to a real starting value.

    Deliberately not 0, "" or []: `self.balance = 0` then `self.balance += x` is an
    ordinary accumulator, and treating it as a placeholder made this fire on any
    class with a counter.
    """
    return isinstance(node, ast.Constant) and node.value is None


def check_temporary_field(tree):
    """An attribute that only holds a meaningful value some of the time.

    Two forms, and the second is much the commoner one in practice: the field is
    declared in __init__ as None or an empty container and only really populated
    inside one other method. An earlier version of this check required the
    attribute to be absent from __init__ altogether, which excluded exactly the
    canonical case and left the detector firing on almost nothing.
    """
    out = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        init_attrs, null_init, other_attrs, used = set(), set(), defaultdict(set), set()
        augmented = set()
        for fn in _methods(cls):
            if fn.name == "__init__":
                for stmt in ast.walk(fn):
                    if isinstance(stmt, ast.Assign) and _is_null_literal(stmt.value):
                        for t in stmt.targets:
                            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                                    and t.value.id == "self":
                                null_init.add(t.attr)
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                        and node.value.id == "self":
                    if isinstance(node.ctx, ast.Store):
                        if fn.name == "__init__":
                            init_attrs.add(node.attr)
                        else:
                            other_attrs[node.attr].add(fn.name)
                    else:
                        used.add(node.attr)
            # `self.x += 1` is mutation of an existing value, not a field left
            # unset, so it must not count as an assignment outside __init__.
            for stmt in ast.walk(fn):
                if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Attribute) \
                        and isinstance(stmt.target.value, ast.Name) \
                        and stmt.target.value.id == "self":
                    augmented.add(stmt.target.attr)
        for attr, setters in other_attrs.items():
            if attr in augmented:
                continue
            never_declared = attr not in init_attrs and attr in used
            declared_empty = attr in null_init
            if never_declared or declared_empty:
                out.append({"smell": "Temporary Field", "class": cls.name,
                            "field": attr, "set_in": sorted(setters),
                            "form": "null-initialised" if declared_empty else "undeclared",
                            "line_number": cls.lineno})
    return out[:8]


def check_inappropriate_intimacy(tree):
    """Two classes that know too much about each other.

    Two signals. One class reaching past another's underscore prefix, and -- the
    commoner shape in this corpus -- a pair of classes that each name the other,
    so neither can be understood or changed on its own.
    """
    out = []
    classes = {c.name: c for c in ast.walk(tree) if isinstance(c, ast.ClassDef)}

    for name, cls in classes.items():
        for fn in _methods(cls):
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and node.attr.startswith("_") \
                        and not node.attr.startswith("__") \
                        and isinstance(node.value, ast.Name) \
                        and node.value.id not in ("self", "cls"):
                    out.append({"smell": "Inappropriate Intimacy",
                                "method": f"{name}.{fn.name}",
                                "accesses": f"{node.value.id}.{node.attr}",
                                "line_number": node.lineno})

    refs = {}
    for name, cls in classes.items():
        named = set()
        for node in ast.walk(cls):
            if isinstance(node, ast.Name) and node.id in classes and node.id != name:
                named.add(node.id)
            elif isinstance(node, ast.arg) and getattr(node.annotation, "id", None) in classes:
                named.add(node.annotation.id)
        refs[name] = named
    for a in classes:
        for b in refs[a]:
            if a < b and a in refs.get(b, set()):
                out.append({"smell": "Inappropriate Intimacy", "classes": [a, b],
                            "kind": "mutual reference",
                            "line_number": classes[a].lineno})
    return out[:5]


CODE_COMMENT = re.compile(
    r"^\s*#\s*(?:(?:def|class|if|for|while|return|import|from|try|except|with|elif|else)\b"
    r"|[A-Za-z_][A-Za-z0-9_]*\s*(?:=[^=]|\())")


def check_comments_smell(src, tree):
    """Commented-out code, and comment volume that substitutes for clarity."""
    out = []
    lines = src.splitlines()
    commented_code = [i + 1 for i, l in enumerate(lines) if CODE_COMMENT.match(l)]
    if commented_code:
        out.append({"smell": "Comments (as smell indicator)", "kind": "commented-out code",
                    "count": len(commented_code), "line_number": commented_code[0]})
    comments = [l for l in lines if l.lstrip().startswith("#")]
    code = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
    if len(comments) >= COMMENT_MIN and code and len(comments) / len(code) >= COMMENT_DENSITY:
        out.append({"smell": "Comments (as smell indicator)", "kind": "density",
                    "comments": len(comments), "code_lines": len(code),
                    "thresholds": {"density": COMMENT_DENSITY, "min": COMMENT_MIN}})
    return out


def check_refused_bequest(tree):
    """A subclass that inherits and then refuses what it inherited."""
    out = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or not cls.bases:
            continue
        base_names = {getattr(b, "id", getattr(b, "attr", "")) for b in cls.bases}
        if base_names <= {"object", "ABC", "Enum", "Exception", ""}:
            continue
        refused = [m.name for m in _methods(cls)
                   if _body_is_stub(m) and not m.name.startswith("__")]
        if refused:
            out.append({"smell": "Refused Bequest", "class": cls.name,
                        "bases": sorted(base_names), "refused_methods": refused,
                        "line_number": cls.lineno})
    return out


def check_speculative_generality(tree):
    """Abstraction built for a future that has not arrived."""
    out = []
    subclassed = set()
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef):
            for b in cls.bases:
                subclassed.add(getattr(b, "id", getattr(b, "attr", "")))
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        stubs = [m.name for m in _methods(cls) if _body_is_stub(m)
                 and not m.name.startswith("__")]
        if len(stubs) >= 2 and cls.name not in subclassed:
            out.append({"smell": "Speculative Generality", "class": cls.name,
                        "unused_hooks": stubs, "line_number": cls.lineno})
    # An "unused parameters" rule lived here and was removed: it fired on 12.8% of
    # files that targeted some other smell, against 21.9% of the ones that targeted
    # this, so it was contributing a base rate rather than a detection.
    return out[:5]


def check_parallel_inheritance(tree):
    """Two hierarchies that must be extended in lockstep."""
    children = defaultdict(list)
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef):
            for b in cls.bases:
                name = getattr(b, "id", getattr(b, "attr", ""))
                if name and name not in ("object", "ABC"):
                    children[name].append(cls.name)

    def stems(names, base):
        out = set()
        for n in names:
            s = n[:-len(base)] if base and n.endswith(base) else n
            s = s.replace(base, "")
            if s:
                out.add(s.lower())
        return out

    hierarchies = [(b, c) for b, c in children.items() if len(c) >= 2]
    results = []
    for i, (b1, c1) in enumerate(hierarchies):
        for b2, c2 in hierarchies[i + 1:]:
            shared = stems(c1, b1) & stems(c2, b2)
            if len(shared) >= 2:
                results.append({"smell": "Parallel Inheritance Hierarchies",
                                "hierarchies": [b1, b2], "mirrored": sorted(shared),
                                "line_number": 0})
    return results


UNDETECTABLE = [
    "Shotgun Surgery",
    "Incomplete Library Class",
    "Alternative Classes with Different Interfaces",
    "Primitive Obsession",
]


def detect_extended_smells(code: str) -> list:
    """Run the additional detectors. Returns [] if the source does not parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found = []
    found += check_data_clumps(tree)
    found += check_message_chains(tree)
    found += check_feature_envy(tree)
    found += check_middle_man(tree)
    found += check_data_class_pure(tree)
    found += check_lazy_class(tree)
    found += check_switch_statements(tree)
    found += check_dead_code(tree)
    found += check_temporary_field(tree)
    found += check_inappropriate_intimacy(tree)
    found += check_comments_smell(code, tree)
    found += check_refused_bequest(tree)
    found += check_speculative_generality(tree)
    found += check_parallel_inheritance(tree)
    return found
