"""
Code Smell Detector - Programmatic API

Adapted from https://github.com/hiimkimchi/Code-Smell-Detector
Original by hiimkimchi. Extended with additional heuristic detectors.
"""

import re
import ast
import textwrap
from itertools import combinations
from typing import List, Dict, Tuple, Optional

# --- Thresholds ---
MAX_LOC = 15
MAX_PARAMS = 3
JACCARD_THRESHOLD = 0.75
MAX_NESTING_DEPTH = 4
MAX_CLASS_METHODS = 10
MAX_CLASS_LINES = 200
MAGIC_NUMBER_PATTERN = re.compile(r'(?<!["\'\w])(?<!\.\d)(?:(?<!\d)\d{2,}(?:\.\d+)?|(?<!\d)\d+\.\d+)(?!["\'\w\.])')


# ==============================================================
# Core detectors (adapted from hiimkimchi/Code-Smell-Detector)
# ==============================================================

def is_method(contents: List[str], index: int) -> bool:
    line = contents[index].lstrip()
    return line.startswith("def ")


def get_method_name(line: str) -> Optional[str]:
    line = line.lstrip()
    match = re.match(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', line)
    return match.group(1) if match else None


def collect_boundaries(contents: List[str]) -> List[Tuple[int, int, int]]:
    boundaries, current_start, blank_count = [], None, 0
    for index in range(len(contents)):
        if not is_method(contents, index):
            if current_start is not None and contents[index].strip() == '':
                blank_count += 1
            continue
        if current_start is None:
            current_start, blank_count = index, 0
        else:
            boundaries.append((current_start, index, blank_count))
            current_start, blank_count = index, 0
    if current_start is not None:
        boundaries.append((current_start, len(contents), blank_count))
    return boundaries


def check_long_methods(contents: List[str]) -> List[Dict]:
    results = []
    boundaries = collect_boundaries(contents)
    for start, end, blanks in boundaries:
        name = get_method_name(contents[start])
        loc = end - start - blanks
        if loc > MAX_LOC and name:
            results.append({
                "smell": "Long Method",
                "function": name,
                "lines": loc,
                "threshold": MAX_LOC,
                "line_number": start + 1
            })
    return results


def check_long_parameter_lists(contents: List[str]) -> List[Dict]:
    results = []
    for i, line in enumerate(contents):
        if not line.lstrip().startswith("def "):
            continue
        # Handle multi-line defs
        full_line = line
        j = i
        while ')' not in full_line and j + 1 < len(contents):
            j += 1
            full_line += contents[j]
        name = get_method_name(full_line)
        param_str = full_line[full_line.find("(") + 1:full_line.rfind(")")]
        params = [p.strip() for p in param_str.split(",") if p.strip()]
        # Remove 'self' and 'cls'
        params = [p for p in params if p not in ("self", "cls")]
        if len(params) > MAX_PARAMS and name:
            results.append({
                "smell": "Long Parameter List",
                "function": name,
                "param_count": len(params),
                "threshold": MAX_PARAMS,
                "line_number": i + 1
            })
    return results


def jaccard_sim(method1: List[str], method2: List[str]) -> float:
    text1 = "".join(method1)
    text2 = "".join(method2)
    set1 = set(text1)
    set2 = set(text2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 1.0


def collect_methods(contents: List[str]) -> List[List[str]]:
    all_methods, current = [], []
    for index in range(len(contents)):
        if is_method(contents, index):
            if current:
                all_methods.append(current)
            current = [contents[index]]
        elif current:
            current.append(contents[index])
    if current:
        all_methods.append(current)
    return all_methods


def check_duplicated_code(contents: List[str]) -> List[Dict]:
    results = []
    all_methods = collect_methods(contents)
    for m1, m2 in combinations(all_methods, 2):
        sim = jaccard_sim(m1, m2)
        if sim >= JACCARD_THRESHOLD:
            name1 = get_method_name(m1[0]) or "unknown"
            name2 = get_method_name(m2[0]) or "unknown"
            results.append({
                "smell": "Duplicated Code",
                "methods": [name1, name2],
                "jaccard_similarity": round(sim, 3),
                "threshold": JACCARD_THRESHOLD
            })
    return results


# ==============================================================
# Extended heuristic detectors
# ==============================================================

def check_deep_nesting(contents: List[str]) -> List[Dict]:
    results = []
    current_func = None
    max_depth_in_func = 0
    func_start_line = 0

    for i, line in enumerate(contents):
        stripped = line.lstrip()
        if stripped.startswith("def "):
            if current_func and max_depth_in_func > MAX_NESTING_DEPTH:
                results.append({
                    "smell": "Deep Nesting",
                    "function": current_func,
                    "max_depth": max_depth_in_func,
                    "threshold": MAX_NESTING_DEPTH,
                    "line_number": func_start_line
                })
            current_func = get_method_name(line)
            func_start_line = i + 1
            max_depth_in_func = 0
            continue

        if stripped == '' or stripped.startswith('#'):
            continue

        indent = len(line) - len(line.lstrip())
        # Rough nesting estimate: each 4-space indent = 1 level
        depth = indent // 4
        if depth > max_depth_in_func:
            max_depth_in_func = depth

    # Check last function
    if current_func and max_depth_in_func > MAX_NESTING_DEPTH:
        results.append({
            "smell": "Deep Nesting",
            "function": current_func,
            "max_depth": max_depth_in_func,
            "threshold": MAX_NESTING_DEPTH,
            "line_number": func_start_line
        })
    return results


def check_god_class(source: str) -> List[Dict]:
    results = []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return results

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            end_line = max((getattr(n, 'end_lineno', n.lineno) for n in ast.walk(node) if hasattr(n, 'lineno')), default=node.lineno)
            class_lines = end_line - node.lineno + 1
            if len(methods) > MAX_CLASS_METHODS or class_lines > MAX_CLASS_LINES:
                results.append({
                    "smell": "God Class",
                    "class": node.name,
                    "method_count": len(methods),
                    "class_lines": class_lines,
                    "thresholds": {"max_methods": MAX_CLASS_METHODS, "max_lines": MAX_CLASS_LINES},
                    "line_number": node.lineno
                })
    return results


def check_magic_numbers(contents: List[str]) -> List[Dict]:
    results = []
    # Common acceptable numbers
    acceptable = {'0', '1', '2', '0.0', '1.0', '100'}

    for i, line in enumerate(contents):
        stripped = line.lstrip()
        if stripped.startswith('#') or stripped.startswith(('"""', "'''")):
            continue
        matches = MAGIC_NUMBER_PATTERN.findall(stripped)
        for m in matches:
            if m not in acceptable:
                results.append({
                    "smell": "Magic Number",
                    "value": m,
                    "line_number": i + 1,
                    "line": stripped.strip()
                })
    return results


def check_global_state(contents: List[str]) -> List[Dict]:
    results = []
    try:
        tree = ast.parse("".join(contents))
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return results

    # Find module-level assignments (not inside functions/classes)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Skip constants (ALL_CAPS)
                    if not name.isupper():
                        results.append({
                            "smell": "Global State",
                            "variable": name,
                            "line_number": node.lineno
                        })

    # Check for 'global' keyword usage
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            for name in node.names:
                results.append({
                    "smell": "Global State (global keyword)",
                    "variable": name,
                    "line_number": node.lineno
                })
    return results


def check_data_class(source: str) -> List[Dict]:
    results = []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return results

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not methods:
            continue
        method_names = {m.name for m in methods}
        # Check if class has only __init__ and property-like / getter-setter methods
        non_dunder = {n for n in method_names if not n.startswith('_')}
        getter_setter_patterns = sum(
            1 for n in non_dunder
            if n.startswith(('get_', 'set_', 'to_', 'from_'))
        )
        if len(methods) >= 2 and getter_setter_patterns >= len(non_dunder) * 0.6:
            results.append({
                "smell": "Data Class",
                "class": node.name,
                "methods": list(method_names),
                "getter_setter_ratio": round(getter_setter_patterns / max(len(non_dunder), 1), 2),
                "line_number": node.lineno
            })
    return results


# ==============================================================
# Main detection API
# ==============================================================

def detect_all_smells(code: str) -> Dict:
    """Run all smell detectors on a code string. Returns structured results."""
    # Ensure each line ends with newline for compatibility
    contents = [(line if line.endswith('\n') else line + '\n') for line in code.split('\n')]

    all_smells = []
    all_smells.extend(check_long_methods(contents))
    all_smells.extend(check_long_parameter_lists(contents))
    all_smells.extend(check_duplicated_code(contents))
    all_smells.extend(check_deep_nesting(contents))
    all_smells.extend(check_god_class(code))
    all_smells.extend(check_magic_numbers(contents))
    all_smells.extend(check_global_state(contents))
    all_smells.extend(check_data_class(code))

    # Categorize
    detected_types = list(set(s["smell"] for s in all_smells))

    return {
        "total_smells": len(all_smells),
        "smell_types_detected": detected_types,
        "smells": all_smells
    }
