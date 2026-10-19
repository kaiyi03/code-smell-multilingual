#!/usr/bin/env python3
"""
The code extractor used for scoring: one rule for every model and language.

The first version, inherited from the Colab notebooks and still the one run at
generation time (pipeline/arc_generate.py), has two blind spots. Both were found by
reading the untouched replies rather than the extracted files:

  * It joins every fenced block in a Python reply, and for Java, C++ and C takes
    the longest. Most second blocks are not part of the answer: a usage line, the
    printed output, or, for StarCoder2-7B, which never stops by itself, a second
    and third copy of the program or an unrelated one. Joined, a repeated program
    reads as Duplicated Code and a usage line as a magic number; the longest block
    of a run-on reply can be a program nobody asked for.
  * It needs a fence. The system prompt asks for no Markdown. A model that obeys
    and still adds a sentence of explanation (Granite in Spanish) has the sentence
    scored as code, so the file fails to parse.

The rule here:

  1. If the reply has a fence, the answer is the first fenced block that is
     untagged or tagged with the target language; a block tagged as something
     else, such as a shell command, is skipped. A block the output limit cut off
     runs to the end of the reply.
  2. With no fence, the answer is the whole reply if it parses. If it does not,
     it is the code from the first line that opens a declaration, trimmed from the
     end until it parses, so prose before and after is dropped.
  3. Otherwise the whole reply, which fails to parse and is scored invalid.

Each file is labelled with the step that produced it (first_block, whole,
fence_free, unparsed), so the effect of each step can be counted rather than
assumed.
"""

import ast
import re
import textwrap
import warnings

# Generated Python is full of invalid escapes like "\w" in ordinary strings.
# Parsing it is the point; the warnings about it are noise.
warnings.filterwarnings("ignore", category=SyntaxWarning)

OPEN = re.compile(r"```[ \t]*([A-Za-z0-9_+\-.#]*)[ \t]*\r?\n")
STRAY_FENCE = re.compile(r"^\s*```[A-Za-z0-9_+\-.#]*\s*$", re.MULTILINE)

# Tags that count as the target language. Untagged blocks count too: most models
# tag their code, and the ones that do not still write it in the language asked.
# C and C++ accept each other's tags because models use them interchangeably.
TAGS = {
    "python": {"", "python", "python3", "py", "py3"},
    "java": {"", "java"},
    "cpp": {"", "cpp", "c++", "cc", "cxx", "hpp", "h", "c"},
    "c": {"", "c", "h", "cpp", "c++"},
}

# Where code starts in a reply with no fence: a line that opens a declaration.
START = {
    "python": re.compile(r"^(def |class |import |from |@|async def )"),
    "java": re.compile(r"^(import |package |public |private |protected |class |interface "
                       r"|enum |abstract |final |record |@)"),
    "cpp": re.compile(r"^(#include|#define|#pragma|using |namespace |class |struct |template"
                      r"|typedef |enum |int |void |auto |static |const |bool |double |float "
                      r"|char |long |unsigned |std::)"),
    "c": re.compile(r"^(#include|#define|#pragma|typedef |struct |enum |int |void |static "
                    r"|const |char |double |float |long |unsigned |bool |size_t )"),
}
MIN_LINES = 3


def clean(code):
    """The first version's normalisation, unchanged: stray fences, indent, trailing space."""
    if not code:
        return ""
    code = STRAY_FENCE.sub("", code)
    code = textwrap.dedent(code)
    return "\n".join(l.rstrip() for l in code.splitlines()).strip("\n").strip()


def blocks(raw):
    """Fenced blocks in order, as (tag, body). An unclosed last block runs to the end."""
    out, pos = [], 0
    while True:
        m = OPEN.search(raw, pos)
        if not m:
            return out
        end = raw.find("```", m.end())
        if end < 0:
            out.append((m.group(1).lower(), raw[m.end():]))
            return out
        out.append((m.group(1).lower(), raw[m.end():end]))
        pos = end + 3


_PARSERS = {}


def parses(src, plang):
    if not src or not src.strip():
        return False
    if plang == "python":
        try:
            ast.parse(src)
            return True
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            # MemoryError is the parser stack overflowing on pathological nesting;
            # ValueError is a null byte. Both mean "does not parse", not "crash".
            return False
    if plang not in _PARSERS:
        from detector.cross_language import parser_for
        _PARSERS[plang] = parser_for(plang)
    tree = _PARSERS[plang].parse(src.encode("utf-8", "replace"))
    return not tree.root_node.has_error


def fence_free(raw, plang):
    """Code from the first declaration line, trimmed from the end until it parses."""
    lines = raw.splitlines()
    start = next((i for i, l in enumerate(lines) if START[plang].match(l)), None)
    if start is None:
        return None
    body = lines[start:]
    for end in range(len(body), 0, -1):
        src = clean("\n".join(body[:end]))
        if len(src.splitlines()) >= MIN_LINES and parses(src, plang):
            return src
    return None


def extract(raw, plang="python"):
    """Return (code, step) for one reply. code ends with a newline unless empty."""
    raw = raw or ""
    found = [(t, clean(b)) for t, b in blocks(raw)]
    found = [(t, b) for t, b in found if b]
    if found:
        target = [b for t, b in found if t in TAGS[plang]]
        return (target[0] if target else found[0][1]) + "\n", "first_block"
    whole = clean(raw)
    if parses(whole, plang):
        return whole + "\n", "whole"
    recovered = fence_free(raw, plang)
    if recovered:
        return recovered + "\n", "fence_free"
    return (whole + "\n" if whole else ""), "unparsed"
