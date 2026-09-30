# Cross-language scope: which smells survive the move to Java, C and C++

The study is now scoped to four programming languages — Python, Java, C, C++ —
and four natural languages. This note records which of the 25 targeted smells can
actually be measured in each, because the answer is not "all of them", and a matrix
with silently empty cells is worse than one that says why they are empty.

## The problem in one line

Eight of the 25 smells are defined in terms of classes. **C has no classes.**

## Smell by language

| Smell | Python | Java | C++ | C |
|---|:--:|:--:|:--:|:--:|
| Long Method | ✅ | ✅ | ✅ | ✅ |
| Long Parameter List | ✅ | ✅ | ✅ | ✅ |
| Deep Nesting | ✅ | ✅ | ✅ | ✅ |
| Duplicated Code | ✅ | ✅ | ✅ | ✅ |
| Magic Numbers/Strings | ✅ | ✅ | ✅ | ✅ |
| Global State | ✅ | ✅ | ✅ | ✅ |
| Dead Code | ✅ | ✅ | ✅ | ✅ |
| Switch Statements | ✅ | ✅ | ✅ | ✅ |
| Data Clumps | ✅ | ✅ | ✅ | ✅ |
| Comments (as smell indicator) | ✅ | ✅ | ✅ | ✅ |
| Primitive Obsession | ✅ | ✅ | ✅ | ✅ |
| Message Chains | ✅ | ✅ | ✅ | ⚠️ struct chains `a->b->c` only |
| Shotgun Surgery | ❌ needs change history in every language |||| 
| God Class / Large Class | ✅ | ✅ | ✅ | ❌ |
| Data Class | ✅ | ✅ | ✅ | ⚠️ a bare `struct` is idiomatic C, not a smell |
| Lazy Class | ✅ | ✅ | ✅ | ❌ |
| Middle Man | ✅ | ✅ | ✅ | ❌ |
| Feature Envy | ✅ | ✅ | ✅ | ⚠️ function favouring one struct's fields |
| Temporary Field | ✅ | ✅ | ✅ | ❌ |
| Inappropriate Intimacy | ✅ | ✅ | ✅ | ❌ |
| Refused Bequest | ✅ | ✅ | ✅ | ❌ |
| Parallel Inheritance | ✅ | ✅ | ✅ | ❌ |
| Speculative Generality | ✅ | ✅ | ✅ | ⚠️ unused params / unreachable hooks |
| Alternative Classes, Diff. Interfaces | ❌ needs semantic equivalence in every language ||||
| Incomplete Library Class | ❌ needs the library's intent in every language ||||

**Roughly: 21 of 25 measurable in Python, Java and C++; about 13 in C.**

## What is actually implemented, as against measurable in principle

The table above is the scoping analysis: which smells *could* be decided in each
language by someone willing to write the checks. It is not what
`detector/cross_language.py` decides today, and the gap is large enough that it
has to be stated rather than left for a reader to discover.

**The implemented cross-language detector decides 12 smells: Long Method, Long
Parameter List, Deep Nesting, Magic Numbers/Strings, Switch Statements, God Class,
Duplicated Code, Global State, Message Chains, Comments-as-smell, Data Class and
Lazy Class.** The last three apply to Python, Java and C++ but not C, which has no
classes and where a bare struct is idiomatic rather than a smell. The authority is
`APPLICABLE` in that module, and `OUT_OF_REACH` beside it lists the other 13 with
the reason each is out of reach; this paragraph describes them and the module is
what runs.

A 13th was implemented and withdrawn. Dead Code was measured both ways it is
usually defined, as statements after a return and as internal functions nothing
calls, and it fired about as often on files that asked for some other smell:
lift came out at +1.4, +0.6, -3.9 and -0.3 across the four languages. The Python
`ast` detector reaches only +7.3 on the same smell, so two independent instruments
both fail to separate it. That is evidence about the smell rather than about either
detector, since "nobody uses this" is a property of a codebase and not of one file.

Those six are the ones a syntax tree answers directly, by counting nodes against a
threshold. The other fifteen in the table are decidable in principle but need
cross-procedural reasoning — which fields a method touches, whether a class
forwards rather than does, whether two bodies are near-duplicates — and the Python
implementations in `detector/extended_smells.py` lean on Python-specific structure
that does not transfer by simply swapping grammars.

What that costs the comparison: **114 of the 426 prompts target a smell the
detector can decide, and 94 of C's 292.** Induction is reported over those files
only. The remaining prompts are scored for syntax validity like any other file but
contribute nothing to the induction figure, and are recorded as blank rather than
as misses — counting "no detector" as "the model failed" would hand every language
a failure rate proportional to how many prompts it was asked.

So the cross-language arm answers a narrower question than the Python arm, on
about a quarter of the prompt set. Extending the detector past these six would
widen the arm rather than change its design, and is the obvious next piece of work.

## What follows

1. **C is not a fourth column of the same table.** Any "smells per language"
   comparison including C must be restricted to the ~13 smells that exist in all
   four, or C will look cleaner purely because two thirds of the checks cannot fire.
   That is the same artefact as the syntax-validity confound already found on the
   natural-language side: an apparent quality difference that is really a
   measurement difference.

2. **The prompt set needs porting, not translating.** `prompts_core.json` names
   Python APIs and asks for Python idioms. A God Class prompt has no C equivalent,
   so the C arm needs its own prompt subset rather than a rewrite of all 426.

3. **One detector, not four.** Re-implementing the same thresholded definitions
   over *tree-sitter* grammars keeps one definition of "long method" across all four
   languages. Using PMD for Java and clang-tidy for C/C++ would give three tools
   with three different definitions and numbers that cannot be compared — which
   would undercut the cross-language comparison the study exists to make.

## Suggested order

Java and C++ first: both are class-based, so all 21 detectors port and the
comparison with Python is like-for-like. Then C as a deliberately reduced arm,
reported on the common subset and labelled as such.
