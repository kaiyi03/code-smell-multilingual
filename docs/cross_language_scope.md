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
