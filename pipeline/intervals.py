#!/usr/bin/env python3
"""
95% intervals for the headline rates, resampling whole prompts.

Files are not independent. Every prompt is answered by every model in every
language, so a hard prompt drags down a whole row of files at once. Resampling
files would treat those as separate evidence and give intervals far too narrow.
Resampling prompts keeps each prompt's files together, which is the honest unit.

This is a summary, not the final analysis: a mixed-effects model with model and
prompt as random effects is the proper treatment, and is listed as next work.
"""

import random
from collections import defaultdict

RESAMPLES = 500
SEED = 7


def _rates(rows, key, value):
    """Validity over all files; induction over parsing files with a decidable target."""
    v = [r for r in rows if r[key] == value]
    if not v:
        return None, None
    valid = 100.0 * sum(int(r["syntax_ok"]) for r in v) / len(v)
    ind = [r for r in v if r["syntax_ok"] == "1" and r["target_covered"] == "1"]
    induction = (100.0 * sum(int(r["target_hit"]) for r in ind) / len(ind)
                 if ind else None)
    return valid, induction


def by_group(rows, key, groups):
    """{group: (valid_lo, valid_hi, induction_lo, induction_hi)} from prompt resamples."""
    byp = defaultdict(list)
    for r in rows:
        byp[r["prompt_id"]].append(r)
    prompts = sorted(byp)
    rng = random.Random(SEED)
    draws = defaultdict(lambda: ([], []))
    for _ in range(RESAMPLES):
        sample = [r for p in rng.choices(prompts, k=len(prompts)) for r in byp[p]]
        for g in groups:
            a, b = _rates(sample, key, g)
            if a is not None:
                draws[g][0].append(a)
            if b is not None:
                draws[g][1].append(b)
    lo, hi = int(0.025 * RESAMPLES), int(0.975 * RESAMPLES) - 1
    out = {}
    for g in groups:
        va, vb = sorted(draws[g][0]), sorted(draws[g][1])
        out[g] = ((va[lo], va[hi]) if va else (None, None),
                  (vb[lo], vb[hi]) if vb else (None, None))
    return out
