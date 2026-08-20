#!/usr/bin/env python3
"""
Translate the 426 prompts into Chinese, Spanish and French.

Only 75 of the 426 were ever translated -- the Colab notebook ran with
MAX_PER_CATEGORY=1, and the cached prompt files on Drive hold that pilot subset.
Every non-English generation run is blocked on this until the rest exist.

The method reproduces the notebook's exactly, because the 75 already generated
have to remain comparable with the 351 this adds: facebook/nllb-200-distilled-600M,
source eng_Latn, four beams, max_length 512, float32, batches of 16. Deterministic
decoding, so re-running gives the same text.

Identifiers are deliberately NOT protected from the translator here. NLLB rewrites
19-30% of the API names the prompts specify (place_order -> lugar_orden), and doing
that consistently would be a different experimental design -- one worth choosing on
purpose rather than changing halfway through a corpus. Measured within each
language, the drift does not predict syntax failure or induction rate, so this
preserves comparability at no known cost. --mask implements the alternative when
that decision is made.

    python -m pipeline.translate_prompts --lang es --lang fr --lang zh
    python -m pipeline.translate_prompts --lang es --limit 4     # smoke test
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET = PROJECT_ROOT / "dataset" / "prompts_core.json"
NLLB_CODE = {"zh": "zho_Hans", "es": "spa_Latn", "fr": "fra_Latn"}
MODEL_ID = "facebook/nllb-200-distilled-600M"

# snake_case names, CamelCase names, and call signatures like foo(a, b, c)
CODE_SPAN = re.compile(r"\b[a-z]+_[a-z_0-9]+\b\s*\([^)]*\)"
                       r"|\b[a-z]+_[a-z_0-9]+\b"
                       r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")


def mask(text):
    """Replace code spans with placeholders NLLB will copy through untouched."""
    table = {}

    def sub(m):
        key = f"Q{len(table)}Z"
        table[key] = m.group(0)
        return key

    return CODE_SPAN.sub(sub, text), table


def unmask(text, table):
    for key, original in table.items():
        text = text.replace(key, original)
    return text, all(k in text or v in text for k, v in table.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", action="append", required=True, choices=list(NLLB_CODE))
    ap.add_argument("--limit", type=int, help="first N prompts (smoke test)")
    ap.add_argument("--out", default=os.environ.get("PROMPT_CACHE", "_prompts"))
    ap.add_argument("--mask", action="store_true",
                    help="protect code identifiers from the translator")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    with open(DATASET, encoding="utf-8") as f:
        prompts = json.load(f)
    if args.limit:
        prompts = prompts[:args.limit]
    print(f"{len(prompts)} prompts, masking {'on' if args.mask else 'off'}", flush=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32).to(device)
    model.eval()
    tok.src_lang = "eng_Latn"

    for lang in args.lang:
        path = outdir / f"prompts_{lang}.json"
        existing = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                existing = {d["id"]: d for d in json.load(f)}
            print(f"[{lang}] {len(existing)} already translated", flush=True)

        todo = [p for p in prompts if p["id"] not in existing]
        if not todo:
            print(f"[{lang}] nothing to do")
            continue

        texts, tables = [], []
        for p in todo:
            if args.mask:
                t, table = mask(p["prompt"])
            else:
                t, table = p["prompt"], {}
            texts.append(t)
            tables.append(table)

        bos = tok.convert_tokens_to_ids(NLLB_CODE[lang])
        out, broken = [], 0
        for i in range(0, len(texts), args.batch):
            enc = tok(texts[i:i + args.batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos,
                                     max_length=512, num_beams=4)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
            print(f"  [{lang}] {min(i + args.batch, len(texts))}/{len(texts)}", flush=True)

        for p, translated, table in zip(todo, out, tables):
            if table:
                translated, ok = unmask(translated, table)
                broken += not ok
            existing[p["id"]] = dict(p, prompt=translated, prompt_en=p["prompt"],
                                     lang=lang)

        ordered = [existing[p["id"]] for p in prompts if p["id"] in existing]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)
        print(f"[{lang}] wrote {len(ordered)} -> {path}"
              + (f"  ({broken} lost a placeholder)" if broken else ""), flush=True)
        if broken:
            print(f"  !! {broken} prompts came back missing a placeholder, so the "
                  f"masking format is not surviving NLLB. Do not use these.",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
