#!/usr/bin/env python3
"""
Generation for the ARC (SLURM) cluster.

The original runs were done in Colab notebooks, which is why they stopped where
they did: sessions expire, so the multilingual arms were capped at
MAX_PER_CATEGORY=1 -- 75 of the 426 prompts. On a batch scheduler there is no
session to lose, so the full set is reachable.

This reproduces the notebook's generation settings exactly, because the outputs
have to sit alongside the ones already on Drive:

  * greedy decoding (do_sample=False), max_new_tokens 2048, bfloat16
  * the model's own chat template, with the per-language system prompt
    hand-translated in the notebook rather than machine-translated
  * the parse-validated extractor from fix_extraction.ipynb
  * the same output layout: <root>/<model>/<lang>/code/*.py + results.jsonl

Resume-safe: a prompt already present in results.jsonl is skipped, so a job that
hits its wall clock can simply be resubmitted.

    python -m pipeline.arc_generate --model qwen2.5-coder-1.5b --lang en --limit 4
    python -m pipeline.arc_generate --model qwen2.5-coder-1.5b --lang es
"""

import argparse
import ast
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.model_registry import MODELS, SYSTEM_PROMPT

DATASET = PROJECT_ROOT / "dataset" / "prompts_core.json"

# Hand-translated in the Colab notebook rather than machine-translated: these
# define the output contract, and a translation slip here would change what every
# model in that language was asked to do.
SYSTEM_PROMPTS = {
    "en": SYSTEM_PROMPT,
    "zh": "你是一位专业的 Python 开发者。请根据给定的需求生成简洁、完整的 Python 代码。"
          "只输出 Python 代码，不要包含任何解释或 Markdown 格式。",
    "es": "Eres un desarrollador experto en Python. Genera código Python limpio y "
          "completo según los requisitos dados. Devuelve únicamente el código Python, "
          "sin explicaciones ni formato Markdown.",
    "fr": "Tu es un développeur Python expert. Génère un code Python propre et complet "
          "à partir des exigences données. Renvoie uniquement le code Python, sans "
          "explications ni mise en forme Markdown.",
}

# ---- extractor, from fix_extraction.ipynb (parse-validated) -----------------
FENCE_BLOCK = re.compile(r"```[ \t]*[A-Za-z0-9_+\-.]*[ \t]*\r?\n(.*?)```", re.DOTALL)
FENCE_TAIL = re.compile(r"```[ \t]*[A-Za-z0-9_+\-.]*[ \t]*\r?\n(.*)\Z", re.DOTALL)
STRAY_FENCE = re.compile(r"^\s*```[A-Za-z0-9_+\-.]*\s*$", re.MULTILINE)


def _clean(code):
    if not code:
        return ""
    code = STRAY_FENCE.sub("", code)
    code = textwrap.dedent(code)
    return "\n".join(l.rstrip() for l in code.splitlines()).strip("\n").strip()


def _parses(code):
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_python_code(response):
    if not response:
        return ""
    cands = []
    closed = FENCE_BLOCK.findall(response)
    if closed:
        cands.append("\n\n".join(b.strip("\n") for b in closed))
        cands.extend(closed)
    tail = FENCE_TAIL.search(response)
    if tail:
        cands.append(tail.group(1))
    cands.append(response)
    cleaned = [_clean(c) for c in cands]
    for c in cleaned:
        if _parses(c):
            return c + "\n"
    for c in cleaned:
        if c:
            return c + "\n"
    return ""


# ---------------------------------------------------------------------------

def load_prompts(lang, cache_dir):
    if lang == "en":
        with open(DATASET, encoding="utf-8") as f:
            return [dict(p, prompt_en=p["prompt"], lang="en") for p in json.load(f)]
    path = Path(cache_dir) / f"prompts_{lang}.json"
    if not path.exists():
        sys.exit(f"no translated prompts at {path}\n"
                 f"  run: python -m pipeline.translate_prompts --lang {lang}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry id, e.g. qwen2.5-coder-7b")
    ap.add_argument("--lang", default="en", choices=list(SYSTEM_PROMPTS))
    ap.add_argument("--limit", type=int, help="first N prompts (smoke test)")
    ap.add_argument("--out-root", default=os.environ.get("OUT_ROOT", "outputs_arc"))
    ap.add_argument("--prompt-cache", default=os.environ.get("PROMPT_CACHE", "_prompts"))
    ap.add_argument("--max-new", type=int, default=int(os.environ.get("GEN_MAX_NEW", "2048")))
    args = ap.parse_args()

    cfg = next((m for m in MODELS if m["id"] == args.model), None)
    if cfg is None:
        sys.exit(f"unknown model {args.model}; registry has "
                 f"{', '.join(m['id'] for m in MODELS)}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # DeepSeek-Coder-V2-Lite ships its own modelling code via trust_remote_code,
    # written against a transformers that still exported is_torch_fx_available.
    # Newer releases dropped it, so the import fails before the model can load.
    # The symbol only gates an optional tracing path, so a False stub is safe.
    import transformers.utils.import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: False

    outdir = Path(args.out_root) / args.model / args.lang
    (outdir / "code").mkdir(parents=True, exist_ok=True)
    results = outdir / "results.jsonl"

    done = set()
    if results.exists():
        with open(results, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["prompt_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    prompts = load_prompts(args.lang, args.prompt_cache)
    if args.limit:
        prompts = prompts[:args.limit]
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{args.model} / {args.lang}: {len(todo)} to do, {len(done)} already present",
          flush=True)
    if not todo:
        return

    print(f"loading {cfg['hf_model_id']}", flush=True)
    # Yi-Coder ships only a sentencepiece tokenizer.model with no tokenizer.json,
    # and transformers 5.x misidentifies it as a tiktoken file, then fails parsing
    # it. Loading the slow tokenizer reads it with sentencepiece as intended.
    try:
        tok = AutoTokenizer.from_pretrained(cfg["hf_model_id"], trust_remote_code=True)
    except Exception as e:
        print(f"  fast tokenizer failed ({type(e).__name__}), retrying slow", flush=True)
        tok = AutoTokenizer.from_pretrained(cfg["hf_model_id"], trust_remote_code=True,
                                            use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_model_id"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    system = SYSTEM_PROMPTS[args.lang]
    started = time.time()
    with open(results, "a", encoding="utf-8") as out:
        for i, p in enumerate(todo, 1):
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": p["prompt"]}]
            try:
                text = tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
            except Exception:
                # Some base-ish checkpoints ship no chat template.
                text = f"{system}\n\n{p['prompt']}\n"
            enc = tok(text, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=False, temperature=None, top_p=None,
                                     pad_token_id=tok.pad_token_id)
            elapsed = time.time() - t0
            raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            code = extract_python_code(raw)

            (outdir / "code" / f"{p['id']}.py").write_text(code, encoding="utf-8")
            out.write(json.dumps({
                "prompt_id": p["id"], "model_id": args.model, "lang": args.lang,
                "prompt": p["prompt"], "prompt_en": p.get("prompt_en", p["prompt"]),
                "code_smells_target": p.get("code_smells", []),
                "complexity": p.get("complexity"), "domain": p.get("domain"),
                "raw_response": raw, "extracted_code": code,
                "input_tokens": int(enc["input_ids"].shape[1]),
                "output_tokens": int(gen.shape[1] - enc["input_ids"].shape[1]),
                "generation_time_s": round(elapsed, 2),
            }, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - started) / i
                print(f"  {i}/{len(todo)}  {rate:.1f}s/prompt  "
                      f"eta {rate * (len(todo) - i) / 60:.0f}m", flush=True)

    print(f"done in {(time.time() - started) / 60:.1f}m -> {outdir}")


if __name__ == "__main__":
    main()
