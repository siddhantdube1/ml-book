"""Part III fixtures: per-token surprisal (Ch 10) and base-model instruction
completions (Ch 12), from real GPT-2. Writes under apps/llm/public/fixtures/.

    /tmp/gpt2tools/bin/python apps/llm/scripts/record_part3.py
"""

import json
import math
import pathlib

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "public" / "fixtures"

SURPRISAL_TEXT = {
    "france": "The capital of France is Paris.",
    "ghost": "She opened the door and saw a ghost.",
    "water": "Water is made of hydrogen and oxygen.",
}

# Instruction-shaped prompts a base model has never been tuned to obey.
INSTRUCTION_PROMPTS = {
    "instr-capital": "Question: What is the capital of France?\nAnswer:",
    "instr-haiku": "Write a haiku about the ocean:\n",
}
N_GEN = 24


def record_surprisal(model, tok):
    out_dir = OUT / "surprisal"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fid, text in SURPRISAL_TEXT.items():
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"][0]
        with torch.no_grad():
            logits = model(**enc).logits[0]  # [seq, vocab]
        probs = torch.softmax(logits, dim=-1)
        toks = []
        for i, tid in enumerate(ids.tolist()):
            if i == 0:
                toks.append({"text": tok.decode([tid]), "prob": None, "surprisal": None})
            else:
                p = float(probs[i - 1, tid])  # prob assigned to the actual token i
                toks.append({
                    "text": tok.decode([tid]),
                    "prob": round(p, 4),
                    "surprisal": round(-math.log2(max(p, 1e-12)), 2),  # bits
                })
        (out_dir / f"{fid}.json").write_text(json.dumps(
            {"id": fid, "model": "gpt2", "text": text, "tokens": toks},
            ensure_ascii=False))
        worst = max((t for t in toks if t["surprisal"]), key=lambda t: t["surprisal"])
        print(f"  surprisal/{fid}: most surprising token = {worst['text']!r} "
              f"({worst['surprisal']} bits)")


def record_instructions(model, tok):
    out_dir = OUT / "generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fid, prompt in INSTRUCTION_PROMPTS.items():
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        steps = []
        for _ in range(N_GEN):
            with torch.no_grad():
                logits = model(ids).logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, 5)
            chosen = int(top.indices[0])
            steps.append({
                "chosen": tok.decode([chosen]),
                "top": [
                    {"text": tok.decode([int(i)]), "prob": round(float(probs[i]), 4)}
                    for i in top.indices.tolist()
                ],
            })
            ids = torch.cat([ids, torch.tensor([[chosen]])], dim=1)
        (out_dir / f"{fid}.json").write_text(json.dumps(
            {"id": fid, "model": "gpt2", "prompt": prompt, "steps": steps},
            ensure_ascii=False))
        print(f"  generation/{fid}: {prompt!r} ->"
              + "".join(s["chosen"] for s in steps).replace("\n", " "))


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    print("surprisal:")
    record_surprisal(model, tok)
    print("instruction completions (base model):")
    record_instructions(model, tok)
    print("done")


if __name__ == "__main__":
    main()
