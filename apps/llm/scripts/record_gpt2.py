"""Offline recorder for the LLM book's mechanism fixtures.

Runs GPT-2 (124M) once and writes real attention maps and next-token
distributions to apps/llm/public/fixtures/ as JSON. These are committed and
served statically; the book never calls a model at runtime for these demos.

Frontier APIs don't expose attention weights or logits, which is why the
mechanism demos use an inspectable open model. See docs/TRILOGY-DESIGN.md §4.

Run from anywhere:
    /tmp/gpt2tools/bin/python apps/llm/scripts/record_gpt2.py
"""

import json
import os
import pathlib

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "gpt2"
ROUND_ATTN = 4
ROUND_LOGIT = 4
TOPK = 40

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "public" / "fixtures"

# Short, legible sentences. The trophy/suitcase line is the classic coreference
# example: late-layer heads send "it" to its referent.
ATTENTION_INPUTS = {
    "trophy-suitcase": "The trophy didn't fit in the suitcase because it was too big.",
    "cat-mat": "The cat sat on the mat.",
    "river-bank": "She sat on the bank by the river.",
}

NEXT_TOKEN_INPUTS = {
    "capital-france": "The capital of France is",
    "once-upon": "Once upon a time, there was a",
    "opposite-hot": "The opposite of hot is",
    "sky-is": "Looking up, the sky is",
}


def decode_tokens(tokenizer, ids):
    return [{"text": tokenizer.decode([int(i)]), "id": int(i)} for i in ids]


def record_attention(model, tokenizer):
    out_dir = OUT / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fid, text in ATTENTION_INPUTS.items():
        enc = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            res = model(**enc, output_attentions=True)
        # attentions: tuple(num_layers) of [batch, heads, q, k]
        attn = res.attentions
        num_layers = len(attn)
        num_heads = attn[0].shape[1]
        grid = [
            [
                [
                    [round(float(v), ROUND_ATTN) for v in attn[l][0, h, q].tolist()]
                    for q in range(attn[l].shape[2])
                ]
                for h in range(num_heads)
            ]
            for l in range(num_layers)
        ]
        fixture = {
            "id": fid,
            "model": MODEL_NAME,
            "text": text,
            "tokens": decode_tokens(tokenizer, enc["input_ids"][0].tolist()),
            "numLayers": num_layers,
            "numHeads": num_heads,
            "attention": grid,
        }
        path = out_dir / f"{fid}.json"
        path.write_text(json.dumps(fixture, ensure_ascii=False))
        print(f"  attention/{fid}.json  ({len(fixture['tokens'])} tokens, "
              f"{num_layers}x{num_heads}, {path.stat().st_size // 1024} KB)")


def record_next_token(model, tokenizer):
    out_dir = OUT / "nexttoken"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fid, context in NEXT_TOKEN_INPUTS.items():
        enc = tokenizer(context, return_tensors="pt")
        with torch.no_grad():
            res = model(**enc)
        logits = res.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, TOPK)
        topk = [
            {
                "text": tokenizer.decode([int(idx)]),
                "id": int(idx),
                "logit": round(float(logits[idx]), ROUND_LOGIT),
                "prob": round(float(probs[idx]), 6),
            }
            for idx in top.indices.tolist()
        ]
        fixture = {
            "id": fid,
            "model": MODEL_NAME,
            "context": context,
            "topk": topk,
        }
        path = out_dir / f"{fid}.json"
        path.write_text(json.dumps(fixture, ensure_ascii=False))
        print(f"  nexttoken/{fid}.json  (top {TOPK}: "
              f"{topk[0]['text']!r} {topk[0]['prob']:.3f})")


def main():
    torch.manual_seed(0)
    print(f"loading {MODEL_NAME} ...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME, attn_implementation="eager"
    )
    model.eval()
    print("recording attention fixtures ...")
    record_attention(model, tokenizer)
    print("recording next-token fixtures ...")
    record_next_token(model, tokenizer)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
