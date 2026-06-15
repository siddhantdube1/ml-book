"""Record GPT-2 internals for the Transformer chapters: logit-lens (Ch 7),
positional-embedding structure (Ch 8), and token-by-token generation (Ch 9).
Writes JSON under apps/llm/public/fixtures/.

    /tmp/gpt2tools/bin/python apps/llm/scripts/record_internals.py
"""

import json
import pathlib

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "public" / "fixtures"

LOGIT_LENS = {
    "capital-france": "The capital of France is",
    "opposite-hot": "The opposite of hot is",
}
GEN_PROMPTS = {
    "once-upon": "Once upon a time, there was a",
    "the-meaning": "The meaning of life is",
}
N_GEN = 20
POS_N = 48


def softmax(x):
    return torch.softmax(x, dim=-1)


def record_logit_lens(model, tok):
    out_dir = OUT / "logitlens"
    out_dir.mkdir(parents=True, exist_ok=True)
    lnf, head = model.transformer.ln_f, model.lm_head
    for fid, context in LOGIT_LENS.items():
        enc = tok(context, return_tensors="pt")
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True)
        hs = res.hidden_states  # tuple(n_layers+1) of [1, seq, 768]
        layers = []
        for L, h in enumerate(hs):
            with torch.no_grad():
                # HF's final hidden state is already post-ln_f; only the earlier
                # ones need the lens's ln_f applied before unembedding.
                last = h[0, -1]
                logits = head(last if L == len(hs) - 1 else lnf(last))
            probs = softmax(logits)
            top = torch.topk(probs, 3)
            layers.append({
                "layer": L,
                "top": [
                    {"text": tok.decode([int(i)]), "prob": round(float(probs[i]), 4)}
                    for i in top.indices.tolist()
                ],
            })
        (out_dir / f"{fid}.json").write_text(json.dumps(
            {"id": fid, "model": "gpt2", "context": context, "layers": layers},
            ensure_ascii=False))
        seq = " -> ".join(l["top"][0]["text"].strip() or "·" for l in layers)
        print(f"  logitlens/{fid}: {seq}")


def record_positional(model):
    out_dir = OUT / "positional"
    out_dir.mkdir(parents=True, exist_ok=True)
    wpe = model.transformer.wpe.weight[:POS_N].detach()
    wn = wpe / wpe.norm(dim=1, keepdim=True)
    sim = (wn @ wn.T).tolist()
    grid = [[round(v, 4) for v in row] for row in sim]
    (out_dir / "gpt2.json").write_text(json.dumps(
        {"model": "gpt2", "n": POS_N, "similarity": grid}, ensure_ascii=False))
    # quick diagnostic: similarity of position 20 to its neighbours
    near = [round(grid[20][j], 2) for j in range(16, 25)]
    print(f"  positional/gpt2: pos20 vs 16..24 -> {near}")


def record_generation(model, tok):
    out_dir = OUT / "generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fid, prompt in GEN_PROMPTS.items():
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        steps = []
        for _ in range(N_GEN):
            with torch.no_grad():
                logits = model(ids).logits[0, -1]
            probs = softmax(logits)
            top = torch.topk(probs, 5)
            chosen = int(top.indices[0])  # greedy
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
        print(f"  generation/{fid}: {prompt}" +
              "".join(s["chosen"] for s in steps))


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    print("logit lens:")
    record_logit_lens(model, tok)
    print("generation:")
    record_generation(model, tok)
    print("done")


if __name__ == "__main__":
    main()
