"""Record GPT-2 token embeddings for a curated word set: 2D projection +
nearest neighbours (+ analogy diagnostics). Writes
apps/llm/public/fixtures/embeddings/words.json.

Nearest neighbours are computed in the FULL 768-d space (where they are
meaningful); the 2D coordinates are a PCA projection for display only.

    /tmp/gpt2tools/bin/python apps/llm/scripts/record_embeddings.py
"""

import json
import pathlib

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "public" / "fixtures" / "embeddings"

# Curated, grouped so structure is legible. Proper nouns capitalised.
GROUPS = {
    "royalty": ["king", "queen", "prince", "princess"],
    "people": ["man", "woman", "boy", "girl", "father", "mother"],
    "country": ["France", "Germany", "Italy", "Spain", "Japan", "China"],
    "capital": ["Paris", "Berlin", "Rome", "Madrid", "Tokyo", "London"],
    "animal": ["cat", "dog", "lion", "tiger", "horse", "mouse", "bird", "fish"],
    "colour": ["red", "blue", "green", "yellow", "black", "white"],
    "number": ["one", "two", "three", "four", "five", "six"],
    "verb": ["walk", "run", "swim", "eat", "sleep", "jump"],
}

ANALOGIES = [
    ("man", "king", "woman", "queen"),       # king - man + woman ?= queen
    ("France", "Paris", "Italy", "Rome"),
    ("France", "Paris", "Japan", "Tokyo"),
    ("boy", "man", "girl", "woman"),
]


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    wte = model.transformer.wte.weight.detach().numpy()  # [vocab, 768]

    words, vecs, groups, dropped = [], [], [], []
    for group, items in GROUPS.items():
        for w in items:
            ids = tok.encode(" " + w)
            if len(ids) != 1:
                dropped.append((w, len(ids)))
                continue
            words.append(w)
            groups.append(group)
            vecs.append(wte[ids[0]])
    X = np.array(vecs)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)  # unit vectors -> cosine

    # Nearest neighbours within the set (cosine), top 5 excluding self.
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -1)
    nbrs = [
        [words[j] for j in sims[i].argsort()[::-1][:5]] for i in range(len(words))
    ]

    # PCA to 2D for display (numpy SVD).
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0))  # -> [0,1]

    idx = {w: i for i, w in enumerate(words)}

    def analogy(a, b, c):
        # b - a + c, nearest word (cosine), excluding the three inputs.
        v = X[idx[b]] - X[idx[a]] + X[idx[c]]
        v = v / np.linalg.norm(v)
        s = Xn @ v
        order = s.argsort()[::-1]
        out = [words[j] for j in order if words[j] not in (a, b, c)][:5]
        return out

    print("dropped (multi-token):", dropped)
    print("\nnearest neighbours (full space):")
    for probe in ["king", "Paris", "cat", "three", "run"]:
        if probe in idx:
            print(f"  {probe:8} -> {nbrs[idx[probe]]}")
    print("\nanalogy a:b :: c:? (want the 4th):")
    analogy_out = []
    for a, b, c, want in ANALOGIES:
        if all(w in idx for w in (a, b, c)):
            top = analogy(a, b, c)
            hit = want in top
            print(f"  {a}:{b} :: {c}:?  -> {top}   want {want}  {'HIT' if hit else 'miss'}")
            analogy_out.append({"a": a, "b": b, "c": c, "want": want,
                                "predicted": top, "hit": hit})

    OUT.mkdir(parents=True, exist_ok=True)
    fixture = {
        "model": "gpt2",
        "words": [
            {
                "text": words[i],
                "group": groups[i],
                "x": round(float(coords[i][0]), 4),
                "y": round(float(coords[i][1]), 4),
                "neighbours": nbrs[i],
            }
            for i in range(len(words))
        ],
        "analogies": analogy_out,
    }
    (OUT / "words.json").write_text(json.dumps(fixture, ensure_ascii=False))
    print(f"\nwrote {len(words)} words -> {OUT / 'words.json'}")


if __name__ == "__main__":
    main()
