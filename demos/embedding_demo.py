"""Hands-on look inside an embedding model: text in, vector out, and one training nudge.

    .venv/bin/python demos/embedding_demo.py

Uses all-MiniLM-L6-v2 (6 transformer layers, 384 numbers per vector), a small free model
that runs on a laptop. Each section prints what the model is doing at that step.
"""
import copy

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(0)
NAME = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME, attn_implementation="eager")
model.eval()


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def nums(v, n=6):
    return "[" + ", ".join(f"{x:+.2f}" for x in v[:n].tolist()) + ", ...]"


def run(text):
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)
    return enc, out


def embed(texts, m=model, grad=False):
    """Mean-pool the last layer's word vectors into one vector per text, length 1."""
    enc = tok(texts, padding=True, return_tensors="pt")
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = m(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return F.normalize(pooled, dim=-1)


# ---------------------------------------------------------------------------------------
banner("STEP 1 · Text becomes token ID numbers")
A = "the court denied him appointed counsel"
B = "counsel for the plaintiff argued the motion"
C = "the court denied him a lawyer"
for s in (A, B, C):
    enc = tok(s)
    toks = tok.convert_ids_to_tokens(enc["input_ids"])
    print(f"\n{s!r}")
    print("  tokens:", toks)
    print("  IDs:   ", enc["input_ids"])
print("\n[CLS] and [SEP] are markers the model adds at the start and end.")

# ---------------------------------------------------------------------------------------
banner("STEP 2 · Each ID looks up a starting vector (384 numbers) in a table")
table = model.embeddings.word_embeddings.weight
cid = tok.convert_tokens_to_ids("counsel")
lid = tok.convert_tokens_to_ids("lawyer")
print(f"The table has {table.shape[0]:,} rows (one per token) x {table.shape[1]} numbers.")
print(f"'counsel' (ID {cid}) row starts {nums(table[cid])}")
print(f"'lawyer'  (ID {lid}) row starts {nums(table[lid])}")
print("\nThe lookup ignores context: 'counsel' fetches this SAME row in sentence A and sentence B.")
print("Similarity of the 'counsel' and 'lawyer' rows before any context:",
      f"{F.cosine_similarity(table[cid], table[lid], dim=0).item():+.2f}")

# ---------------------------------------------------------------------------------------
banner("STEP 3 · Each layer mixes context into every word's vector")
encA, outA = run(A)
encB, outB = run(B)
encC, outC = run(C)
ia = tok.convert_ids_to_tokens(encA["input_ids"][0]).index("counsel")
ib = tok.convert_ids_to_tokens(encB["input_ids"][0]).index("counsel")
ic = tok.convert_ids_to_tokens(encC["input_ids"][0]).index("lawyer")
print("Tracking three word vectors through the 6 layers:")
print("  counsel_A = 'counsel' in A (denied him appointed counsel)")
print("  counsel_B = 'counsel' in B (counsel for the plaintiff argued)")
print("  lawyer_C  = 'lawyer'  in C (denied him a lawyer)\n")
print(f"{'after':<14}{'counsel_A vs counsel_B':>24}{'counsel_A vs lawyer_C':>24}")
names = ["lookup"] + [f"layer {i}" for i in range(1, 7)]
for name, hA, hB, hC in zip(names, outA.hidden_states, outB.hidden_states, outC.hidden_states):
    sab = F.cosine_similarity(hA[0, ia], hB[0, ib], dim=0).item()
    sac = F.cosine_similarity(hA[0, ia], hC[0, ic], dim=0).item()
    print(f"{name:<14}{sab:>24.2f}{sac:>24.2f}")
print("\nRead down the columns. The same word 'counsel' starts nearly identical in A and B")
print("(only its position in the sentence differs) and drifts apart as each layer mixes in its")
print("neighbors. Meanwhile 'counsel' in A drifts toward 'lawyer' in C: a different word in the")
print("same kind of sentence.")

print("\nWhich words did 'counsel' in sentence A pull information from in the last layer?")
att = outA.attentions[-1][0].mean(0)[ia]  # average over the layer's attention heads
toksA = tok.convert_ids_to_tokens(encA["input_ids"][0])
for t, w in sorted(zip(toksA, att.tolist()), key=lambda x: -x[1]):
    print(f"  {t:<12}{'#' * int(w * 60):<40}{w:.2f}")
print("Each row is a share of attention; they add up to 1. The layer builds counsel's new")
print("vector as a weighted mix of these words' vectors, using these shares as the weights.")

# ---------------------------------------------------------------------------------------
banner("STEP 4 · Averaging many word vectors into one fixed-length vector")
last = outA.last_hidden_state[0]
print(f"After layer 6, sentence A has {last.shape[0]} word vectors of {last.shape[1]} numbers each:")
for t, v in zip(toksA, last):
    print(f"  {t:<12}{nums(v, 5)}")
mean = last.mean(0)
print(f"\nAverage of those {last.shape[0]} rows (column by column) = one vector of {mean.shape[0]}:")
print(f"  {'(sentence)':<12}{nums(mean, 5)}")
long = "the court denied him appointed counsel " * 20
print(f"\nA {len(tok(long)['input_ids'])}-token passage also averages down to {embed([long]).shape[1]} numbers.")
print("That's why every text, short or long, becomes the same length of vector.")

# ---------------------------------------------------------------------------------------
banner("STEP 5 · Comparing vectors: multiply position by position, then add")
phrases = ["refused appointed counsel", "denied a lawyer", "counsel for the plaintiff argued",
           "the weather was mild"]
E = embed(phrases)
a, b = E[0], E[1]
print("First 4 numbers of 'refused appointed counsel':", nums(a, 4))
print("First 4 numbers of 'denied a lawyer':          ", nums(b, 4))
prods = (a[:4] * b[:4]).tolist()
print("Products of those 4 pairs:", ", ".join(f"{p:+.3f}" for p in prods), f"  sum so far = {sum(prods):+.3f}")
print(f"Doing that for all 384 positions and adding gives the similarity: {float(a @ b):.2f}\n")
print(f"{'':<34}" + "".join(f"{p[:14]:>16}" for p in phrases))
for p, row in zip(phrases, E @ E.T):
    print(f"{p:<34}" + "".join(f"{x:>16.2f}" for x in row.tolist()))
print("\n1.00 on the diagonal is each phrase compared with itself.")

# ---------------------------------------------------------------------------------------
banner("STEP 6 · Training: a few backpropagation nudges on one example")
student = copy.deepcopy(model)  # a throwaway copy, so the real model above is unchanged
student.train()
opt = torch.optim.AdamW(student.parameters(), lr=3e-5)
anchor = "the judge refused to appoint a public defender"
positive = "the defendant was denied a lawyer"
negatives = ["counsel for the plaintiff argued", "the weather was mild"]
cands = [positive] + negatives
print(f"Teach the model that {anchor!r}")
print(f"belongs with {positive!r} and not with the other two.\n")
watch = student.embeddings.word_embeddings.weight
print(f"{'step':<6}{'loss':>8}{'positive':>12}{'plaintiff':>12}{'weather':>10}   "
      f"'lawyer' row, first number (before -> after nudge)")
for step in range(1, 7):
    opt.zero_grad()
    q = embed([anchor], student, grad=True)
    k = embed(cands, student, grad=True)
    sims = (q @ k.T)[0]
    # Contrastive loss: treat it as a 3-way choice and reward picking the positive.
    loss = F.cross_entropy((sims / 0.05).unsqueeze(0), torch.tensor([0]))
    before = watch[lid, 0].item()
    loss.backward()  # backpropagation: how should every number move to lower the loss?
    opt.step()       # the nudge: move every number a small step that way
    after = watch[lid, 0].item()
    print(f"{step:<6}{loss.item():>8.3f}{sims[0].item():>12.3f}{sims[1].item():>12.3f}"
          f"{sims[2].item():>10.3f}   {before:+.6f} -> {after:+.6f}")
n = sum(p.numel() for p in student.parameters())
print(f"\nEvery step adjusts all {n:,} numbers in the model by a tiny amount, each in the")
print("direction that would have made the positive pair score higher than the negatives.")
print("Real training repeats this over billions of pairs; meaning emerges from those nudges.")
