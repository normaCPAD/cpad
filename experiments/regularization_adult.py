"""Regularization ablation REPLICATED ON ADULT (not Hospital), to show the L1 finding is not
dataset-specific. Value-swap errors are injected into the FD-governed categorical columns to
form the ground truth; the differentiable gated model is then swept over the L1 weight lambda.
As on Hospital, lambda=0 mixes all columns and collapses, while a moderate lambda sparsifies
the gates onto each FD's left-hand side and detection improves.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from _common import DATA as D

rng = np.random.default_rng(0)
raw = pd.read_csv(D + "adult.csv", dtype=str).fillna("").apply(lambda s: s.str.strip().str.lower())
cols = [c for c in raw.columns if 2 <= raw[c].nunique() <= 50]      # categorical, FD-bearing
df = raw[cols].reset_index(drop=True)
n, nc = len(df), len(cols)
GOV = [c for c in ["education-num", "education", "relationship", "marital-status"] if c in cols]

# inject value-swaps (frequency-weighted valid values) into governed columns -> ground truth
err = np.zeros((n, nc), bool)
dirty = df.copy()
for c in GOV:
    j = cols.index(c)
    freq = df[c].value_counts(normalize=True); pool, p = freq.index.values, freq.values
    idx = np.where(rng.random(n) < 0.05)[0]
    repl = rng.choice(pool, size=len(idx), p=p)
    keep = repl != df[c].values[idx]; idx, repl = idx[keep], repl[keep]
    dirty.iloc[idx, j] = repl; err[idx, j] = True

codes, cards = [], []
for c in cols:
    code, uniq = pd.factorize(dirty[c]); codes.append(code); cards.append(len(uniq))
X = torch.tensor(np.stack(codes, 1), dtype=torch.long)
d, EPOCHS = 24, 400
eye = torch.eye(nc)
ce = nn.CrossEntropyLoss()


def row(s):
    return roc_auc_score(err.any(1), s.max(1))


def corrupt(xb, p=0.15):
    xc = xb.clone()
    for j in range(nc):
        m = torch.rand(n) < p; xc[m, j] = xb[torch.randperm(n), j][m]
    return xc


def train_eval(lam):
    torch.manual_seed(0)
    embs = nn.ModuleList([nn.Embedding(c, d) for c in cards])
    heads = nn.ModuleList([nn.Linear(d, c) for c in cards])
    gate_logits = nn.Parameter(torch.zeros(nc, nc) - 2.0)
    opt = torch.optim.Adam(list(embs.parameters()) + list(heads.parameters()) + [gate_logits], lr=0.01)
    gates = lambda: F.softplus(gate_logits) * (1 - eye)
    for ep in range(EPOCHS):
        xin = corrupt(X)
        E = torch.stack([embs[j](xin[:, j]) for j in range(nc)], 1)
        G = gates(); R = torch.einsum('ba,nad->nbd', G, E)
        loss = sum(ce(heads[B](R[:, B, :]), X[:, B]) for B in range(nc)) + lam * G.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        E = torch.stack([embs[j](X[:, j]) for j in range(nc)], 1)
        G = gates(); R = torch.einsum('ba,nad->nbd', G, E)
        viol = np.zeros((n, nc))
        for B in range(nc):
            p = torch.softmax(heads[B](R[:, B, :]), 1)
            viol[:, B] = 1.0 - p[torch.arange(n), X[:, B]].numpy()
        srcs = (G > 0.05).float().sum().item() / nc
    return viol, srcs


print(f"ADULT diff. CPAD + L1 gate regularization ({n} rows, {nc} cat. cols, {EPOCHS} ep.)")
print(f"{'lambda':>8}{'srcs/col':>10}{'cellAUROC':>11}{'rowAUROC':>10}")
print("-" * 40)
for lam in [0.0, 0.1, 0.3, 0.5, 0.8]:
    viol, srcs = train_eval(lam)
    ca = roc_auc_score(err.ravel(), viol.ravel())
    print(f"{lam:>8.1f}{srcs:>10.1f}{ca:>11.3f}{row(viol):>10.3f}")
