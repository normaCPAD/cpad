"""Direct experimental test of Theorem 1 (separation): the theory predicts that errors become
separable from clean tuples only as the governing constraint approaches determinism (the
contrastive gap / FD confidence -> 1). We therefore PLANT a single FD S->T of controlled
confidence c, inject value-swap errors into T, and measure detection AUROC as a function of c.

If the theory is connected to practice, AUROC should rise sharply with c and approach 1 only
near determinism, since at confidence c a clean tuple itself violates the FD with probability
1-c and is indistinguishable from an injected error. The measured curve is the empirical
manifestation of the separation bound.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
N, ERR, SEEDS = 5000, 0.05, 5


def fd_row_score(df):
    size = df.groupby("S")["T"].transform("size").values
    own = df.groupby(["S", "T"])["T"].transform("size").values
    return 1.0 - own / size


def trial(conf, seed):
    rg = np.random.default_rng(seed)
    s = rg.integers(0, 25, N)
    fmap = rg.integers(0, 8, 25)
    t = fmap[s].copy()
    flip = rg.random(N) < (1 - conf)                       # break the FD on a (1-conf) fraction -> conf(S->T)=c
    t[flip] = rg.integers(0, 8, flip.sum())
    df = pd.DataFrame({"S": s.astype(str), "T": t.astype(str)})
    y = np.zeros(N, bool)                                   # inject value-swap errors into T (ground truth)
    idx = np.where(rg.random(N) < ERR)[0]
    df.iloc[idx, 1] = rg.integers(0, 8, len(idx)).astype(str); y[idx] = True
    return roc_auc_score(y, fd_row_score(df))


print("Theorem 1 (separation) -- detection AUROC vs planted constraint confidence")
print(f"{'FD confidence c':>16}{'detection AUROC':>18}")
print("-" * 34)
for c in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 1.00]:
    a = np.array([trial(c, s) for s in range(SEEDS)])
    print(f"{c:>16.2f}{a.mean():>13.3f} ±{a.std():.3f}")
print("\nPrediction: AUROC tracks c and approaches 1 only near determinism -- exactly Thm 1.")
