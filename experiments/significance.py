"""Statistical significance: repeat the Hospital detection over many random seeds and
report mean +/- std, plus a paired Wilcoxon test of CPAD against Isolation Forest.
Addresses the "single run / no confidence intervals" concern.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

from _common import DATA

SEEDS = list(range(10))
TAUS = [0.80, 0.85, 0.90, 0.95]


def norm(df):
    return df.fillna("").apply(lambda s: s.astype(str).str.strip().str.lower())


dirty = norm(pd.read_csv(DATA + "hospital_dirty.csv", dtype=str)).reset_index(drop=True)
err = np.load(DATA + "hospital_errmask.npy")
cols = list(dirty.columns); n = len(dirty); ci = {c: j for j, c in enumerate(cols)}
y = err.any(1).astype(int)
Xoh = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dirty).toarray()


def pairwise(df):
    """For every ordered pair (A->B): FD strength and per-row conditional violation,
    fully vectorized (no groupby-apply)."""
    out = {}
    for A in cols:
        a = df.groupby(A).size()
        grp = a.reindex(df[A].values).to_numpy()
        for B in cols:
            if A == B:
                continue
            ab = df.groupby([A, B]).size()
            joint = ab.reindex(list(zip(df[A], df[B]))).to_numpy()
            strength = ab.groupby(level=0).max().sum() / len(df)
            out[(A, B)] = (strength, 1.0 - joint / grp)
    return out


def cell_scores(pairs, tau):
    cell = np.zeros((n, len(cols)))
    for (A, B), (strength, viol) in pairs.items():
        if strength >= tau:
            cell[:, ci[B]] = np.maximum(cell[:, ci[B]], viol)
    return cell


def inject(df, rng, rate=0.03):
    dc = df.copy(); M = np.zeros((n, len(cols)), bool)
    for c in cols:
        idx = rng.random(n) < rate
        repl = rng.permutation(df[c].values); idx &= (repl != df[c].values)
        dc.iloc[idx, ci[c]] = repl[idx]; M[idx, ci[c]] = True
    return dc.reset_index(drop=True), M


real_pairs = pairwise(dirty)                                # deterministic; computed once
cpad_au, if_au = [], []
for s in SEEDS:
    rng = np.random.default_rng(s)
    dinj, M = inject(dirty, rng)
    inj_pairs = pairwise(dinj)
    tau = max(TAUS, key=lambda t: roc_auc_score(M.ravel(), cell_scores(inj_pairs, t).ravel()))
    cpad_au.append(roc_auc_score(y, cell_scores(real_pairs, tau).max(1)))
    if_au.append(roc_auc_score(y, -IsolationForest(random_state=s, n_estimators=200).fit(Xoh).score_samples(Xoh)))

cpad_au, if_au = np.array(cpad_au), np.array(if_au)
stat, p = wilcoxon(cpad_au, if_au)
print(f"Hospital row-level AUROC over {len(SEEDS)} seeds (mean +/- std)")
print(f"  CPAD-cat          {cpad_au.mean():.3f} +/- {cpad_au.std():.3f}")
print(f"  Isolation Forest  {if_au.mean():.3f} +/- {if_au.std():.3f}")
print(f"  Wilcoxon CPAD vs IForest: W={stat:.1f}, p={p:.2e}")
