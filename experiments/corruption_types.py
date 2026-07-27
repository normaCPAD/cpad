"""Corruption-diversity check: CPAD is evaluated against five *different* error-generation
mechanisms, only one of which (frequency-weighted value-swap) matches the contrastive
corrupter used for self-supervision. Each mechanism corrupts FD-governed columns, so each
breaks a dependency; the question is whether CPAD detects them all or only the aligned one.

Reports row-level AUROC (mean +/- std over seeds) for CPAD and Isolation Forest. Answers the
reviewer concern that "the experimental design favors the proposed approach".
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

from _common import DATA as D
norm = lambda df: df.fillna("").apply(lambda s: s.astype(str).str.strip().str.lower())
clean = norm(pd.read_csv(D + "hospital_clean_wide.csv", dtype=str)).reset_index(drop=True)
cols = list(clean.columns); n = len(clean); nc = len(cols); ci = {c: i for i, c in enumerate(cols)}
TARGETS = [c for c in cols if 2 <= clean[c].nunique() <= 60]
TAU = 0.85


def fd_row_score(df):
    cell = np.zeros((n, nc))
    for bi, B in enumerate(cols):
        for A in cols:
            if A == B:
                continue
            g = df.groupby(A)[B]
            mode = g.transform(lambda s: s.value_counts().idxmax())
            if (df[B].values == mode.values).mean() < TAU:
                continue
            size = g.transform("size").values
            own = df.groupby([A, B])[B].transform("size").values
            cell[:, bi] = np.maximum(cell[:, bi], 1.0 - own / size)
    return cell.max(1)


def corrupt(df, rng, mech, rate=0.05):
    dc = df.copy(); M = np.zeros((n, nc), bool)
    for c in TARGETS:
        vals = df[c].values
        idx = np.where(rng.random(n) < rate)[0]
        if mech == "swap":                                   # freq-weighted valid value (the aligned one)
            freq = df[c].value_counts(normalize=True); pool, p = freq.index.values, freq.values
            repl = rng.choice(pool, size=len(idx), p=p)
        elif mech == "uniform":                              # uniform random valid value
            pool = df[c].unique(); repl = rng.choice(pool, size=len(idx))
        elif mech == "typo":                                 # unique perturbed string (marginally rare)
            repl = np.array([str(vals[i]) + "x" + str(rng.integers(99)) for i in idx])
        elif mech == "missing":                              # blanked-out cell
            repl = np.array([""] * len(idx))
        elif mech == "constant":                             # one fixed valid value injected everywhere
            const = df[c].value_counts().idxmax(); repl = np.array([const] * len(idx))
        keep = repl != vals[idx]
        idx, repl = idx[keep], repl[keep]
        dc.iloc[idx, ci[c]] = repl; M[idx, ci[c]] = True
    return dc.reset_index(drop=True), M.any(1)


def marginal_row(df):                                    # density baseline: rarity of the cell value
    m = np.zeros((n, nc))
    for c in cols:
        m[:, ci[c]] = 1.0 - df[c].map(df[c].value_counts() / n).values
    return m.max(1)


MECHS = [("typo", "typo"), ("value-swap*", "swap"), ("uniform", "uniform"),
         ("missing", "missing"), ("constant", "constant")]
SEEDS = range(5)
print(f"Clean Hospital: {n} tuples, {len(TARGETS)} FD-bearing columns. (* = aligned with the corrupter)")
print(f"{'mechanism':<14}{'CPAD':>16}{'Marginal':>16}{'IForest':>16}")
print("-" * 62)
for label, mech in MECHS:
    cp, mg, ifr = [], [], []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        dc, y = corrupt(clean, rng, mech)
        cp.append(roc_auc_score(y, fd_row_score(dc)))
        mg.append(roc_auc_score(y, marginal_row(dc)))
        X = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dc).toarray()
        ifr.append(roc_auc_score(y, -IsolationForest(random_state=s, n_estimators=150).fit(X).score_samples(X)))
    cp, mg, ifr = np.array(cp), np.array(mg), np.array(ifr)
    print(f"{label:<14}{cp.mean():>9.3f} +/-{cp.std():.3f}"
          f"{mg.mean():>10.3f} +/-{mg.std():.3f}{ifr.mean():>10.3f} +/-{ifr.std():.3f}")
