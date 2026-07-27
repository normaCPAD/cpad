"""tab:contam (contamination robustness) and tab:budget (label-budget sensitivity) REPLICATED
ON ADULT, to remove the Hospital-only character of these two studies. Value-swap errors are
injected into Adult's FD-governed categorical columns to form the ground truth; contamination
adds extra value-swaps to the (transductively) fitted table, and the few-shot baseline mirrors
the HoloDetect-style detector used on Hospital.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")

from _common import DATA as D
RNG = np.random.default_rng(0)
raw = pd.read_csv(D + "adult.csv", dtype=str).fillna("").apply(lambda s: s.str.strip().str.lower())
cols = [c for c in raw.columns if 2 <= raw[c].nunique() <= 50]
clean = raw[cols].reset_index(drop=True)
n = len(clean); ci = {c: i for i, c in enumerate(cols)}
GOV = [c for c in ["education-num", "education", "relationship", "marital-status"] if c in cols]


def fd_cell_scores(df, tau=0.85):
    cell = np.zeros((n, len(cols)))
    for B in cols:
        for A in cols:
            if A == B:
                continue
            mode = df.groupby(A)[B].transform(lambda s: s.value_counts().idxmax())
            if (df[B].values == mode.values).mean() < tau:
                continue
            size = df.groupby(A)[B].transform("size").values
            own = df.groupby([A, B])[B].transform("size").values
            cell[:, ci[B]] = np.maximum(cell[:, ci[B]], 1.0 - own / size)
    return cell


def inject_targets(df, rate=0.05):                       # ground-truth errors in governed columns
    dc = df.copy(); M = np.zeros((n, len(cols)), bool)
    for c in GOV:
        freq = df[c].value_counts(normalize=True); pool, p = freq.index.values, freq.values
        idx = np.where(RNG.random(n) < rate)[0]; repl = RNG.choice(pool, len(idx), p=p)
        keep = repl != df[c].values[idx]; idx, repl = idx[keep], repl[keep]
        dc.iloc[idx, ci[c]] = repl; M[idx, ci[c]] = True
    return dc.reset_index(drop=True), M


def add_contamination(df, eps):                          # extra value-swaps across all columns
    dc = df.copy()
    for c in cols:
        freq = df[c].value_counts(normalize=True); pool, p = freq.index.values, freq.values
        idx = np.where(RNG.random(n) < eps)[0]; repl = RNG.choice(pool, len(idx), p=p)
        keep = repl != df[c].values[idx]; idx, repl = idx[keep], repl[keep]
        dc.iloc[idx, ci[c]] = repl
    return dc.reset_index(drop=True)


def features(df, fd):
    marg = np.zeros((n, len(cols)))
    for c in cols:
        marg[:, ci[c]] = 1.0 - df[c].map(df[c].value_counts() / n).values
    f = []
    for c in cols:
        v = df[c].astype(str)
        f.append(np.stack([marg[:, ci[c]], v.str.len().values,
                           v.str.count(r"\d").values / (v.str.len().values + 1),
                           np.full(n, df[c].nunique()), fd[:, ci[c]]], 1))
    return np.concatenate(f, 0)


def fewshot(b, dirty, err):
    Xreal = features(dirty, fd_cell_scores(dirty)); y = err.T.ravel()
    k = b // 2
    lab = np.r_[RNG.choice(np.where(y == 1)[0], k, replace=False),
                RNG.choice(np.where(y == 0)[0], k, replace=False)]
    dinj, M = inject_targets(dirty); Xaug = features(dinj, fd_cell_scores(dinj))
    aug = np.where(M.T.ravel() == 1)[0]
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(np.vstack([Xreal[lab], Xaug[aug]]), np.r_[y[lab], np.ones(len(aug))])
    return clf.predict_proba(Xreal)[:, 1].reshape(len(cols), n).T


dirty, err = inject_targets(clean)

print(f"ADULT contamination robustness ({n} rows, {len(cols)} cols, {int(err.sum())} target errors)")
print(f"{'eps added':>10}{'cellAUROC':>11}{'rowAUROC':>10}")
for eps in [0.0, 0.05, 0.10, 0.20, 0.30]:
    cont = add_contamination(dirty, eps) if eps > 0 else dirty
    s = fd_cell_scores(cont)
    print(f"{eps:>10.2f}{roc_auc_score(err.ravel(), s.ravel()):>11.3f}{roc_auc_score(err.any(1), s.max(1)):>10.3f}")

print(f"\nADULT label budget")
print(f"{'method':<26}{'cellAUROC':>11}{'rowAUROC':>10}")
s0 = fd_cell_scores(dirty)
print(f"{'CPAD-cat (0 labels)':<26}{roc_auc_score(err.ravel(), s0.ravel()):>11.3f}{roc_auc_score(err.any(1), s0.max(1)):>10.3f}")
for b in [50, 100, 200, 500]:
    s = fewshot(b, dirty, err)
    print(f"{'HoloDetect (' + str(b) + ' lab.)':<26}{roc_auc_score(err.ravel(), s.ravel()):>11.3f}{roc_auc_score(err.any(1), s.max(1)):>10.3f}")
