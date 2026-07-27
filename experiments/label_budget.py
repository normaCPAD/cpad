"""Label-budget sensitivity: CPAD (0 labels) vs a HoloDetect-style few-shot detector
as its label budget grows. Addresses the reviewer's concern that the comparison to
HoloDetect could be label-budget sensitive.

CPAD-cat is fully unsupervised (tau chosen by a self-supervised value-swap signal).
The few-shot baseline is the SAME featurization + classifier, trained on `b` real
labels plus synthetic augmentation. We sweep b and report cell/row AUROC and AUPRC on
Hospital (HoloClean), with CPAD as a horizontal, label-free reference.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from _common import DATA

RNG = np.random.default_rng(0)


def norm(df):
    return df.fillna("").apply(lambda s: s.astype(str).str.strip().str.lower())


dirty = norm(pd.read_csv(DATA + "hospital_dirty.csv", dtype=str)).reset_index(drop=True)
err = np.load(DATA + "hospital_errmask.npy")
cols = list(dirty.columns)
n = len(dirty)
ci = {c: j for j, c in enumerate(cols)}


def conditional_viol(df, lhs, B):
    grp = df.groupby(lhs)[B]
    freq = grp.transform(lambda s: s.map(s.value_counts(normalize=True)))
    return 1.0 - freq.values


def fd_cell_scores(df, tau):
    cell = np.zeros((n, len(cols)))
    for B in cols:
        for A in cols:
            if A == B:
                continue
            mode = df.groupby(A)[B].transform(lambda s: s.value_counts().idxmax())
            if (df[B].values == mode.values).mean() < tau:
                continue
            cell[:, ci[B]] = np.maximum(cell[:, ci[B]], conditional_viol(df, [A], B))
    return cell


def inject(df, rate=0.05):
    dc = df.copy(); mask = np.zeros((n, len(cols)), bool)
    for c in cols:
        vals = df[c].values; idx = RNG.random(n) < rate
        repl = RNG.permutation(vals); idx &= (repl != vals)
        dc.iloc[idx, ci[c]] = repl[idx]; mask[idx, ci[c]] = True
    return dc.reset_index(drop=True), mask


def features(df, fd):
    f, marg = [], np.zeros((n, len(cols)))
    for c in cols:
        marg[:, ci[c]] = 1.0 - df[c].map(df[c].value_counts() / n).values
    for c in cols:
        vals = df[c].astype(str)
        f.append(np.stack([marg[:, ci[c]], vals.str.len().values,
                           vals.str.count(r"\d").values / (vals.str.len().values + 1),
                           np.full(n, df[c].nunique()), fd[:, ci[c]]], 1))
    return np.concatenate(f, 0)


def fewshot(n_labels):
    fd_real = fd_cell_scores(dirty, 0.9); Xreal = features(dirty, fd_real); y = err.T.ravel()
    k = n_labels // 2
    lab = np.r_[RNG.choice(np.where(y == 1)[0], k, replace=False),
                RNG.choice(np.where(y == 0)[0], k, replace=False)]
    dinj, M = inject(dirty); Xaug = features(dinj, fd_cell_scores(dinj, 0.9))
    aug = np.where(M.T.ravel() == 1)[0]
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(np.vstack([Xreal[lab], Xaug[aug]]), np.r_[y[lab], np.ones(len(aug))])
    return clf.predict_proba(Xreal)[:, 1].reshape(len(cols), n).T


def report(name, s):
    cell_auroc = roc_auc_score(err.ravel(), s.ravel())
    cell_aupr = average_precision_score(err.ravel(), s.ravel())
    row_auroc = roc_auc_score(err.any(1), s.max(1))
    print(f"  {name:<28}{cell_auroc:>10.3f}{cell_aupr:>10.3f}{row_auroc:>10.3f}")


if __name__ == "__main__":
    Dinj, M = inject(dirty, 0.03)
    tau_self = max([0.80, 0.85, 0.90, 0.95],
                   key=lambda t: roc_auc_score(M.ravel(), fd_cell_scores(Dinj, t).ravel()))
    print(f"Hospital: {n} rows, {len(cols)} cols, {int(err.sum())} errors ({100*err.mean():.2f}%). "
          f"CPAD self-sup tau={tau_self}")
    print(f"  {'method':<28}{'cellAUROC':>10}{'cellAUPRC':>10}{'rowAUROC':>10}")
    print("  " + "-" * 58)
    report("CPAD-cat (0 labels)", fd_cell_scores(dirty, tau_self))
    for b in [50, 100, 200, 500, 1000]:
        report(f"HoloDetect-style ({b} lab.)", fewshot(b))
