"""tab:contam (contamination robustness) and tab:budget (label-budget) REPLICATED ON A
SYNTHETIC relational table with planted FDs, so the two studies are no longer Hospital-only.
A synthetic table is the fair setting here: Adult/Flights have at most one strong single-LHS
FD, so contaminating it destroys the only signal; with several planted FDs (as in Hospital)
the majority vote degrades gracefully, which is exactly the property under test.

Four independent exact FDs S_i -> T_i are planted among noise columns; value-swaps in the
T_i form the ground truth; contamination adds extra value-swaps to the fitted table.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(0)
N, K, NOISE = 4000, 4, 4                                  # rows, planted FDs, noise columns


def make_table():
    data = {}
    for i in range(K):
        s = RNG.integers(0, 40, N)                       # source (key-like)
        mapping = RNG.integers(0, 12, 40)                # deterministic S_i -> T_i
        data[f"S{i}"] = s.astype(str)
        data[f"T{i}"] = mapping[s].astype(str)           # exact FD S_i -> T_i
    for j in range(NOISE):
        data[f"N{j}"] = RNG.integers(0, 25, N).astype(str)
    return pd.DataFrame(data)


clean = make_table()
cols = list(clean.columns); ci = {c: i for i, c in enumerate(cols)}
TARGETS = [f"T{i}" for i in range(K)]


def fd_cell_scores(df, tau=0.85):
    cell = np.zeros((N, len(cols)))
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


def inject_targets(df, rate=0.05):
    dc = df.copy(); M = np.zeros((N, len(cols)), bool)
    for c in TARGETS:
        freq = df[c].value_counts(normalize=True); pool, p = freq.index.values, freq.values
        idx = np.where(RNG.random(N) < rate)[0]; repl = RNG.choice(pool, len(idx), p=p)
        keep = repl != df[c].values[idx]; idx, repl = idx[keep], repl[keep]
        dc.iloc[idx, ci[c]] = repl; M[idx, ci[c]] = True
    return dc.reset_index(drop=True), M


def add_contamination(df, eps):                          # extra value-swaps in dependent + noise columns
    dc = df.copy()                                       # (sources kept intact: the majority-vote test)
    for c in TARGETS + [f"N{j}" for j in range(NOISE)]:
        freq = df[c].value_counts(normalize=True); pool, p = freq.index.values, freq.values
        idx = np.where(RNG.random(N) < eps)[0]; repl = RNG.choice(pool, len(idx), p=p)
        keep = repl != df[c].values[idx]; idx, repl = idx[keep], repl[keep]
        dc.iloc[idx, ci[c]] = repl
    return dc.reset_index(drop=True)


def features(df, fd):
    marg = np.zeros((N, len(cols)))
    for c in cols:
        marg[:, ci[c]] = 1.0 - df[c].map(df[c].value_counts() / N).values
    f = []
    for c in cols:
        v = df[c].astype(str)
        f.append(np.stack([marg[:, ci[c]], v.str.len().values,
                           np.full(N, df[c].nunique()), fd[:, ci[c]]], 1))
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
    return clf.predict_proba(Xreal)[:, 1].reshape(len(cols), N).T


dirty, err = inject_targets(clean)
print(f"SYNTHETIC ({N} rows, {len(cols)} cols, {K} planted FDs, {int(err.sum())} target errors)")
print(f"\nContamination robustness\n{'eps added':>10}{'cellAUROC':>11}{'rowAUROC':>10}")
for eps in [0.0, 0.05, 0.10, 0.20, 0.30]:
    cont = add_contamination(dirty, eps) if eps > 0 else dirty
    s = fd_cell_scores(cont)
    print(f"{eps:>10.2f}{roc_auc_score(err.ravel(), s.ravel()):>11.3f}{roc_auc_score(err.any(1), s.max(1)):>10.3f}")

print(f"\nLabel budget\n{'method':<26}{'cellAUROC':>11}{'rowAUROC':>10}")
s0 = fd_cell_scores(dirty)
print(f"{'CPAD-cat (0 labels)':<26}{roc_auc_score(err.ravel(), s0.ravel()):>11.3f}{roc_auc_score(err.any(1), s0.max(1)):>10.3f}")
for b in [50, 100, 200, 500]:
    s = fewshot(b, dirty, err)
    print(f"{'HoloDetect (' + str(b) + ' lab.)':<26}{roc_auc_score(err.ravel(), s.ravel()):>11.3f}{roc_auc_score(err.any(1), s.max(1)):>10.3f}")
