"""Numerical-constraint validation (linear + monotone), with a contamination sweep.

The theory (separation + identifiability) is stated for near-deterministic LINEAR
constraints; most of the paper's empirical results are categorical, so this experiment
validates the linear and order instantiations directly, as the reviewer asked.

Synthetic data, two governed laws plus valid decoys:
  - linear:   total = sub + tax                 (exact, governs {sub,tax,total})
  - monotone: grade is isotonic in score        (governs {score,grade})
  - qty:      high-variance, VALID (lognormal)   -> must NOT be flagged when extreme
  - noise1/noise2: ungoverned valid noise.

Errors are VALUE-SWAPS (a governed cell replaced by another row's value of the same
column: marginally plausible, breaks the law), split evenly between `total` (linear)
and `grade` (order). We sweep the error rate eps PRESENT IN THE FITTING TABLE
(transductive) and report detection AUROC/AUPRC for CPAD (linear+order) and
IsolationForest, plus the false-positive rate on the 5% most extreme (valid) `qty`
rows -- the rare-but-valid test.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from cpad.core.table import Table
from cpad.models import LinearCPAD, OrderCPAD

RNG = np.random.default_rng(0)
N = 4000
LIN = ["sub", "tax", "total", "qty", "noise1", "noise2"]
ORD = ["level", "grade"]
ALL = LIN + ["level", "grade"]


def gen_clean(n):
    sub = RNG.gamma(3.0, 50.0, n)
    tax = RNG.gamma(2.0, 15.0, n)                       # independent of sub
    total = sub + tax                                   # exact linear law: total = sub + tax
    qty = RNG.lognormal(2.0, 0.8, n)                    # high-variance, VALID decoy
    level = RNG.integers(1, 16, n)                      # ordinal context (repeated values)
    grade = 8.0 * level + RNG.normal(0, 2.0, n)         # monotone (order) law: grade increases with level
    return pd.DataFrame(dict(sub=sub, tax=tax, total=total, qty=qty, level=level,
                             grade=grade, noise1=RNG.normal(0, 1, n), noise2=RNG.normal(0, 1, n)))


def inject(df, eps):
    df = df.copy(); y = np.zeros(len(df), bool)
    for c in ("total", "grade"):
        k = int(eps / 2 * len(df))
        idx = RNG.choice(len(df), k, replace=False)
        df.loc[idx, c] = df[c].values[RNG.permutation(len(df))][idx]
        y[idx] = True
    return df, y


def T(df, cols):
    return Table(df[cols].astype(str), id_cardinality=10**9)


def cpad_score(df):
    lin = LinearCPAD(n_constraints=2, l1=0.05).fit(T(df, LIN)).score(T(df, LIN)).max(axis=1)
    ordr = OrderCPAD(min_group=20).fit(T(df, ORD)).score(T(df, ORD)).max(axis=1)
    return np.maximum(lin, ordr)


def iforest_score(df):
    X = StandardScaler().fit_transform(df[ALL].values)
    return -IsolationForest(random_state=0).fit(X).score_samples(X)


if __name__ == "__main__":
    clean = gen_clean(N)
    extreme = clean["qty"].values >= np.quantile(clean["qty"].values, 0.95)
    print("=== Numerical constraints: detection vs contamination (linear total=sub+tax & monotone grade~score) ===")
    print(f"{'eps':>5} | {'CPAD (linear+order)':^32} | {'IsolationForest':^32}")
    print(f"{'':>5} | {'AUROC':>7} {'AUPRC':>7} {'FP@extreme':>11} | {'AUROC':>7} {'AUPRC':>7} {'FP@extreme':>11}")
    for eps in [0.02, 0.05, 0.10, 0.20, 0.30]:
        dirty, y = inject(clean, eps)
        rows = []
        for name, sc in [("cpad", cpad_score), ("if", iforest_score)]:
            s = sc(dirty)
            thr = np.quantile(s, 0.90)
            fp = float((s[extreme & ~y] > thr).mean())
            rows.append((roc_auc_score(y, s), average_precision_score(y, s), fp))
        (a1, p1, f1), (a2, p2, f2) = rows
        print(f"{eps:5.2f} | {a1:7.3f} {p1:7.3f} {f1:11.3f} | {a2:7.3f} {p2:7.3f} {f2:11.3f}")
