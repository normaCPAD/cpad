"""Inter-tuple ORDER-DC detection on Tax. The order denial constraint is, per state, rate is
monotone non-decreasing in salary  (salary up => rate up). A violation is a *valid, common*
rate value that is wrong for the tuple's salary within its state -- the prototypical
"error != rare" case, and one that *no* equality FD/DC miner (FASTDC, HyFD, DCFinder, DCMiner)
can express, since it compares PAIRS of tuples.

We inject such violations, score each tuple by its isotonic residual within its state group,
and contrast with (i) marginal rarity, (ii) the same monotone fit ignoring the state context,
and (iii) the best single-LHS equality FD on rate. Only the conditioned order DC detects them.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

from _common import DATA as D
rng = np.random.default_rng(0)
df = pd.read_csv(D + "tax/clean.csv", dtype=str)
df = df[["state", "salary", "rate"]].dropna()
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
df = df.dropna().reset_index(drop=True)
df = df.sample(n=min(30000, len(df)), random_state=0).reset_index(drop=True)
n = len(df)


def inject_order_violations(frame, rate=0.05):
    """Replace a tuple's rate by another VALID rate drawn from its own state (frequency-weighted),
    so the value is common but breaks the salary->rate monotonicity. Ground truth = changed rows."""
    out = frame.copy(); y = np.zeros(n, bool)
    for st, g in frame.groupby("state"):
        idx = g.index.values
        sel = idx[rng.random(len(idx)) < rate]
        if len(sel) == 0 or len(g) < 5:
            continue
        pool = g["rate"].values
        repl = rng.choice(pool, size=len(sel))
        keep = repl != frame.loc[sel, "rate"].values
        sel, repl = sel[keep], repl[keep]
        out.loc[sel, "rate"] = repl; y[sel] = True
    return out, y


def isotonic_residual(frame, by_state=True):
    """Per-(state) isotonic fit of rate on salary; score = |rate - fitted|. If by_state=False,
    a single global fit (no context) -- the ablation that loses the conditioning."""
    score = np.zeros(n)
    groups = frame.groupby("state").groups if by_state else {"_": frame.index}
    for _, idx in groups.items():
        idx = np.asarray(idx)
        if len(idx) < 3:
            continue
        s = frame.loc[idx, "salary"].values.astype(float)
        r = frame.loc[idx, "rate"].values.astype(float)
        ir = IsotonicRegression(out_of_bounds="clip").fit(s, r)
        score[idx] = np.abs(r - ir.predict(s))
    return score


def marginal_rarity(frame):
    f = frame["rate"].map(frame["rate"].value_counts() / n).values
    return 1.0 - f


def best_equality_fd(frame):
    """Best single-LHS equality FD A->rate confidence (rate is numeric/continuous in context:
    equality determinants are weak). Returns the conditional-violation score of the best A."""
    best_s, best_A = -1, None
    for A in ["state", "salary"]:
        g = frame.groupby(A)["rate"]
        conf = g.transform(lambda s: s.value_counts(normalize=True).max()).mean()
        if conf > best_s:
            best_s, best_A = conf, A
    g = frame.groupby(best_A)["rate"]
    own = frame.groupby([best_A, "rate"])["rate"].transform("size").values
    size = g.transform("size").values
    return 1.0 - own / size, best_s, best_A


dirty, y = inject_order_violations(df)
print(f"Tax order-DC detection: {n} rows, {int(y.sum())} injected order violations "
      f"(valid rates, wrong salary context)")
print(f"{'method':<42}{'AUROC':>8}")
print("-" * 50)
print(f"{'CPAD order DC (isotonic residual | state)':<42}{roc_auc_score(y, isotonic_residual(dirty, True)):>8.3f}")
print(f"{'  -- same fit, NO state context (ablation)':<42}{roc_auc_score(y, isotonic_residual(dirty, False)):>8.3f}")
print(f"{'Marginal rarity (1 - freq(rate))':<42}{roc_auc_score(y, marginal_rarity(dirty)):>8.3f}")
eq, conf, A = best_equality_fd(dirty)
print(f"{'Best equality FD (' + A + ' -> rate, conf=' + format(conf, '.2f') + ')':<42}{roc_auc_score(y, eq):>8.3f}")
