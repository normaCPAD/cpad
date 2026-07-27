"""Large-scale stress test on 5 synthetic relational datasets with planted FD structure.

Each dataset plants `k` deterministic FDs (source -> governed), surrounds them with noise
columns, then injects value-swap violations into the governed columns. We report, at scale
(up to n=10^6 rows and d=150 columns), using CPAD's discrete conditional-violation scoring
in a vectorized form (factorize + bincount; same method as the engine, faster):
  - detection quality   : tuple-level AUROC / AUPRC, and Isolation Forest on a subsample;
  - constraint recovery : planted FDs recovered (strength >= tau);
  - cost                : scan wall-clock (s) and peak memory (MB).
"""
from __future__ import annotations
import time, tracemalloc
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

RNG = np.random.default_rng(0)
TAU = 0.85
SWAP = 0.08
SETS = [("S1",   500_000,  30,  8),
        ("S2", 1_000_000,  30,  8),
        ("S3",   200_000,  60, 12),
        ("S4",   100_000, 100, 15),
        ("S5",    50_000, 150, 18),
        ("S6",   500_000, 100, 15),
        ("S7", 1_000_000, 100, 15),
        ("S8",    10_000,1000, 20)]


def gen(n, d, k, card=30, fd_noise=0.02):
    """Plant k APPROXIMATE FDs: governed = lut[source], then perturb a fraction `fd_noise`
    of governed cells so the dependency holds with confidence ~1-fd_noise (realistic)."""
    cols, planted = {}, []
    for i in range(k):
        src = RNG.integers(0, card, n)
        lut = RNG.integers(0, max(card - 5, 5), card)
        gov = lut[src]
        m = RNG.random(n) < fd_noise                      # natural FD violations (not errors)
        gov[m] = RNG.integers(0, max(card - 5, 5), m.sum())
        cols[f"s{i}"] = src; cols[f"g{i}"] = gov
        planted.append((f"s{i}", f"g{i}"))
    for j in range(d - 2 * k):
        cols[f"n{j}"] = RNG.integers(0, card, n)
    return cols, planted


def inject(cols, planted, rate):
    n = len(next(iter(cols.values()))); y = np.zeros(n, bool)
    out = {c: v.copy() for c, v in cols.items()}
    for _, g in planted:
        idx = np.where(RNG.random(n) < rate)[0]
        repl = out[g][RNG.permutation(n)][idx]
        keep = repl != out[g][idx]
        out[g][idx[keep]] = repl[keep]; y[idx[keep]] = True
    return out, y


def scan(cols, tau, planted):
    """Vectorized discrete scoring: tuple violation score, plus how many planted FDs are
    recovered (strength >= tau). No per-pair dict, so it scales to d=1000."""
    names = list(cols)
    pset = set(planted)
    codes = {c: pd.factorize(cols[c])[0] for c in names}
    cards = {c: int(codes[c].max()) + 1 for c in names}
    cnt = {c: np.bincount(codes[c]) for c in names}
    n = len(codes[names[0]])
    row = np.zeros(n); rec = 0
    for A in names:
        cA, nA, grpA = codes[A], cards[A], cnt[A][codes[A]]
        for B in names:
            if A == B:
                continue
            cB, nB = codes[B], cards[B]
            jc = np.bincount(cA * nB + cB, minlength=nA * nB)
            st = jc.reshape(nA, nB).max(axis=1).sum() / n
            if st >= tau:
                row = np.maximum(row, 1.0 - jc[cA * nB + cB] / grpA)
                if (A, B) in pset:
                    rec += 1
    return row, rec


def main():
    print("Large synthetic stress test (vectorized discrete scoring)")
    print(f"{'set':>4}{'n':>10}{'d':>5}{'%err':>6} | {'CPAD AUROC':>11}{'AUPRC':>7} | "
          f"{'IF AUROC':>9} | {'rec':>4} | {'time(s)':>8}{'mem(MB)':>8}")
    print("-" * 84)
    for name, n, d, k in SETS:
        cols, planted = gen(n, d, k)
        dirty, y = inject(cols, planted, SWAP)
        tracemalloc.start(); t0 = time.perf_counter()
        row, rec = scan(dirty, TAU, planted)
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        # Isolation Forest reference: subsample of rows, capped to <=50 columns (one-hot of
        # the full high-d table is infeasible); it is a density baseline, near chance anyway.
        idx = RNG.choice(n, min(n, 15000), replace=False)
        subcols = list(cols)[:min(len(cols), 50)]
        sub = pd.DataFrame({c: dirty[c][idx] for c in subcols})
        X = OneHotEncoder(handle_unknown="ignore", max_categories=40).fit_transform(sub).toarray()
        ifa = roc_auc_score(y[idx], -IsolationForest(random_state=0, n_estimators=150).fit(X).score_samples(X))
        print(f"{name:>4}{n:>10}{d:>5}{100*y.mean():>5.1f}% | "
              f"{roc_auc_score(y, row):>11.3f}{average_precision_score(y, row):>7.3f} | "
              f"{ifa:>9.3f} | {rec:>2}/{k:<2}| {dt:>8.1f}{peak/1e6:>8.1f}")


if __name__ == "__main__":
    main()
