"""Controlled generality: inject the TARGET error type (value-swap DC violations) into
many real CLEAN schemas and measure detection. This isolates the question "does CPAD's
capability generalize across datasets?" from "does it detect non-relational errors?"
(which it does not claim to).

For each clean table we replace, at rate r, a cell by another row's value of the same
column (a marginally-plausible value that breaks the inter-column regularities). We
report tuple-level AUROC/AUPRC for the complete routed CPAD vs Isolation Forest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

from _common import DATA
from cpad.core.table import Table
from cpad.models import RoutedCPAD

RNG = np.random.default_rng(0)
SETS = [("Hospital", "hospital_clean_wide.csv"), ("Adult", "adult.csv"),
        ("Beers", "beers/clean.csv"), ("Flights", "flights/clean.csv"),
        ("Rayyan", "rayyan/clean.csv"), ("Tax", "tax/clean.csv")]


def norm(df):
    return df.fillna("").apply(lambda s: s.astype(str).str.strip().str.lower())


def fd_conf(df, A, B):
    g = df.groupby(A)[B]
    return (g.transform(lambda s: s.value_counts().idxmax()).values == df[B].values).mean()


def governed_columns(df, tau=0.90, cap=300):
    """Columns that are the RHS of an approximate FD A->B (conf >= tau), so a value-swap
    there is a genuine denial-constraint violation. Model-independent (plain FD confidence)."""
    cols = [c for c in df.columns if 1 < df[c].nunique() <= cap]
    gov = []
    for B in cols:
        if any(A != B and fd_conf(df, A, B) >= tau for A in cols):
            gov.append(B)
    return gov


def inject_swaps(df, gov, rate=0.10):
    df = df.reset_index(drop=True).copy()
    n = len(df); y = np.zeros(n, bool)
    for c in gov:                                   # inject only into governed columns
        idx = np.where(RNG.random(n) < rate)[0]
        repl = df[c].values[RNG.permutation(n)][idx]
        keep = repl != df[c].values[idx]
        df.loc[idx[keep], c] = repl[keep]
        y[idx[keep]] = True
    return df, y


def tuple_metrics(y, s):
    return roc_auc_score(y, s), average_precision_score(y, s)


if __name__ == "__main__":
    print("Value-swap DC-violations injected into FD-governed columns of clean real schemas (rate 10%)")
    print(f"{'dataset':<10}{'n':>7}{'gov':>4}{'%err':>7}  | {'CPAD (complete)':>17} | {'IForest':>15}")
    print(f"{'':<28}  | {'AUROC':>9}{'AUPRC':>8} | {'AUROC':>7}{'AUPRC':>8}")
    print("-" * 78)
    for name, path in SETS:
        df = norm(pd.read_csv(DATA + path, dtype=str))
        if len(df) > 5000:
            df = df.sample(5000, random_state=0)
        gov = governed_columns(df)
        if not gov:
            print(f"{name:<10}{len(df):>7}{0:>4}   (no FD-governed columns)"); continue
        dirty, y = inject_swaps(df, gov)
        t = Table(dirty)
        cp = RoutedCPAD().fit(t).score(t).max(axis=1)
        X = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dirty).toarray()
        itf = -IsolationForest(random_state=0, n_estimators=200).fit(X).score_samples(X)
        (ac, pc), (ai, pi) = tuple_metrics(y, cp), tuple_metrics(y, itf)
        print(f"{name:<10}{len(dirty):>7}{len(gov):>4}{100*y.mean():>6.1f}%  | "
              f"{ac:>9.3f}{pc:>8.3f} | {ai:>7.3f}{pi:>8.3f}")
