"""Ablations of the differentiable gated model on Hospital: the contrastive corruption
and the L1 gate penalty are each necessary, and performance is stable across a range of
the L1 weight. Addresses the "not enough ablation / lambda sensitivity" concern.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _common import DATA
from cpad.core.table import Table
from cpad.models import GatedCPAD


def norm(df):
    return df.fillna("").apply(lambda s: s.astype(str).str.strip().str.lower())


dirty = norm(pd.read_csv(DATA + "hospital_dirty.csv", dtype=str)).reset_index(drop=True)
err = np.load(DATA + "hospital_errmask.npy")
yrow = err.any(1)


def evaluate(**kw):
    m = GatedCPAD(epochs=200, **kw).fit(Table(dirty))
    S = m.score(Table(dirty))
    return roc_auc_score(err.ravel(), S.ravel()), roc_auc_score(yrow, S.max(1)), len(m.rules_)


if __name__ == "__main__":
    print("Ablations of the gated model on Hospital (200 epochs)")
    print(f"{'configuration':<28}{'cellAUROC':>10}{'rowAUROC':>10}{'#rules':>8}")
    print("-" * 56)
    for name, kw in [("full (corrupt=.15, L1=.3)", dict(corrupt_p=0.15, l1=0.3)),
                     ("no corruption (corrupt=0)", dict(corrupt_p=0.0, l1=0.3)),
                     ("no L1 (L1=0)",              dict(corrupt_p=0.15, l1=0.0))]:
        ca, ra, nr = evaluate(**kw)
        print(f"{name:<28}{ca:>10.3f}{ra:>10.3f}{nr:>8}")
    print("\nL1 weight sweep (corrupt=.15)")
    print(f"{'lambda':<28}{'cellAUROC':>10}{'rowAUROC':>10}{'#rules':>8}")
    print("-" * 56)
    for lam in [0.0, 0.1, 0.3, 0.6, 1.0]:
        ca, ra, nr = evaluate(corrupt_p=0.15, l1=lam)
        print(f"{lam:<28.2f}{ca:>10.3f}{ra:>10.3f}{nr:>8}")
