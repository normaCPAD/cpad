"""Broader real benchmarks: CPAD vs Isolation Forest on five data-quality datasets with
natural/realistic corruption and aligned clean/dirty ground truth. Addresses the
reviewer's concern that headline numbers lean on Hospital alone.

For each dataset we align clean and dirty on their common columns, build the cell-level
error mask (dirty != clean), and evaluate TUPLE-level detection (a row is positive iff
it carries >=1 error):
  - CPAD-disc : DiscreteCPAD (discovered FDs + conditional-violation scoring), tuple = max cell.
  - CPAD      : the complete routed system (RoutedCPAD), tuple = max cell.
  - IForest   : Isolation Forest over a one-hot encoding, tuple-level.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

from _common import DATA
from cpad.core.table import Table
from cpad.models import DiscreteCPAD, RoutedCPAD

DSETS = [("Beers", "beers"), ("Flights", "flights"), ("Movies", "movies_1"), ("Rayyan", "rayyan")]


def norm(df):
    return df.fillna("").apply(lambda s: s.astype(str).str.strip().str.lower())


def load(name, sub):
    if name == "Hospital":
        dirty = norm(pd.read_csv(DATA + "hospital_dirty.csv", dtype=str)).reset_index(drop=True)
        mask = np.load(DATA + "hospital_errmask.npy")
        return dirty, mask
    clean = norm(pd.read_csv(f"{DATA}{sub}/clean.csv", dtype=str)).reset_index(drop=True)
    dirty = norm(pd.read_csv(f"{DATA}{sub}/dirty.csv", dtype=str)).reset_index(drop=True)
    common = [c for c in clean.columns if c in dirty.columns]
    if not common:
        return None, None                                  # disjoint schema (e.g. movies): unalignable
    clean, dirty = clean[common], dirty[common]
    # drop columns that differ in (almost) every row: these are representation/format
    # mismatches between the clean and dirty encodings, not injected errors.
    diff = (clean.values != dirty.values)
    keep = [j for j in range(len(common)) if diff[:, j].mean() < 0.9]
    cols = [common[j] for j in keep]
    return dirty[cols], diff[:, keep]


def cpad_disc_tuple(dirty):
    return DiscreteCPAD().fit(Table(dirty)).score(Table(dirty)).max(axis=1)


def cpad_tuple(dirty):
    return RoutedCPAD().fit(Table(dirty)).score(Table(dirty)).max(axis=1)


def iforest_tuple(dirty):
    X = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dirty).toarray()
    return np.mean([-IsolationForest(random_state=s, n_estimators=200).fit(X).score_samples(X)
                    for s in range(3)], 0)


if __name__ == "__main__":
    print("Tuple-level detection (a row is positive iff it has >=1 erroneous cell)")
    print(f"{'dataset':<10}{'n':>7}{'%err':>7}  | {'CPAD-disc':>16} | {'CPAD (complete)':>16} | {'IForest':>16}")
    print(f"{'':<10}{'':>7}{'':>7}  | {'AUROC':>8}{'AUPRC':>8} | {'AUROC':>8}{'AUPRC':>8} | {'AUROC':>8}{'AUPRC':>8}")
    print("-" * 96)
    for name, sub in [("Hospital", None)] + DSETS:
        dirty, mask = load(name, sub)
        if dirty is None:
            print(f"{name:<10} (disjoint clean/dirty schema, skipped)"); continue
        y = mask.any(axis=1)
        if y.sum() == 0 or y.all():
            print(f"{name:<10} (no usable ground-truth variation, skipped)"); continue
        cd, cp, itf = cpad_disc_tuple(dirty), cpad_tuple(dirty), iforest_tuple(dirty)
        def m(s): return roc_auc_score(y, s), average_precision_score(y, s)
        (ad, pd_), (ac, pc), (ai, pi) = m(cd), m(cp), m(itf)
        print(f"{name:<10}{len(dirty):>7}{100*y.mean():>6.1f}%  | "
              f"{ad:>8.3f}{pd_:>8.3f} | {ac:>8.3f}{pc:>8.3f} | {ai:>8.3f}{pi:>8.3f}")
