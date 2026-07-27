"""Marginal surprise detector -- the per-column outlier score CPAD routes to for
columns NOT governed by any constraint.

It is the marginal counterpart of CPAD's conditional violation: where the latter
measures -log P(t.B | sources), this estimates -log p(t.B) under the column's own law.

  numeric      : robust studentized residual |x - median| / MAD   (proportional to z)
  categorical  : max(empirical rarity, length z-score, digit-share z-score)

Scores are rank-normalized to [0, 1] per column so they compose with the conditional
scores in the routed model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cpad.core.table import Table, NUMERIC


def _rank01(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x)


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return np.abs(x - med) / mad


def _categorical_surprise(s: pd.Series) -> np.ndarray:
    n = len(s)
    rarity = 1.0 - s.map(s.value_counts() / n).to_numpy(float)
    length = _robust_z(s.str.len().to_numpy(float))
    digits = _robust_z((s.str.count(r"\d") / (s.str.len() + 1)).to_numpy(float))
    return np.maximum.reduce([_rank01(rarity), _rank01(length), _rank01(digits)])


class MarginalSurprise:
    """Type-aware, fully unsupervised per-column surprise score."""

    def fit(self, table: Table) -> "MarginalSurprise":
        self.columns_ = table.columns
        return self

    def score(self, table: Table) -> np.ndarray:
        n, k = table.n, len(table.columns)
        out = np.zeros((n, k))
        for j, c in enumerate(table.columns):
            if table.kinds[c] == NUMERIC:
                x = pd.to_numeric(table.df[c], errors="coerce").to_numpy(float)
                x = np.where(np.isfinite(x), x, np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0)
                out[:, j] = _rank01(_robust_z(x))
            else:
                out[:, j] = _categorical_surprise(table.df[c].astype(str))
        return out
