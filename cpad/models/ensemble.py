"""EnsembleCPAD -- rank-mean of several CPAD sub-models.

Combines complementary variants (e.g. the precise discrete miner and the globally
stronger gated model). Cell scores are rank-normalized per column and averaged; the
discovered rule set is the union, keeping the highest-confidence rule per target.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata

from cpad.core.table import Table
from cpad.models.base import CPADModel


def _rank01_cols(S: np.ndarray) -> np.ndarray:
    out = np.zeros_like(S, dtype=float)
    for j in range(S.shape[1]):
        col = S[:, j]
        out[:, j] = rankdata(col) / len(col) if np.ptp(col) > 0 else 0.0
    return out


class EnsembleCPAD(CPADModel):
    name = "CPAD-ensemble"

    def __init__(self, models: list[CPADModel]):
        if not models:
            raise ValueError("EnsembleCPAD needs at least one sub-model")
        self.models = list(models)

    def fit(self, table: Table) -> "EnsembleCPAD":
        self.columns_ = table.columns
        for m in self.models:
            m.fit(table)
        best: dict[str, object] = {}
        for m in self.models:
            for fd in m.rules():
                cur = best.get(fd.rhs)
                if cur is None or fd.confidence > cur.confidence:
                    best[fd.rhs] = fd
        self.rules_ = list(best.values())
        self.governed_ = set(best.keys())
        return self

    def score(self, table: Table) -> np.ndarray:
        return np.mean([_rank01_cols(m.score(table)) for m in self.models], axis=0)
