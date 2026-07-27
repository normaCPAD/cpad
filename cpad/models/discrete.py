"""DiscreteCPAD -- the discrete instantiation of CPAD.

Mines functional dependencies by majority vote per group (mode/frequency) and scores
each governed cell by its conditional violation  1 - freq(t.B | t.A).

`max_lhs=1` reproduces the paper's discrete variant (simple FDs only). `max_lhs>=2`
enables composite FDs through the greedy minimal-rule search, so the discrete miner
can also recover constraints like (state, marital_status) -> single_exemp without any
neural model.
"""
from __future__ import annotations
import numpy as np

from cpad.core.table import Table
from cpad.models.base import CPADModel
from cpad.rules.confidence import fd_confidence
from cpad.rules.extract import mine_dependencies


class DiscreteCPAD(CPADModel):
    name = "CPAD-discrete"

    def __init__(self, tau: float = 0.90, lift: float = 0.10, max_lhs: int = 2):
        self.tau, self.lift, self.max_lhs = tau, lift, max_lhs

    def fit(self, table: Table) -> "DiscreteCPAD":
        self.columns_ = table.columns
        cols = table.modeling_columns()
        df = table.df
        order_fn = lambda rhs: sorted((a for a in cols if a != rhs),
                                      key=lambda a: -fd_confidence(df, [a], rhs))
        self.rules_ = mine_dependencies(df, cols, order_fn=order_fn,
                                        tau=self.tau, lift=self.lift, max_lhs=self.max_lhs)
        self.governed_ = {fd.rhs for fd in self.rules_}
        return self

    def score(self, table: Table) -> np.ndarray:
        df = table.df
        n, k = table.n, len(table.columns)
        col_idx = {c: j for j, c in enumerate(table.columns)}
        S = np.zeros((n, k))
        for fd in self.rules_:
            lhs = list(fd.lhs)
            size = df.groupby(lhs)[fd.rhs].transform("size").to_numpy(float)
            own = df.groupby(lhs + [fd.rhs])[fd.rhs].transform("size").to_numpy(float)
            viol = 1.0 - own / size
            j = col_idx[fd.rhs]
            S[:, j] = np.maximum(S[:, j], viol)
        return S
