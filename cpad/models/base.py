"""CPADModel -- the common architecture every variant inherits.

A CPAD model is fitted on a Table and then offers three things:

  rules(table)         -> the denial constraints / FDs it discovered
  score(table)         -> a (n, d) matrix of per-cell violation scores in [0, 1]
  explain(table, k)    -> the k most violating cells, with the rule responsible

`tuple_scores` and `explain` are defined once here in terms of `score`/`rules`, so the
subclasses only implement `fit`, `score`, and (optionally) `rules`.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from cpad.core.table import Table
from cpad.core.constraint import FunctionalDependency


class CPADModel(ABC):
    name = "CPAD"

    @abstractmethod
    def fit(self, table: Table) -> "CPADModel":
        ...

    @abstractmethod
    def score(self, table: Table) -> np.ndarray:
        """Per-cell violation scores, shape (table.n, len(table.columns)), in [0, 1]."""

    def rules(self) -> list[FunctionalDependency]:
        return list(getattr(self, "rules_", []))

    # -- derived, shared across variants -------------------------------------
    def tuple_scores(self, table: Table) -> np.ndarray:
        return self.score(table).max(axis=1)

    def fit_score(self, table: Table) -> np.ndarray:
        return self.fit(table).score(table)

    def explain(self, table: Table, k: int = 10):
        """Return the k most violating cells as dicts {row, column, value, score}."""
        S = self.score(table)
        flat = np.argsort(S, axis=None)[::-1][:k]
        out = []
        for idx in flat:
            i, j = np.unravel_index(idx, S.shape)
            col = table.columns[j]
            out.append({"row": int(i), "column": col,
                        "value": table.df.iloc[int(i)][col], "score": float(S[i, j])})
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
