"""The Table abstraction: a tabular dataset with inferred column kinds and the
encodings the CPAD models consume.

Column kinds
------------
NUMERIC      : >95% of values parse as numbers and cardinality <= id_cardinality
IDENTIFIER   : cardinality > id_cardinality (keys, names, free text) -- skipped for
               FD/DC discovery, as such columns are not governed by constraints
CATEGORICAL  : everything else

`modeling_columns()` returns the columns usable for constraint discovery (more than
one value, not identifier-like).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

CATEGORICAL, NUMERIC, IDENTIFIER = "categorical", "numeric", "identifier"


class Table:
    def __init__(self, df: pd.DataFrame, name: str = "table", id_cardinality: int = 300):
        self.df = df.reset_index(drop=True)
        self.name = name
        self.columns = list(self.df.columns)
        self.n = len(self.df)
        self.id_cardinality = id_cardinality
        self.kinds = {c: self._infer_kind(c) for c in self.columns}

    @classmethod
    def from_csv(cls, path: str, name: str | None = None, **kwargs) -> "Table":
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df.apply(lambda s: s.str.strip())
        return cls(df, name=name or os.path.splitext(os.path.basename(path))[0], **kwargs)

    # -- column typing -------------------------------------------------------
    def numeric_fraction(self, col: str) -> float:
        return float(pd.to_numeric(self.df[col], errors="coerce").notna().mean())

    def cardinality(self, col: str) -> int:
        return int(self.df[col].nunique())

    def _infer_kind(self, col: str) -> str:
        card = self.cardinality(col)
        if card > self.id_cardinality:
            return IDENTIFIER
        if self.numeric_fraction(col) > 0.95:
            return NUMERIC
        return CATEGORICAL

    def modeling_columns(self) -> list[str]:
        """Columns eligible for FD/DC discovery: constant columns and identifiers excluded."""
        return [c for c in self.columns if 1 < self.cardinality(c) <= self.id_cardinality]

    def numeric_columns(self) -> list[str]:
        return [c for c in self.modeling_columns() if self.kinds[c] == NUMERIC]

    # -- encodings -----------------------------------------------------------
    def codes(self, cols: list[str] | None = None):
        """Integer-encode the given columns. Returns (codes[n, k], cardinalities[k])."""
        cols = cols or self.columns
        mats, cards = [], []
        for c in cols:
            code, uniq = pd.factorize(self.df[c])
            mats.append(code); cards.append(len(uniq))
        return np.stack(mats, axis=1), cards

    def numeric_matrix(self, cols: list[str] | None = None):
        cols = cols if cols is not None else self.numeric_columns()
        if not cols:
            return np.zeros((self.n, 0)), []
        X = np.column_stack([pd.to_numeric(self.df[c], errors="coerce").to_numpy(float) for c in cols])
        return X, cols

    def error_mask_vs(self, clean: "Table") -> np.ndarray:
        """Cell-level ground-truth mask (dirty != clean) for aligned clean/dirty tables."""
        if list(clean.columns) != self.columns or clean.n != self.n:
            raise ValueError("clean table must be row- and column-aligned")
        return self.df.values != clean.df.values

    def __repr__(self) -> str:
        return f"Table({self.name!r}, n={self.n}, cols={len(self.columns)})"
