"""Vectorized confidence/support measures for candidate functional dependencies.

confidence(X -> A) = fraction of tuples whose A-value equals the most frequent A-value
within their X-group (i.e. the share kept if every group adopted its majority value).
A deterministic FD has confidence 1; the measure is robust to a minority of errors
(majority vote per group).
"""
from __future__ import annotations
import pandas as pd


def fd_confidence(df: pd.DataFrame, lhs, rhs: str) -> float:
    lhs = list(lhs)
    if not lhs:
        return float((df[rhs].value_counts() / len(df)).max())
    grp = df.groupby(lhs + [rhs]).size()
    return float(grp.groupby(level=list(range(len(lhs)))).max().sum() / len(df))


def base_rate(df: pd.DataFrame, col: str) -> float:
    """Confidence of the empty LHS: how predictable the column is on its own."""
    return float((df[col].value_counts() / len(df)).max())


def avg_group_size(df: pd.DataFrame, cols) -> float:
    """Mean tuples per LHS group; guards against confidence inflated by singleton groups."""
    cols = list(cols)
    if not cols:
        return float(len(df))
    return float(len(df) / max(1, df.groupby(cols).ngroups))
