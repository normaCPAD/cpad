"""Turn ranked candidate sources into a minimal functional dependency.

`minimal_rule` keeps an FD SIMPLE (single LHS attribute) unless no single source
reaches the confidence threshold; only then does it grow a composite LHS, and only
while each added attribute yields a real confidence jump AND keeps groups non-singleton
(mean size >= min_group). This prevents the trivial inflation of confidence toward 1
that comes from conditioning on ever more attributes.

`mine_dependencies` applies it to every candidate target. The per-target candidate
ORDER is supplied by the caller: confidence-ranked for the discrete miner, gate-ranked
for the gated model -- so the same extractor serves every CPAD variant.
"""
from __future__ import annotations
from typing import Callable, Sequence
import pandas as pd

from cpad.core.constraint import FunctionalDependency
from cpad.rules.confidence import fd_confidence, base_rate, avg_group_size


def minimal_rule(df: pd.DataFrame, rhs: str, order: Sequence[str],
                 tau: float = 0.90, lift: float = 0.10, jump: float = 0.05,
                 min_group: float = 3.0, max_lhs: int = 3) -> FunctionalDependency | None:
    """Smallest LHS (taken from `order`) reaching confidence `tau` for target `rhs`."""
    cands = [a for a in order if a != rhs]
    if not cands:
        return None
    singles = sorted(((a, fd_confidence(df, [a], rhs)) for a in cands), key=lambda z: -z[1])
    best_a, best_c = singles[0]
    base = base_rate(df, rhs)

    if best_c >= tau:                                     # a single attribute already explains rhs
        lhs, conf = (best_a,), best_c
    else:                                                 # build a minimal composite LHS
        lhs_list, conf = [best_a], best_c
        for a, _ in singles[1:]:
            if len(lhs_list) >= max_lhs:
                break
            c = fd_confidence(df, lhs_list + [a], rhs)
            if c > conf + jump and avg_group_size(df, lhs_list + [a]) >= min_group:
                lhs_list.append(a); conf = c
            if conf >= tau:
                break
        lhs = tuple(lhs_list)

    if conf >= tau and conf - base >= lift:
        return FunctionalDependency(lhs, rhs, confidence=round(conf, 4), support=1.0)
    return None


def mine_dependencies(df: pd.DataFrame, columns: Sequence[str],
                      order_fn: Callable[[str], Sequence[str]] | None = None,
                      **kwargs) -> list[FunctionalDependency]:
    """Discover one minimal FD per target column. `order_fn(target)` ranks candidate
    sources; default ranks them by single-source confidence."""
    cols = list(columns)
    rules: list[FunctionalDependency] = []
    for rhs in cols:
        if order_fn is not None:
            order = [a for a in order_fn(rhs) if a in cols]
        else:
            order = sorted((a for a in cols if a != rhs),
                           key=lambda a: -fd_confidence(df, [a], rhs))
        fd = minimal_rule(df, rhs, order, **kwargs)
        if fd is not None:
            rules.append(fd)
    return rules
