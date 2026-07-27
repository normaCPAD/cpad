"""Classical functional-dependency reasoning (Armstrong): attribute closure, minimal
cover, candidate keys. FDs are normalized to (frozenset(lhs), rhs) with a single RHS
attribute, which is all the normalization algorithms need.
"""
from __future__ import annotations
from itertools import combinations

from cpad.core.constraint import FunctionalDependency

FD = tuple  # (frozenset[str], str)


def as_pairs(fds: list[FunctionalDependency]) -> list[FD]:
    return [(frozenset(fd.lhs), fd.rhs) for fd in fds]


def attribute_closure(attrs, fds: list[FD]) -> set:
    """X+ : all attributes functionally determined by `attrs` under `fds`."""
    closure = set(attrs)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in fds:
            if lhs <= closure and rhs not in closure:
                closure.add(rhs); changed = True
    return closure


def minimal_cover(fds: list[FD]) -> list[FD]:
    """A canonical minimal cover: single RHS, no extraneous LHS attribute, no redundant FD."""
    cover = list(dict.fromkeys(fds))                      # dedup, RHS already single
    # 1) remove extraneous LHS attributes
    reduced: list[FD] = []
    for lhs, rhs in cover:
        lhs = set(lhs)
        for a in list(lhs):
            if len(lhs) > 1 and rhs in attribute_closure(lhs - {a}, cover):
                lhs.discard(a)
        reduced.append((frozenset(lhs), rhs))
    # 2) remove redundant FDs
    result = list(dict.fromkeys(reduced))
    i = 0
    while i < len(result):
        lhs, rhs = result[i]
        without = result[:i] + result[i + 1:]
        if rhs in attribute_closure(lhs, without):
            result = without
        else:
            i += 1
    return result


def candidate_keys(all_attrs, fds: list[FD], max_extra: int = 4, limit: int = 16) -> list[frozenset]:
    """Minimal attribute sets whose closure is the whole relation.

    Attributes never on a RHS must belong to every key; the rest are searched by
    increasing size (bounded by `max_extra` / `limit` to stay tractable on wide tables).
    """
    all_attrs = set(all_attrs)
    rhs_attrs = {rhs for _, rhs in fds}
    essential = all_attrs - rhs_attrs                     # appear only on LHS / nowhere
    if attribute_closure(essential, fds) == all_attrs:
        return [frozenset(essential)]
    optional = sorted(all_attrs - essential)
    keys: list[frozenset] = []
    for r in range(1, min(max_extra, len(optional)) + 1):
        for combo in combinations(optional, r):
            cand = essential | set(combo)
            if any(k <= cand for k in keys):              # keep only minimal keys
                continue
            if attribute_closure(cand, fds) == all_attrs:
                keys.append(frozenset(cand))
                if len(keys) >= limit:
                    return keys
        if keys:                                          # smallest keys found at this size
            break
    return keys or [frozenset(all_attrs)]
