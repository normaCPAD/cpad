"""Schema synthesis from discovered functional dependencies: 3NF synthesis and BCNF
decomposition. Each output Relation carries its attributes, a key, and the FDs that
hold inside it -- a ready-to-use relational model.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from cpad.modeling.closure import FD, attribute_closure, minimal_cover, candidate_keys


@dataclass
class Relation:
    name: str
    attributes: frozenset
    key: frozenset
    fds: list = field(default_factory=list)              # list[FD] holding within the relation

    def to_dict(self):
        return {"name": self.name,
                "attributes": sorted(self.attributes),
                "key": sorted(self.key),
                "fds": [f"{sorted(l)} -> {r}" for l, r in self.fds]}


def _projected_fds(attrs: frozenset, fds: list[FD]) -> list[FD]:
    return [(l, r) for (l, r) in fds if l <= attrs and r in attrs]


def synthesize_3nf(all_attrs, fds: list[FD]) -> list[Relation]:
    """Bernstein synthesis: one relation per minimal-cover group, plus a key relation."""
    all_attrs = set(all_attrs)
    cover = minimal_cover(fds)
    keys = candidate_keys(all_attrs, cover)

    groups: dict[frozenset, set] = {}
    for lhs, rhs in cover:
        groups.setdefault(lhs, set(lhs)).add(rhs)

    relations: list[Relation] = []
    for i, (lhs, attrs) in enumerate(groups.items(), 1):
        attrs = frozenset(attrs)
        relations.append(Relation(f"R{i}", attrs, frozenset(lhs), _projected_fds(attrs, cover)))

    # ensure some relation contains a candidate key (lossless join / full coverage)
    if keys and not any(any(k <= r.attributes for k in keys) for r in relations):
        k = min(keys, key=len)
        relations.append(Relation(f"R{len(relations) + 1}", frozenset(k), frozenset(k), []))

    # drop relations whose attributes are a subset of another's
    pruned = [r for r in relations
              if not any(r is not o and r.attributes < o.attributes for o in relations)]
    for i, r in enumerate(pruned, 1):
        r.name = f"R{i}"
    return pruned


def decompose_bcnf(all_attrs, fds: list[FD]) -> list[Relation]:
    """Recursive BCNF decomposition: split on any FD X -> A whose X is not a superkey."""
    all_attrs = frozenset(all_attrs)
    cover = minimal_cover(fds)

    def is_superkey(x, attrs, local):
        return attribute_closure(x, local) >= attrs

    def decompose(attrs):
        local = _projected_fds(attrs, cover)
        for lhs, rhs in local:
            if rhs in attrs and lhs < attrs and not is_superkey(lhs, attrs, local):
                closed = frozenset(attribute_closure(lhs, local)) & attrs
                r1 = closed                                # X+ (contains the violating FD)
                r2 = (attrs - closed) | frozenset(lhs)     # rest plus X (lossless join)
                if r1 == attrs or r2 == attrs:
                    continue
                return decompose(r1) + decompose(r2)
        key = candidate_keys(attrs, local)
        return [Relation("", attrs, key[0] if key else attrs, local)]

    rels = decompose(all_attrs)
    uniq: list[Relation] = []
    for r in rels:
        if not any(r.attributes == o.attributes for o in uniq):
            uniq.append(r)
    for i, r in enumerate(uniq, 1):
        r.name = f"R{i}"
    return uniq
