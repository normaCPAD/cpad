"""Constraint vocabulary.

A denial constraint (DC) forbids the simultaneous satisfaction of a conjunction of
predicates over a pair of tuples:

    DC : not-exists (t1, t2) such that  p_1 and ... and p_m

A functional dependency (FD)  X -> A  is the special case that uses only equality and
inequality predicates:

    FD : not-exists (t1, t2) such that  (and_k t1.X_k = t2.X_k)  and  t1.A != t2.A

so every FD is convertible to a DC (`FunctionalDependency.to_dc`). FDs are the
work-horse for normalization; the general DC form is kept so the engine can also
express order/comparison constraints (<, <=, >, >=).
"""
from __future__ import annotations
from dataclasses import dataclass

EQ, NE, LT, LE, GT, GE = "=", "!=", "<", "<=", ">", ">="
OPS = (EQ, NE, LT, LE, GT, GE)


@dataclass(frozen=True)
class Predicate:
    """A binary predicate `t1.left op t2.right` over two tuples of the same relation."""
    left: str
    op: str
    right: str

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"unknown operator {self.op!r}")

    def __str__(self) -> str:
        return f"t1.{self.left} {self.op} t2.{self.right}"


@dataclass(frozen=True)
class FunctionalDependency:
    """X -> A with a measured confidence and support (fraction of tuples covered)."""
    lhs: tuple[str, ...]
    rhs: str
    confidence: float = 1.0
    support: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "lhs", tuple(self.lhs))

    @property
    def arity(self) -> int:
        return len(self.lhs)

    @property
    def is_composite(self) -> bool:
        return len(self.lhs) > 1

    def to_dc(self) -> "DenialConstraint":
        preds = tuple(Predicate(a, EQ, a) for a in self.lhs) + (Predicate(self.rhs, NE, self.rhs),)
        return DenialConstraint(preds, confidence=self.confidence, support=self.support)

    def __str__(self) -> str:
        body = ", ".join(self.lhs)
        lhs = self.lhs[0] if self.arity == 1 else f"({body})"
        return f"{lhs} -> {self.rhs}"


@dataclass(frozen=True)
class LinearConstraint:
    """A single-tuple linear constraint  sum_i coef_i * x_i ~= offset  (within tolerance).

    The denial-constraint reading is: not-exists a tuple whose weighted sum deviates from
    `offset` by more than `tolerance`. Discovered contrastively (not by PCA), with an L1
    penalty so only a few columns carry weight.
    """
    coefficients: tuple                                   # ((column, coef), ...)
    offset: float = 0.0
    tolerance: float = 0.0
    confidence: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "coefficients", tuple(self.coefficients))

    @property
    def attributes(self) -> set:
        return {c for c, _ in self.coefficients}

    def __str__(self) -> str:
        terms = " ".join(f"{co:+.2f}*{c}" for c, co in self.coefficients)
        return f"{terms.lstrip('+ ')} ~= {self.offset:+.2f}  (+-{self.tolerance:.2g})"


@dataclass(frozen=True)
class DenialConstraint:
    """A conjunction of predicates that must never hold for any pair of tuples."""
    predicates: tuple[Predicate, ...]
    confidence: float = 1.0
    support: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "predicates", tuple(self.predicates))

    @property
    def attributes(self) -> set[str]:
        a: set[str] = set()
        for p in self.predicates:
            a.add(p.left); a.add(p.right)
        return a

    def __str__(self) -> str:
        body = " and ".join(str(p) for p in self.predicates)
        return f"not-exists (t1, t2): {body}"
