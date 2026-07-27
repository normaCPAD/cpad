"""cpad -- learn the denial constraints governing a table and propose its
normalized relational model.

The discovery engine is CPAD (Constrained Predicates for Anomaly Detection): a base
model with several variants (discrete, gated/differentiable, null-space, ensemble,
routed). On top of the discovered functional dependencies and denial constraints,
the `modeling` package computes candidate keys and a 3NF/BCNF decomposition.
"""
from cpad.core.constraint import FunctionalDependency, DenialConstraint, Predicate
from cpad.core.table import Table

__version__ = "0.1.0"
__all__ = ["Table", "FunctionalDependency", "DenialConstraint", "Predicate"]
