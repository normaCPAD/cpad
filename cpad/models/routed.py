"""RoutedCPAD -- the complete CPAD system (the one the paper simply calls "CPAD").

It composes several constraint learners with the marginal surprise detector and routes
every column to the most specific applicable model, by priority:

    FD / composite DC   (DiscreteCPAD)   -- equality constraints, most interpretable
    order DC            (OrderCPAD)       -- monotone <, > constraints
    linear DC           (LinearCPAD)      -- contrastive sparse linear constraints
    marginal surprise                     -- fallback for ungoverned columns

The first learner whose `governed_` claims a column scores it; everything else falls
back to the marginal detector. `rules()` aggregates the constraints found by every
learner (FDs and denial constraints alike), feeding the modeling layer.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata

from cpad.core.table import Table
from cpad.models.base import CPADModel
from cpad.models.discrete import DiscreteCPAD
from cpad.detect.marginal import MarginalSurprise


def _rank01(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x) if np.ptp(x) > 0 else np.zeros_like(x, dtype=float)


def _default_learners():
    from cpad.models.order import OrderCPAD
    from cpad.models.linear import LinearCPAD
    return [DiscreteCPAD(max_lhs=2), OrderCPAD(), LinearCPAD()]


class RoutedCPAD(CPADModel):
    name = "CPAD"

    def __init__(self, learners: list[CPADModel] | None = None, marginal: MarginalSurprise | None = None):
        self.learners = learners if learners is not None else _default_learners()
        self.marginal = marginal or MarginalSurprise()

    def fit(self, table: Table) -> "RoutedCPAD":
        self.columns_ = table.columns
        for learner in self.learners:
            learner.fit(table)
        self.marginal.fit(table)
        self.rules_ = [r for learner in self.learners for r in learner.rules()]
        self.owner_ = {}                                  # column -> learner (first to claim it)
        for learner in self.learners:
            for c in getattr(learner, "governed_", set()):
                self.owner_.setdefault(c, learner)
        self.governed_ = set(self.owner_)
        return self

    def score(self, table: Table) -> np.ndarray:
        marg = self.marginal.score(table)
        cache: dict[int, np.ndarray] = {}
        idx = {c: j for j, c in enumerate(table.columns)}
        S = np.zeros((table.n, len(table.columns)))
        for c, j in idx.items():
            owner = self.owner_.get(c)
            if owner is None:
                S[:, j] = marg[:, j]
            else:
                if id(owner) not in cache:
                    cache[id(owner)] = owner.score(table)
                S[:, j] = _rank01(cache[id(owner)][:, j])
        return S
