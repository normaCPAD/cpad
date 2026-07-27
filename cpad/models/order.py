"""OrderCPAD -- the order/inequality instantiation of CPAD.

Discovers *order dependencies*: pairs of numeric columns (X, Y), optionally within a
categorical context C, whose values move monotonically together. Such a dependency is
the denial constraint

    increasing :  not-exists (t1, t2):  t1.C = t2.C  and  t1.X > t2.X  and  t1.Y < t2.Y
    decreasing :  not-exists (t1, t2):  t1.C = t2.C  and  t1.X > t2.X  and  t1.Y > t2.Y

i.e. genuine `<` / `>` predicates, not equality -- the part of the DC formalism that FD
mining cannot express. Monotonicity strength is estimated by Kendall's tau (globally,
or as a size-weighted average within context groups). A pair is kept when it is
monotone but NOT already an exact FD, so the order constraint adds information.

Native scoring : within each context group, fit a monotone (isotonic) regression
Y ~ X and flag the cells whose Y breaks the monotonic trend (studentized residual).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from cpad.core.table import Table, NUMERIC
from cpad.core.constraint import DenialConstraint, Predicate, EQ, GT, LT
from cpad.models.base import CPADModel
from cpad.rules.confidence import fd_confidence


class OrderCPAD(CPADModel):
    name = "CPAD-order"

    def __init__(self, tau_threshold: float = 0.6, ctx_cap: int = 60, min_group: int = 30,
                 min_distinct: int = 5, max_functional: float = 0.95, sample: int = 8000, seed: int = 0):
        self.tau_threshold = tau_threshold
        self.ctx_cap = ctx_cap
        self.min_group = min_group
        self.min_distinct = min_distinct              # X must take enough values (else tau is degenerate)
        self.max_functional = max_functional          # skip pairs already (near-)exact FDs
        self.sample = sample
        self.seed = seed

    # -- monotonicity strength ----------------------------------------------
    def _conditional_tau(self, df, X, Y, C):
        num, den = 0.0, 0
        for _, g in df.groupby(C):
            if len(g) < self.min_group:
                continue
            t, _ = kendalltau(g[X], g[Y])
            if t == t:                                # not NaN
                num += t * len(g); den += len(g)      # signed, size-weighted
        return num / den if den else 0.0

    def fit(self, table: Table) -> "OrderCPAD":
        self.columns_ = table.columns
        num = table.numeric_columns()
        cats = [c for c in table.modeling_columns()
                if table.kinds[c] != NUMERIC and table.cardinality(c) <= self.ctx_cap]
        df = table.df
        if len(df) > self.sample:
            df = df.sample(self.sample, random_state=self.seed).reset_index(drop=True)
        dn = df.copy()
        for c in num:
            dn[c] = pd.to_numeric(dn[c], errors="coerce")

        self.specs_ = {}                              # Y -> (X, context, increasing, strength)
        rules = []
        seen = set()
        for X in num:
            for Y in num:
                if X == Y or (Y, X) in seen:
                    continue
                seen.add((X, Y))
                m = dn[X].notna() & dn[Y].notna()
                tg, _ = kendalltau(dn[X][m], dn[Y][m])
                best_c, best_t = None, (tg if tg == tg else 0.0)
                for C in cats:                        # is it stronger inside some context?
                    tc = self._conditional_tau(dn[m], X, Y, C)
                    if abs(tc) > abs(best_t):
                        best_c, best_t = C, tc
                if abs(best_t) < self.tau_threshold:
                    continue
                # X must vary enough (per context group) -- otherwise tau is degenerate
                # (e.g. two points per group) and the "order" is really an FD in disguise
                distinct = (dn.groupby(best_c)[X].nunique().mean() if best_c
                            else dn[X].nunique())
                if distinct < self.min_distinct:
                    continue
                if fd_confidence(df, [X], Y) >= self.max_functional:
                    continue                          # already an exact FD: no new information
                inc = best_t > 0
                strength = (abs(best_t) + 1) / 2      # fraction of concordant pairs
                self.specs_[Y] = (X, best_c, inc, strength)
                yop = LT if inc else GT
                preds = ([Predicate(best_c, EQ, best_c)] if best_c else []) + \
                        [Predicate(X, GT, X), Predicate(Y, yop, Y)]
                rules.append(DenialConstraint(tuple(preds), confidence=round(strength, 4)))
        self.rules_ = rules
        self.governed_ = set(self.specs_.keys())
        return self

    # -- scoring: deviation from the monotone trend -------------------------
    def score(self, table: Table) -> np.ndarray:
        from sklearn.isotonic import IsotonicRegression
        S = np.zeros((table.n, len(table.columns)))
        idx = {c: j for j, c in enumerate(table.columns)}
        for Y, (X, C, inc, _strength) in self.specs_.items():
            x = pd.to_numeric(table.df[X], errors="coerce").to_numpy(float)
            y = pd.to_numeric(table.df[Y], errors="coerce").to_numpy(float)
            resid = np.zeros(table.n)
            groups = table.df[C] if C else pd.Series(["_"] * table.n)
            for _, gi in table.df.groupby(groups).groups.items():
                gi = np.asarray(gi)
                xs, ys = x[gi], y[gi]
                ok = np.isfinite(xs) & np.isfinite(ys)
                if ok.sum() < 5:
                    continue
                iso = IsotonicRegression(increasing=inc, out_of_bounds="clip")
                yhat = iso.fit_transform(xs[ok], ys[ok])
                r = np.abs(ys[ok] - yhat)
                mad = np.median(r) + 1e-9
                resid[gi[ok]] = r / mad
            S[:, idx[Y]] = rankdata(resid) / len(resid)
        return S
