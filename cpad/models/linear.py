"""LinearCPAD -- the numeric instantiation of CPAD, discovered CONTRASTIVELY (no PCA).

A linear constraint is a direction `a` that is tight on the (mostly clean) data but is
broken by the value-swap corrupter that independently shuffles each column. We look for
directions that maximize the contrastive separation

    J(a) = a^T (Sigma_corrupt - Sigma_clean) a ,   ||a|| = 1 ,   with an L1 penalty,

solved by sparse power iteration with soft-thresholding (and deflation for several
constraints). This differs from PCA in three ways that matter:

  * the signal is the CORRUPTER CONTRAST, not raw minimum variance -> it targets columns
    that are normally tied but broken by corruption (genuine constraints);
  * an L1 penalty makes each constraint SPARSE and readable (e.g. total = qty * price),
    not a dense eigenvector;
  * tightness/scale are estimated by the MEDIAN of squared projections, so a few real
    violations cannot blur the constraint (robust where PCA is not).

Score: max_r |a_r . x| / robust_scale_r. Extract: the sparse coefficients as a
LinearConstraint. Uses the same A.1 contrastive corrupter as the gated categorical model.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata

from cpad.core.table import Table
from cpad.core.constraint import LinearConstraint
from cpad.models.base import CPADModel


def _soft_threshold(a: np.ndarray, t: float) -> np.ndarray:
    return np.sign(a) * np.maximum(np.abs(a) - t, 0.0)


class LinearCPAD(CPADModel):
    name = "CPAD-linear"

    def __init__(self, n_constraints: int = 3, l1: float = 0.15, iters: int = 300,
                 tight: float = 0.05, contrast: float = 0.25, winsor: float = 4.0,
                 min_coef: float = 0.1, seed: int = 0):
        self.n_constraints = n_constraints
        self.l1 = l1                                       # relative soft-threshold (sparsity)
        self.iters = iters
        self.tight = tight                                # max clean (robust) variance to accept
        self.contrast = contrast                          # min corrupt-minus-clean separation
        self.winsor = winsor                              # clip (in robust-sd units) for a robust M
        self.min_coef = min_coef                          # prune tiny coefficients when reporting
        self.seed = seed

    # -- robust standardization ---------------------------------------------
    def _standardize(self, X):
        med = np.nanmedian(X, axis=0)
        Xf = np.where(np.isfinite(X), X, med)
        mad = np.median(np.abs(Xf - med), axis=0) * 1.4826 + 1e-9
        return (Xf - med) / mad, med, mad

    def fit(self, table: Table) -> "LinearCPAD":
        self.columns_ = table.columns
        X, self.cols_ = table.numeric_matrix()
        if X.shape[1] < 2:
            self.cols_ = []; self.dirs_ = []; self.rules_ = []; self.governed_ = set()
            return self
        self._X = X
        Z, self.med_, self.mad_ = self._standardize(X)
        rng = np.random.default_rng(self.seed)
        Zc = np.column_stack([Z[rng.permutation(len(Z)), j] for j in range(Z.shape[1])])  # value-swap corrupter
        W = self.winsor                                   # clip extremes so a few errors can't tilt M
        M = np.cov(np.clip(Zc, -W, W).T) - np.cov(np.clip(Z, -W, W).T)
        M = np.atleast_2d(M)
        # shift to PSD so power iteration targets the MOST-POSITIVE eigenvalue (the
        # constraint = tight-on-clean, broken-on-corrupt) rather than the largest in magnitude
        shift = float(np.abs(M).sum(axis=1).max())        # Gershgorin bound; same eigenvectors
        Mp = M + shift * np.eye(M.shape[0])

        d = Z.shape[1]
        dirs = []
        for _ in range(self.n_constraints):
            a = rng.normal(size=d); a /= np.linalg.norm(a)
            for _ in range(self.iters):                   # sparse power iteration (ISTA-style)
                a = Mp @ a
                if not np.any(a):
                    break
                a = _soft_threshold(a, self.l1 * np.max(np.abs(a)))
                nrm = np.linalg.norm(a)
                if nrm < 1e-9:
                    break
                a = a / nrm
            if np.linalg.norm(a) < 1e-9:
                break
            clean = float(np.median((Z @ a) ** 2))        # robust: a few errors don't inflate it
            corrupt = float(np.median((Zc @ a) ** 2))
            if clean < self.tight and corrupt - clean > self.contrast:
                dirs.append((a, clean))
            Mp = Mp - (a @ Mp @ a) * np.outer(a, a)        # deflate

        self.dirs_ = dirs
        self.governed_ = set(self.cols_) if dirs else set()
        self._build_rules()
        return self

    def _winsor_col(self, col):
        m = np.median(col); md = np.median(np.abs(col - m)) * 1.4826 + 1e-9
        return np.clip(col, m - self.winsor * md, m + self.winsor * md)

    def _build_rules(self):
        """L1 selects the support; a robust OLS on that support recovers clean,
        debiased coefficients (e.g. total = qty + price rather than distorted weights)."""
        self.rules_ = []
        X = self._X
        for a, clean in self.dirs_:
            support = [i for i in range(len(a)) if abs(a[i]) >= self.min_coef * np.max(np.abs(a))]
            if len(support) < 2:
                continue
            dep = max(support, key=lambda i: abs(a[i]))    # the column with the strongest loading
            others = [i for i in support if i != dep]
            A = np.column_stack([self._winsor_col(X[:, i]) for i in others] + [np.ones(len(X))])
            y = self._winsor_col(X[:, dep])
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = X[:, dep] - (np.column_stack([X[:, i] for i in others] + [np.ones(len(X))]) @ beta)
            coefs = [(self.cols_[dep], 1.0)] + \
                    [(self.cols_[others[k]], float(round(-beta[k], 3))) for k in range(len(others))]
            tol = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
            self.rules_.append(LinearConstraint(tuple(coefs), offset=round(float(beta[-1]), 3),
                                                tolerance=round(tol, 4), confidence=round(1.0 - clean, 4)))

    def score(self, table: Table) -> np.ndarray:
        if not self.cols_:
            return np.zeros((table.n, len(table.columns)))
        X, _ = table.numeric_matrix(self.cols_)
        Z = (np.where(np.isfinite(X), X, self.med_) - self.med_) / self.mad_
        row = np.zeros(table.n)
        for a, clean in self.dirs_:
            row = np.maximum(row, np.abs(Z @ a) / (np.sqrt(clean) + 1e-9))
        s = rankdata(row) / len(row)
        S = np.zeros((table.n, len(table.columns)))
        idx = {c: j for j, c in enumerate(table.columns)}
        for c in self.cols_:
            S[:, idx[c]] = s
        return S
