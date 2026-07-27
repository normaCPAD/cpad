"""Constraint-guided repair: propose the consistent value for each violating cell.

Repair operators (Appendix "Reparation"):
  FD  L->B      : replace a flagged cell by the group mode b*(t.L)  (majority recovery),
                  but ONLY inside the safe zone -- confidence >= min_confidence, group
                  size >= min_group, a clear majority (mode share > mode_frac), and (for
                  categorical columns) the current value marginally rare. The last gate
                  protects legitimate rare-but-conforming values.
  order DC      : project Y onto the isotonic (monotone) fit of Y on X within each context
                  group, repairing only the cells whose residual is anomalous.
  linear  a.x=c : solve the constraint for the dominant column (orthogonal projection
                  reduced to one cell), repairing the tuples that violate it.

Every change is recorded as an `Edit` (row, column, old, new, rule, confidence) so the
repair is fully auditable and reversible.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from cpad.core.table import NUMERIC
from cpad.core.constraint import FunctionalDependency, DenialConstraint, LinearConstraint


@dataclass
class RepairConfig:
    min_confidence: float = 0.90
    min_group: int = 5
    mode_frac: float = 0.50
    max_rarity: float = 0.05
    repair_fd: bool = True
    repair_order: bool = True
    repair_linear: bool = True


@dataclass
class Edit:
    row: int
    column: str
    old: str
    new: str
    rule: str
    kind: str
    confidence: float


@dataclass
class RepairResult:
    edits: list = field(default_factory=list)
    repaired: pd.DataFrame = None
    by_column: dict = field(default_factory=dict)

    @property
    def n_edits(self) -> int:
        return len(self.edits)


def _fd_edits(df, fd, cfg, kinds, n):
    L, B = list(fd.lhs), fd.rhs
    if any(a not in df.columns for a in L + [B]):
        return []
    g = df.groupby(L)[B]
    mode = g.transform(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0])
    size = g.transform("size").to_numpy()
    share = (df[B].values == mode.values)
    frac = pd.Series(share).groupby([df[c].values for c in L]).transform("mean").to_numpy()
    flag = (df[B].values != mode.values) & (size >= cfg.min_group) & (frac > cfg.mode_frac)
    if kinds is not None and kinds.get(B) != NUMERIC:
        freq = df[B].map(df[B].value_counts() / n).to_numpy()
        flag = flag & (freq < cfg.max_rarity)
    out = []
    for i in np.where(flag)[0]:
        out.append(Edit(int(i), B, str(df[B].iat[i]), str(mode.iat[i]), str(fd), "FD",
                        float(fd.confidence)))
    return out


def _order_edits(df, dc, cfg, k=3.0):
    from sklearn.isotonic import IsotonicRegression
    preds = dc.predicates
    ctx = [p.left for p in preds if p.op == "="]
    ineq = [p for p in preds if p.op in ("<", ">")]
    if len(ineq) < 2:
        return []
    X, Yp = ineq[0].left, ineq[-1]
    Y, inc = Yp.left, (Yp.op == "<")            # X> & Y< forbidden  =>  increasing
    if any(c not in df.columns for c in ctx + [X, Y]):
        return []
    x = pd.to_numeric(df[X], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[Y], errors="coerce").to_numpy(float)
    out = []
    groups = df.groupby(ctx).groups if ctx else {"_": df.index}
    for _, idx in groups.items():
        gi = np.asarray(idx)
        ok = np.isfinite(x[gi]) & np.isfinite(y[gi])
        if ok.sum() < 5:
            continue
        iso = IsotonicRegression(increasing=inc, out_of_bounds="clip")
        yhat = iso.fit_transform(x[gi][ok], y[gi][ok])
        r = np.abs(y[gi][ok] - yhat)
        mad = np.median(r) + 1e-9
        bad = r > k * mad
        for j in np.where(bad)[0]:
            ridx = int(gi[ok][j])
            out.append(Edit(ridx, Y, str(df[Y].iat[ridx]), f"{yhat[j]:.4g}", str(dc), "order",
                            float(dc.confidence)))
    return out


def _linear_edits(df, lc, cfg, tol_mult=1.0):
    cols = [c for c, _ in lc.coefficients]
    if any(c not in df.columns for c in cols):
        return []
    coefs = np.array([c for _, c in lc.coefficients], float)
    dep = int(np.argmax(np.abs(coefs)))                  # the "=1" column to solve for
    M = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in cols])
    resid = M @ coefs - lc.offset
    tol = max(lc.tolerance, 1e-9) * tol_mult
    bad = np.where(np.abs(resid) > tol)[0]
    out = []
    for i in bad:
        target = (lc.offset - (M[i] @ coefs - coefs[dep] * M[i, dep])) / coefs[dep]
        out.append(Edit(int(i), cols[dep], str(df[cols[dep]].iat[i]), f"{target:.4g}",
                        str(lc), "linear", float(lc.confidence)))
    return out


def repair_table(df: pd.DataFrame, rules, config: RepairConfig | None = None, kinds=None) -> RepairResult:
    """`rules` is a list of constraint objects (FunctionalDependency / DenialConstraint /
    LinearConstraint). Returns a RepairResult with the edits and the repaired table."""
    cfg = config or RepairConfig()
    n = len(df)
    edits = []
    for o in rules:
        conf = float(getattr(o, "confidence", 1.0))
        if conf < cfg.min_confidence:
            continue
        if cfg.repair_fd and isinstance(o, FunctionalDependency):
            edits += _fd_edits(df, o, cfg, kinds, n)
        elif cfg.repair_order and isinstance(o, DenialConstraint):
            edits += _order_edits(df, o, cfg)
        elif cfg.repair_linear and isinstance(o, LinearConstraint):
            edits += _linear_edits(df, o, cfg)
    rep = df.copy()
    by_column = {}
    for e in edits:
        rep.iat[e.row, rep.columns.get_loc(e.column)] = e.new
        by_column[e.column] = by_column.get(e.column, 0) + 1
    return RepairResult(edits=edits, repaired=rep, by_column=by_column)
