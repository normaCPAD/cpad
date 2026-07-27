"""Constraint-guided repair: not just flag a violating cell, but propose the consistent
value. For a (possibly composite) FD L->B, a flagged cell is repaired to the majority
value (mode) of its L-group. We measure, against ground truth (clean vs dirty), the
repair precision/recall and the error-rate reduction.

Datasets with aligned clean/dirty: Tax (wide) and Hospital (clean is long tid/attribute).
"""
import sys, numpy as np, pandas as pd
from _common import DATA
from cpad.core.table import Table
from cpad.models import DiscreteCPAD

pd.options.mode.chained_assignment = None
norm = lambda df: df.apply(lambda s: s.astype(str).str.strip().str.lower())


def load_tax(n=40000, seed=0):
    cl = norm(pd.read_csv(DATA + "tax/clean.csv", dtype=str).fillna(""))
    dy = norm(pd.read_csv(DATA + "tax/dirty.csv", dtype=str).fillna(""))
    idx = np.random.default_rng(seed).choice(len(dy), min(n, len(dy)), replace=False)
    return dy.iloc[idx].reset_index(drop=True), cl.iloc[idx].reset_index(drop=True)


def load_hospital():
    dy = norm(pd.read_csv(DATA + "hospital_dirty.csv", dtype=str).fillna("")).reset_index(drop=True)
    long = pd.read_csv(DATA + "hospital_clean.csv", dtype=str).fillna("")
    wide = long.pivot(index="tid", columns="attribute", values="correct_val")
    wide.index = wide.index.astype(int); wide = wide.sort_index().reset_index(drop=True)
    cols = [c for c in dy.columns if c in wide.columns]
    return norm(dy[cols]), norm(wide[cols].reset_index(drop=True))


def repair(dirty, fds, min_group=5, mode_frac=0.5, rare=0.05):
    """Repair cell (t,B) to its L-group mode only when (i) the majority is a clear, well-
    supported winner (group size >= min_group, mode share > mode_frac) AND (ii) the current
    value is marginally rare (freq < `rare`) -- the typo signature. The second gate avoids
    'repairing' a legitimate rare-but-conforming value, the paper's central distinction."""
    rep = dirty.copy()
    governed = {}
    n = len(dirty)
    for fd in fds:
        L, B = list(fd.lhs), fd.rhs
        g = dirty.groupby(L)[B]
        mode = g.transform(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0])
        size = g.transform("size").values
        share = (dirty[B].values == mode.values)
        frac = pd.Series(share).groupby([dirty[c].values for c in L]).transform("mean").values
        freq = dirty[B].map(dirty[B].value_counts() / n).values
        flagged = ((dirty[B].values != mode.values) & (size >= min_group)
                   & (frac > mode_frac) & (freq < rare))
        rep.loc[flagged, B] = mode.values[flagged]
        governed.setdefault(B, set()).update(np.where(flagged)[0])
    return rep, governed


def evaluate(name, dirty, clean, conf=0.9):
    fds = [r for r in DiscreteCPAD(max_lhs=2, tau=0.9).fit(Table(dirty, name=name)).rules()
           if r.confidence >= conf]
    rep, governed = repair(dirty, fds)
    gcols = sorted(governed)
    print(f"\n==== {name} : {len(fds)} FD, {len(gcols)} colonnes gouvernees ====")
    print(f"{'colonne':16}{'err_av':>8}{'err_ap':>8}{'corrig':>8}{'casse':>7}{'prec':>7}{'rec':>7}")
    tot_before = tot_after = 0
    for c in gcols:
        true_err = dirty[c].values != clean[c].values            # real errors (ground truth)
        repaired = rep[c].values != dirty[c].values              # cells we changed
        now_err = rep[c].values != clean[c].values               # errors remaining after repair
        corrected = int((true_err & repaired & (rep[c].values == clean[c].values)).sum())
        broken = int((~true_err & repaired).sum())               # we changed a correct cell
        prec = corrected / max(1, repaired.sum())
        rec = corrected / max(1, true_err.sum())
        print(f"{c:16}{true_err.sum():>8}{now_err.sum():>8}{corrected:>8}{broken:>7}{prec:>7.2f}{rec:>7.2f}")
        tot_before += int(true_err.sum()); tot_after += int(now_err.sum())
    print(f"  TOTAL erreurs (colonnes gouvernees) : {tot_before} -> {tot_after}  "
          f"(reduction {100*(1-tot_after/max(1,tot_before)):.1f}%)")


if __name__ == "__main__":
    dy, cl = load_tax()
    evaluate("Tax", dy, cl)
    try:
        dy, cl = load_hospital()
        evaluate("Hospital", dy, cl)
    except Exception as e:
        print("Hospital:", e)
