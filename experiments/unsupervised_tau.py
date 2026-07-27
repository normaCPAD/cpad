"""
Fully UNSUPERVISED CPAD-cat on Hospital.

No real labels are used anywhere except final reporting. Hyperparameter (tau)
selection uses a SELF-SUPERVISED signal: inject synthetic value-swap corruptions
(A.1: replace a cell by another *valid, marginally-plausible* value of the same
column), then pick the tau that best detects those self-injected errors.

We compare:
  - tau* selected SELF-SUPERVISED (no real labels)   <- the unsupervised method
  - tau selected by ORACLE (real labels)             <- upper bound
  - fixed tau = 0.90                                  <- naive default
and report real-error AUROC/AUPRC at cell and row level. IsolationForest for context.
"""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

from _common import DATA as D
norm = lambda df: df.fillna("").map(lambda s: str(s).strip().lower())
dirty = norm(pd.read_csv(D + "hospital_dirty.csv", dtype=str)).reset_index(drop=True)
err = np.load(D + "hospital_errmask.npy")
cols = list(dirty.columns); n = len(dirty)
TAUS = [0.80, 0.85, 0.90, 0.95]

def fd_cell_scores(df, tau):
    cell = np.zeros((n, len(cols)))
    for bi, B in enumerate(cols):
        for A in cols:
            if A == B: continue
            g = df.groupby(A)[B]
            mode = g.transform(lambda s: s.value_counts().idxmax())
            if (df[B].values == mode.values).mean() < tau: continue
            size = g.transform("size").values
            own = df.groupby([A, B])[B].transform("size").values
            cell[:, bi] = np.maximum(cell[:, bi], 1.0 - own / size)
    return cell

def inject_valueswap(df, rate=0.03):
    """A.1 corrupter: replace cells with another VALID value from the column marginal."""
    dc = df.copy(); mask = np.zeros((n, len(cols)), bool)
    for ci, c in enumerate(cols):
        vals = df[c].values
        idx = rng.random(n) < rate
        repl = rng.permutation(vals)                     # valid values from same column
        same = repl == vals; idx &= ~same                # ensure an actual change
        dc.iloc[idx, ci] = repl[idx]; mask[idx, ci] = True
    return dc.reset_index(drop=True), mask

# ---- SELF-SUPERVISED tau selection (no real labels) ----
Dinj, M = inject_valueswap(dirty, rate=0.03)
self_auc = {}
for tau in TAUS:
    s = fd_cell_scores(Dinj, tau)
    self_auc[tau] = roc_auc_score(M.ravel(), s.ravel())
tau_self = max(self_auc, key=self_auc.get)

# ---- evaluate every tau on REAL errors (for oracle + comparison) ----
real = {}
for tau in TAUS:
    s = fd_cell_scores(dirty, tau)
    real[tau] = (roc_auc_score(err.ravel(), s.ravel()),
                 average_precision_score(err.ravel(), s.ravel()),
                 roc_auc_score(err.any(1), s.max(1)))
tau_oracle = max(TAUS, key=lambda t: real[t][0])

Xoh = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dirty).toarray()
if_row = roc_auc_score(err.any(1), np.mean(
    [-IsolationForest(random_state=s, n_estimators=200).fit(Xoh).score_samples(Xoh) for s in range(5)], 0))

print("Self-supervised tau selection (detecting injected value-swaps, NO real labels):")
for tau in TAUS:
    star = "  <- selected" if tau == tau_self else ""
    print(f"   tau={tau:.2f}  self-AUROC={self_auc[tau]:.3f}{star}")
print("=" * 66)
print(f"{'tau choice':<26}{'cellAUROC':>10}{'cellAUPRC':>10}{'rowAUROC':>10}")
print("-" * 56)
for label, tau in [(f"UNSUP self-sup (tau={tau_self})", tau_self),
                   (f"ORACLE labels (tau={tau_oracle})", tau_oracle),
                   ("fixed default (tau=0.90)", 0.90)]:
    a, p, r = real[tau]
    print(f"{label:<26}{a:>10.3f}{p:>10.3f}{r:>10.3f}")
print("-" * 56)
print(f"{'IsolationForest (row)':<26}{'-':>10}{'-':>10}{if_row:>10.3f}")
print("=" * 66)
gap = real[tau_oracle][0] - real[tau_self][0]
print(f"unsupervised vs oracle gap (cell AUROC): {gap:+.3f}")
