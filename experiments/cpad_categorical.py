"""
CPAD-categorical on Hospital (HoloClean): error detection where the anomaly IS a
denial-constraint (FD) violation. Standard transductive data-cleaning protocol.

CPAD-cat:
  - discover approximate FDs A->B (strength = fraction of rows matching their
    (A)-group mode of B); keep those with strength >= tau  [= "tight" constraints],
  - cell (t,B) violation under A->B = 1 - freq(t.B | t.A)  [minority in its group],
  - cell score = max over discovered FDs (*->col) of that violation.
Both column directions are discovered, so a typo in either side of an FD is caught.

Baselines:
  - marginal rarity     : 1 - freq(value in its column)            (no conditional structure)
  - discrete DC + count : # of strong FDs the cell violates (binary)  [the hard discrete baseline]
  - IsolationForest      : on one-hot encoding (row-level only)
"""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

from _common import DATA as D
norm = lambda df: df.fillna("").map(lambda s: str(s).strip().lower())
dirty = norm(pd.read_csv(D + "hospital_dirty.csv", dtype=str)).reset_index(drop=True)
err = np.load(D + "hospital_errmask.npy")            # (1000,19) boolean ground truth
cols = list(dirty.columns); n = len(dirty)

def fd_scores(df, tau):
    """Return (cell_score, strong_count) matrices, shape (n, ncols)."""
    cell = np.zeros((n, len(cols)))
    cnt  = np.zeros((n, len(cols)))
    for bi, B in enumerate(cols):
        for A in cols:
            if A == B: continue
            g = df.groupby(A)[B]
            mode = g.transform(lambda s: s.value_counts().idxmax())
            strength = (df[B].values == mode.values).mean()      # FD A->B strength
            if strength < tau: continue
            size = g.transform("size").values
            own  = df.groupby([A, B])[B].transform("size").values
            viol = 1.0 - own / size                              # 1 - freq(t.B | t.A)
            cell[:, bi] = np.maximum(cell[:, bi], viol)
            cnt[:, bi] += (viol > 0.5).astype(float)             # discrete violation
    return cell, cnt

def marginal(df):
    m = np.zeros((n, len(cols)))
    for ci, c in enumerate(cols):
        f = df[c].map(df[c].value_counts() / n).values
        m[:, ci] = 1.0 - f
    return m

def cell_auc(score): return roc_auc_score(err.ravel(), score.ravel()), average_precision_score(err.ravel(), score.ravel())
def row_auc(score):
    rs = score.max(1); ry = err.any(1)
    return roc_auc_score(ry, rs)

# IsolationForest on one-hot (row-level)
Xoh = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dirty).toarray()
if_row = np.mean([-IsolationForest(random_state=s, n_estimators=200).fit(Xoh).score_samples(Xoh)
                  for s in range(5)], 0)

print(f"Hospital: {n} rows, {len(cols)} cols, {int(err.sum())} error cells ({100*err.mean():.2f}%), "
      f"{int(err.any(1).sum())} rows w/ error ({100*err.any(1).mean():.1f}%)")
print("=" * 76)
print("CELL-LEVEL error detection (transductive):")
print(f"{'method':<28}{'AUROC':>9}{'AUPRC':>9}")
print("-" * 46)
mscore = marginal(dirty)
print(f"{'marginal rarity':<28}{cell_auc(mscore)[0]:>9.3f}{cell_auc(mscore)[1]:>9.3f}")
for tau in (0.8, 0.9, 0.95):
    cell, cnt = fd_scores(dirty, tau)
    cs = cell_auc(cell); ds = cell_auc(cnt)
    print(f"{'CPAD-cat (tau=%.2f)'%tau:<28}{cs[0]:>9.3f}{cs[1]:>9.3f}")
    print(f"{'  discrete DC count':<28}{ds[0]:>9.3f}{ds[1]:>9.3f}")
print("=" * 76)
print("ROW-LEVEL detection (row has >=1 error):")
print(f"{'method':<28}{'AUROC':>9}")
print("-" * 38)
cell, _ = fd_scores(dirty, 0.9)
print(f"{'CPAD-cat (tau=0.90)':<28}{row_auc(cell):>9.3f}")
print(f"{'marginal rarity':<28}{row_auc(mscore):>9.3f}")
print(f"{'IsolationForest (one-hot)':<28}{roc_auc_score(err.any(1), if_row):>9.3f}")
