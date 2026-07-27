"""
Final improved CPAD (full unsupervised), covering type-2a errors.
Per cell, score = max( rank(CPAD conditional violation), rank(type-aware outlier) ).
- CPAD conditional: catches constraint violations (incl. value-swaps) in governed cols.
- type-aware outlier: numeric -> robust-z; categorical -> max(rarity, length-z, digit-z),
  catches statistical/format outliers in UNgoverned cols.
The UNION (not hard routing) lets governed columns benefit from BOTH detectors.
Evaluated on Tax (mostly ungoverned errors) and Hospital (mostly governed errors).
"""
import numpy as np, pandas as pd, warnings
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
from _common import DATA as D
norm = lambda df: df.fillna("").apply(lambda s: s.astype(str).str.strip().str.lower())
rn = lambda x: rankdata(x) / len(x)
def rz(x):
    x = np.asarray(x, float); m = np.nanmedian(x); s = np.nanmedian(np.abs(x - m)) + 1e-9
    return np.nan_to_num(np.abs(x - m) / s)

def cpad_scores(df, tau=0.9):
    n, cols = len(df), list(df.columns); cell = np.zeros((n, len(cols)))
    for bi, B in enumerate(cols):
        for A in cols:
            if A == B: continue
            g = df.groupby(A)[B]; mode = g.transform(lambda s: s.value_counts().idxmax())
            if (df[B].values == mode.values).mean() < tau: continue
            size = g.transform("size").values; own = df.groupby([A, B])[B].transform("size").values
            cell[:, bi] = np.maximum(cell[:, bi], 1.0 - own / size)
    return cell

def outlier_scores(df):
    n, cols = len(df), list(df.columns); S = np.zeros((n, len(cols)))
    for j, B in enumerate(cols):
        num = pd.to_numeric(df[B], errors="coerce")
        if num.notna().mean() > 0.8:
            S[:, j] = rn(rz(num.values))
        else:
            rar = 1.0 - df[B].map(df[B].value_counts() / n).values
            ln = rz(df[B].str.len().values)
            dg = rz(df[B].str.count(r"\d").values / (df[B].str.len().values + 1))
            S[:, j] = np.max([rn(rar), rn(ln), rn(dg)], axis=0)
    return S

def incoming_strength(df):
    n, cols = len(df), list(df.columns); s = {}
    for B in cols:
        if df[B].nunique() < 2: s[B] = 0; continue
        base = (df[B].value_counts() / n).max(); best = 0.0
        for A in cols:
            if A == B or n / df[A].nunique() < 5: continue
            mode = df.groupby(A)[B].transform(lambda x: x.value_counts().idxmax())
            strg = (df[B].values == mode.values).mean()
            best = max(best, strg if strg - base >= 0.15 else 0)
        s[B] = best
    return s

def run(name, df, err):
    cols = list(df.columns); C = cpad_scores(df); O = outlier_scores(df)
    Cr = np.apply_along_axis(rn, 0, C); Or_ = np.apply_along_axis(rn, 0, O)
    STR = incoming_strength(df)
    R = np.where([STR[B] >= 0.90 for B in cols], Cr, Or_)        # ROUTING by FD strength
    def m(S): return (roc_auc_score(err.ravel(), S.ravel()),
                      average_precision_score(err.ravel(), S.ravel()),
                      roc_auc_score(err.any(1), S.max(1)))
    ng = sum(STR[B] >= 0.90 for B in cols)
    print(f"\n=== {name} ({df.shape[0]}x{df.shape[1]}, {100*err.mean():.1f}% err, "
          f"{ng}/{len(cols)} col. gouvernées) ===")
    print(f"{'détecteur':<30}{'cellAUROC':>10}{'AUPRC':>8}{'rowAUROC':>10}")
    for nm, S in [("CPAD seul", Cr), ("outlier 2a seul", Or_),
                  ("CPAD+outlier ROUTÉ (final)", R)]:
        a, p, r = m(S); print(f"{nm:<30}{a:>10.3f}{p:>8.3f}{r:>10.3f}")

# Tax
d = norm(pd.read_csv(f"{D}tax/dirty.csv", dtype=str, nrows=5000))
c = norm(pd.read_csv(f"{D}tax/clean.csv", dtype=str, nrows=5000))
cols = [x for x in d.columns if x in c.columns and x.lower() not in
        ("tuple_id", "index", "id", "phone", "f_name", "l_name")]
d, c = d[cols].reset_index(drop=True), c[cols].reset_index(drop=True)
run("Tax", d, (d.values != c.values))

# Hospital
h = norm(pd.read_csv(f"{D}hospital_dirty.csv", dtype=str))
run("Hospital", h, np.load(f"{D}hospital_errmask.npy"))
