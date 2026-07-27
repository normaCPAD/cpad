"""
BART value-swap experiment: errors that are VALID, MARGINALLY-COMMON values which
nonetheless violate a functional dependency. The decisive test that separates
constraint reasoning from marginal rarity.

On the CLEAN Hospital table we inject two error types with exact ground truth:
  - TYPO        : replace a cell by a unique perturbed string (rare marginally).
  - VALUE-SWAP  : replace a cell by another value drawn from the column's FREQUENCY
                  distribution (a common, valid value) -> breaks the FD, but is NOT
                  marginally rare.
We compare CPAD (conditional), marginal rarity, and Isolation Forest on each.
Expectation: marginal rarity strong on TYPOs, blind on VALUE-SWAPs; CPAD robust to both.
"""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

from _common import DATA as D
norm = lambda df: df.fillna("").map(lambda s: str(s).strip().lower())
clean = norm(pd.read_csv(D + "hospital_clean_wide.csv", dtype=str)).reset_index(drop=True)
cols = list(clean.columns); n = len(clean); nc = len(cols); ci = {c: i for i, c in enumerate(cols)}
# FD-bearing columns (avoid near-unique id columns where any swap is "valid")
TARGETS = [c for c in cols if 2 <= clean[c].nunique() <= 60]

def fd_cell_scores(df, tau=0.85):
    cell = np.zeros((n, nc))
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

def marginal(df):
    m = np.zeros((n, nc))
    for c in cols: m[:, ci[c]] = 1.0 - df[c].map(df[c].value_counts() / n).values
    return m

def inject_typo(df, rate=0.05):
    dc = df.copy(); M = np.zeros((n, nc), bool)
    for c in TARGETS:
        idx = np.where(rng.random(n) < rate)[0]
        for i in idx:
            dc.iat[i, ci[c]] = str(df.iat[i, ci[c]]) + "x" + str(rng.integers(99))  # unique string
            M[i, ci[c]] = True
    return dc.reset_index(drop=True), M

def inject_valueswap(df, rate=0.05):
    dc = df.copy(); M = np.zeros((n, nc), bool)
    for c in TARGETS:
        vals = df[c].values; freq = df[c].value_counts(normalize=True)
        pool, probs = freq.index.values, freq.values            # frequency-weighted (common values)
        idx = np.where(rng.random(n) < rate)[0]
        for i in idx:
            repl = rng.choice(pool, p=probs)
            if repl != vals[i]:
                dc.iat[i, ci[c]] = repl; M[i, ci[c]] = True
    return dc.reset_index(drop=True), M

def evaluate(dirty, M, label):
    Xoh = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dirty).toarray()
    ifr = np.mean([-IsolationForest(random_state=s, n_estimators=200).fit(Xoh).score_samples(Xoh)
                   for s in range(3)], 0)
    det = {"CPAD (conditionnel)": fd_cell_scores(dirty), "Rareté marginale": marginal(dirty)}
    print(f"\n[{label}]  {int(M.sum())} erreurs injectées ({100*M.mean():.2f}% des cellules)")
    print(f"{'méthode':<24}{'cellAUROC':>10}{'cellAUPRC':>10}{'rowAUROC':>10}")
    print("-" * 54)
    for name, s in det.items():
        a = roc_auc_score(M.ravel(), s.ravel()); p = average_precision_score(M.ravel(), s.ravel())
        r = roc_auc_score(M.any(1), s.max(1))
        print(f"{name:<24}{a:>10.3f}{p:>10.3f}{r:>10.3f}")
    print(f"{'Isolation Forest':<24}{'---':>10}{'---':>10}{roc_auc_score(M.any(1), ifr):>10.3f}")

print(f"Clean Hospital: {n} tuples, {len(TARGETS)} FD-bearing target columns")
print("=" * 54)
Dt, Mt = inject_typo(clean);       evaluate(Dt, Mt, "TYPO (chaînes rares)")
Dv, Mv = inject_valueswap(clean);  evaluate(Dv, Mv, "VALUE-SWAP (valeurs valides fréquentes)")
print("=" * 54)
print("Attendu: marginale forte sur TYPO, AVEUGLE sur VALUE-SWAP ; CPAD robuste aux deux.")
