"""Error-type dissection against a broad SOTA panel. On the clean Hospital table we inject
typo errors (marginally rare strings) and value-swap errors (valid, frequent values that
break an FD), and compare a dozen anomaly detectors plus CPAD on each, at the tuple level.
Density/deep detectors fare on typos but collapse on value-swaps; CPAD holds on both.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

from _common import DATA as D
norm = lambda df: df.fillna("").map(lambda s: str(s).strip().lower())
clean = norm(pd.read_csv(D + "hospital_clean_wide.csv", dtype=str)).reset_index(drop=True)
cols = list(clean.columns); n = len(clean); nc = len(cols); ci = {c: i for i, c in enumerate(cols)}
TARGETS = [c for c in cols if 2 <= clean[c].nunique() <= 60]


def fd_row_score(df, tau=0.85):
    cell = np.zeros((n, nc))
    for bi, B in enumerate(cols):
        for A in cols:
            if A == B:
                continue
            g = df.groupby(A)[B]
            mode = g.transform(lambda s: s.value_counts().idxmax())
            if (df[B].values == mode.values).mean() < tau:
                continue
            size = g.transform("size").values
            own = df.groupby([A, B])[B].transform("size").values
            cell[:, bi] = np.maximum(cell[:, bi], 1.0 - own / size)
    return cell.max(1)


def inject_typo(df, rate=0.05):
    dc = df.copy(); M = np.zeros((n, nc), bool)
    for c in TARGETS:
        idx = np.where(rng.random(n) < rate)[0]
        for i in idx:
            dc.iat[i, ci[c]] = str(df.iat[i, ci[c]]) + "x" + str(rng.integers(99))
            M[i, ci[c]] = True
    return dc.reset_index(drop=True), M.any(1)


def inject_valueswap(df, rate=0.05):
    dc = df.copy(); M = np.zeros((n, nc), bool)
    for c in TARGETS:
        vals = df[c].values; freq = df[c].value_counts(normalize=True)
        pool, probs = freq.index.values, freq.values
        idx = np.where(rng.random(n) < rate)[0]
        for i in idx:
            repl = rng.choice(pool, p=probs)
            if repl != vals[i]:
                dc.iat[i, ci[c]] = repl; M[i, ci[c]] = True
    return dc.reset_index(drop=True), M.any(1)


def detectors(X):
    from pyod.models.iforest import IForest
    from pyod.models.ecod import ECOD
    from pyod.models.cblof import CBLOF
    from pyod.models.copod import COPOD
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA
    from pyod.models.hbos import HBOS
    from pyod.models.deep_svdd import DeepSVDD
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE
    d = {"IF": IForest(random_state=0, n_estimators=200), "ECOD": ECOD(),
         "CBLOF": CBLOF(random_state=0, n_clusters=8), "COPOD": COPOD(),
         "KNN": KNN(), "LOF": LOF(), "OCSVM": OCSVM(), "PCA": PCA(random_state=0),
         "HBOS": HBOS(),
         "DSVDD": DeepSVDD(n_features=X.shape[1], random_state=0, epochs=30),
         "AE": AutoEncoder(epoch_num=30, verbose=0), "VAE": VAE(epoch_num=30, verbose=0)}
    out = {}
    for name, clf in d.items():
        try:
            clf.fit(X); out[name] = clf.decision_scores_
        except Exception as e:
            out[name] = None; print(f"  [{name}] failed: {type(e).__name__}")
    return out


def run(dirty, y, label):
    X = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dirty).toarray()
    res = {}
    for name, s in detectors(X).items():
        res[name] = roc_auc_score(y, s) if s is not None else float("nan")
    res["CPAD"] = roc_auc_score(y, fd_row_score(dirty))
    print(f"\n[{label}]  {int(y.sum())} corrupted tuples")
    print("  " + "  ".join(f"{k}:{v:.2f}" for k, v in res.items()))
    return res


if __name__ == "__main__":
    print(f"Clean Hospital: {n} tuples, {len(TARGETS)} FD-bearing columns")
    Dt, yt = inject_typo(clean)
    Dv, yv = inject_valueswap(clean)
    rt = run(Dt, yt, "TYPO")
    rv = run(Dv, yv, "VALUE-SWAP")
    print("\n=== LaTeX-ready (tuple AUROC) ===")
    order = ["IF", "LOF", "KNN", "OCSVM", "PCA", "HBOS", "COPOD", "ECOD", "CBLOF", "DSVDD", "AE", "VAE", "CPAD"]
    print("           " + " ".join(f"{k:>6}" for k in order))
    print("typo       " + " ".join(f"{rt.get(k, float('nan')):>6.2f}" for k in order))
    print("value-swap " + " ".join(f"{rv.get(k, float('nan')):>6.2f}" for k in order))
