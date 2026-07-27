"""Generic AND deep tabular anomaly detectors (pyod) on Hospital, tuple-level.
They score whole rows (density/reconstruction), not cells, and target statistical
outliers -- not relational inconsistencies. Even recent deep detectors (DeepSVDD,
autoencoder, VAE) trail the constraint-based CPAD by a wide margin."""
import numpy as np, pandas as pd, warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
from _common import DATA as D
from cpad.core.table import Table
from cpad.models import RoutedCPAD
norm = lambda df: df.fillna("").apply(lambda s: s.astype(str).str.strip().str.lower())
d = norm(pd.read_csv(f"{D}hospital_dirty.csv", dtype=str))
err = np.load(f"{D}hospital_errmask.npy"); y = err.any(1).astype(int)
X = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(d).toarray()
print(f"Hospital one-hot: {X.shape}, {100*y.mean():.1f}% lignes avec erreur")

results = {}
def run(name, make):
    try:
        clf = make(); clf.fit(X); s = clf.decision_scores_
        results[name] = (roc_auc_score(y, s), average_precision_score(y, s))
    except Exception as e:
        results[name] = None; print(f"  [{name}] échec: {type(e).__name__}: {str(e)[:80]}")

from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.cblof import CBLOF
run("IForest", lambda: IForest(random_state=0, n_estimators=200))
run("ECOD", lambda: ECOD())
run("CBLOF", lambda: CBLOF(random_state=0, n_clusters=8))
try:
    from pyod.models.deep_svdd import DeepSVDD
    run("DeepSVDD", lambda: DeepSVDD(n_features=X.shape[1], random_state=0, epochs=30))
except Exception as e: print("DeepSVDD import:", e)
try:
    from pyod.models.auto_encoder import AutoEncoder
    run("AutoEncoder", lambda: AutoEncoder(epoch_num=30, hidden_neuron_list=[64,32], verbose=0))
except Exception as e:
    try:
        from pyod.models.auto_encoder import AutoEncoder
        run("AutoEncoder", lambda: AutoEncoder(epochs=30, verbose=0))
    except Exception as e2: print("AE import:", e2)
try:
    from pyod.models.vae import VAE
    run("VAE", lambda: VAE(epoch_num=30, verbose=0))
except Exception as e:
    try:
        from pyod.models.vae import VAE
        run("VAE", lambda: VAE(epochs=30, verbose=0))
    except Exception as e2: print("VAE import:", e2)

# constraint-based reference: the complete routed CPAD on the same table
cp = RoutedCPAD().fit(Table(d)).score(Table(d)).max(axis=1)
results["CPAD (complete)"] = (roc_auc_score(y, cp), average_precision_score(y, cp))

print("\n=== tuple-level detection on Hospital (AUROC / AUPRC) ===")
for k, v in results.items():
    print(f"  {k:<16} {v[0]:.3f}  {v[1]:.3f}" if v is not None else f"  {k:<16} échec")
