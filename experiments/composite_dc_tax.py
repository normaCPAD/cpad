"""NATIVE composite-DC discovery on Tax with CPAD's gated differentiable model.

Goal: show CPAD natively discovers a MULTI-LHS (composite) denial constraint that the
single-LHS discrete miner cannot, and exploits it to detect real errors -- all from
CPAD's own machinery (learned gates G[B,A] + surprise score), no external statistics.

  expected:  (state, marital_status) -> single_exemp     [DC: equality on TWO sources]
             (state, has_child)      -> child_exemp

Native extraction  = read the gate row G[target,:] (which sources it lights up).
Native exploitation = surprise score 1 - P(t.B | gated sources), tested against the
natural error mask (dirty != clean). Identifier-like columns (cardinality > CAP) are
auto-dropped, as the paper's incoming-FD routing already does.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2)                                  # be polite to the running benchmark

from _common import DATA as _DATA
D = _DATA + "tax/"
norm = lambda df: df.fillna("").apply(lambda s: s.str.strip().str.lower())
clean = norm(pd.read_csv(D + "clean.csv", dtype=str))
dirty = norm(pd.read_csv(D + "dirty.csv", dtype=str))

CAP = 300                                                 # auto-drop identifier-like columns
keep = [c for c in dirty.columns if dirty[c].nunique() <= CAP]
dirty, clean = dirty[keep], clean[keep]
errmask = (dirty.values != clean.values)                 # natural ground truth, row-aligned
cols = keep; nc = len(cols)
print("colonnes retenues (card<=%d):" % CAP, cols)

codes, cards = [], []
for c in cols:
    code, uniq = pd.factorize(dirty[c]); codes.append(code); cards.append(len(uniq))
Xfull = torch.tensor(np.stack(codes, 1), dtype=torch.long)
N = len(Xfull)

idx = np.random.default_rng(0).choice(N, 15000, replace=False)
Xtr = Xfull[idx]; ntr = len(Xtr)
eye = torch.eye(nc); ce = nn.CrossEntropyLoss()
d, EPOCHS, LAM = 24, 300, 0.3

embs = nn.ModuleList([nn.Embedding(c, d) for c in cards])
heads = nn.ModuleList([nn.Linear(d, c) for c in cards])
gate_logits = nn.Parameter(torch.zeros(nc, nc) - 2.0)     # [target B, source A]
opt = torch.optim.Adam(list(embs.parameters()) + list(heads.parameters()) + [gate_logits], lr=0.01)
def gates(): return F.softplus(gate_logits) * (1 - eye)   # nonneg, no self-loop

def corrupt(xb, p=0.15):
    xc = xb.clone()
    for j in range(nc):
        m = torch.rand(len(xb)) < p; xc[m, j] = xb[torch.randperm(len(xb)), j][m]
    return xc

print("entraînement du modèle à portes (%d ép., L1=%.2f)..." % (EPOCHS, LAM))
for ep in range(EPOCHS):
    xin = corrupt(Xtr)
    E = torch.stack([embs[j](xin[:, j]) for j in range(nc)], 1)
    G = gates(); R = torch.einsum('ba,nad->nbd', G, E)
    loss = sum(ce(heads[B](R[:, B, :]), Xtr[:, B]) for B in range(nc)) + LAM * G.abs().sum()
    opt.zero_grad(); loss.backward(); opt.step()

# ---- NATIVE EXTRACTION: read the learned gate rows ----
G = gates().detach()
print("\n=== Règles extraites des portes  G[cible, :]  (sources allumées) ===")
for target in ["single_exemp", "married_exemp", "child_exemp", "rate", "state"]:
    if target not in cols: continue
    B = cols.index(target); row = G[B].numpy()
    top = sorted([(cols[a], row[a]) for a in range(nc)], key=lambda z: -z[1])[:3]
    srcs = ", ".join(f"{a}:{w:.2f}" for a, w in top if w > 0.05) or "(aucune)"
    print(f"  {target:14} <- {srcs}")

# ---- NATIVE EXPLOITATION: surprise score, scored on ALL rows in mini-batches ----
viol = np.zeros((N, nc))
with torch.no_grad():
    G = gates()
    for s in range(0, N, 20000):
        e = min(s + 20000, N); xb = Xfull[s:e]
        E = torch.stack([embs[j](xb[:, j]) for j in range(nc)], 1)
        R = torch.einsum('ba,nad->nbd', G, E)
        for B in range(nc):
            p = torch.softmax(heads[B](R[:, B, :]), 1)
            viol[s:e, B] = 1.0 - p[torch.arange(e - s), xb[:, B]].numpy()

print("\n=== Détection des VRAIES erreurs via le score de surprise (AUROC) ===")
for target in ["single_exemp", "child_exemp", "state", "marital_status", "has_child"]:
    if target not in cols: continue
    B = cols.index(target); y = errmask[:, B].astype(int)
    if y.sum() >= 5:
        print(f"  {target:14} erreurs={y.sum():>4}  AUROC={roc_auc_score(y, viol[:, B]):.3f}")
    else:
        print(f"  {target:14} erreurs={y.sum():>4}  (trop peu pour AUROC)")

# ---- contrast: single-LHS discrete miner MISSES the composite FD ----
print("\n=== Contraste : mineur discret LHS-unique sur single_exemp ===")
n = len(dirty)
def fd1(A, B):
    g = dirty.groupby([A, B]).size(); return g.groupby(level=0).max().sum() / n
best = sorted([(A, fd1(A, "single_exemp")) for A in cols if A != "single_exemp"], key=lambda z: -z[1])[:3]
print("  meilleures FD simples ->single_exemp:", ", ".join(f"{A}:{s:.2f}" for A, s in best))
comp = dirty.groupby(["state", "marital_status", "single_exemp"]).size().groupby(level=[0, 1]).max().sum() / n
print(f"  FD composée (state,marital_status)->single_exemp : {comp:.3f}  (hors de portée du LHS-unique)")
