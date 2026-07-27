"""
CPAD with epochs + STRUCTURAL SPARSITY REGULARIZATION (categorical, Hospital).

Fixes the dense-mixing problem of the plain differentiable model: instead of summing
ALL other columns, each target B uses a learned gate vector g[B,:] >= 0 over source
columns, with an L1 penalty pushing it sparse -> each column depends on FEW sources
(the FD's left-hand side). This is the "learned gates + L1" acquisition of the paper.

Violation score of cell (t,B) = 1 - P(observed | gated sources). Fully unsupervised.
Sweeps the L1 weight lambda.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, warnings
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore"); torch.manual_seed(0)

from _common import DATA as D
norm = lambda df: df.fillna("").map(lambda s: str(s).strip().lower())
dirty = norm(pd.read_csv(D + "hospital_dirty.csv", dtype=str)).reset_index(drop=True)
err = np.load(D + "hospital_errmask.npy")
cols = list(dirty.columns); n = len(dirty); nc = len(cols)
codes, cards = [], []
for c in cols:
    code, uniq = pd.factorize(dirty[c]); codes.append(code); cards.append(len(uniq))
X = torch.tensor(np.stack(codes, 1), dtype=torch.long)
d = 24; EPOCHS = 400
eye = torch.eye(nc)
ce = nn.CrossEntropyLoss()

def cell(s): return roc_auc_score(err.ravel(), s.ravel()), average_precision_score(err.ravel(), s.ravel())
def row(s):  return roc_auc_score(err.any(1), s.max(1))

def corrupt(xb, p=0.15):
    xc = xb.clone()
    for j in range(nc):
        m = torch.rand(n) < p; xc[m, j] = xb[torch.randperm(n), j][m]
    return xc

def train_eval(lam):
    torch.manual_seed(0)
    embs  = nn.ModuleList([nn.Embedding(c, d) for c in cards])
    heads = nn.ModuleList([nn.Linear(d, c) for c in cards])
    gate_logits = nn.Parameter(torch.zeros(nc, nc) - 2.0)          # [target B, source A]
    params = list(embs.parameters()) + list(heads.parameters()) + [gate_logits]
    opt = torch.optim.Adam(params, lr=0.01)

    def gates():
        g = F.softplus(gate_logits) * (1 - eye)                   # nonneg, no self
        return g

    for ep in range(EPOCHS):
        xin = corrupt(X)
        E = torch.stack([embs[j](xin[:, j]) for j in range(nc)], 1)  # (n, nc, d)
        G = gates()
        R = torch.einsum('ba,nad->nbd', G, E)                     # (n, nc_target, d)
        loss = sum(ce(heads[B](R[:, B, :]), X[:, B]) for B in range(nc))
        loss = loss + lam * G.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        E = torch.stack([embs[j](X[:, j]) for j in range(nc)], 1)
        G = gates(); R = torch.einsum('ba,nad->nbd', G, E)
        viol = np.zeros((n, nc))
        for B in range(nc):
            p = torch.softmax(heads[B](R[:, B, :]), 1)
            viol[:, B] = 1.0 - p[torch.arange(n), X[:, B]].numpy()
        sparsity = (G > 0.05).float().sum().item() / nc            # avg sources per target
    violn = (viol - viol.mean(0)) / (viol.std(0) + 1e-9)
    return viol, violn, sparsity

print(f"Hospital diff. CPAD + L1 gate regularization ({EPOCHS} ep.)")
print(f"{'lambda':>8}{'srcs/col':>10}{'cellAUROC':>11}{'cellAUPRC':>11}{'rowAUROC':>10}  (z-norm rowAUROC)")
print("-" * 72)
for lam in [0.1, 0.2, 0.3, 0.5, 0.8]:
    viol, violn, sp = train_eval(lam)
    a, p = cell(viol)
    print(f"{lam:>8.3f}{sp:>10.1f}{a:>11.3f}{p:>11.3f}{row(viol):>10.3f}{row(violn):>14.3f}")
print("-" * 72)
print("rappel: CPAD-cat discret = 0.915 / 0.809 / 0.941   (DC discovery+count 0.944/0.751/0.937)")
