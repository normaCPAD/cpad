"""NATIVE per-dataset rule extraction with CPAD's gated model, for the data-quality
benchmarks. For each dataset:

  1. auto-drop identifier-like columns (cardinality > CAP), as the paper's routing does;
  2. train CPAD's gated differentiable model (native);
  3. read the learned gate row G[B,:] -> candidate sources for each target B;
  4. turn it into a MINIMAL rule by greedy forward selection on FD confidence
     (add gate-ranked sources while confidence rises) -> LHS -> B with confidence;
  5. keep rules with confidence >= TAU and lift over the base rate;
  6. where a clean version exists, validate by detecting the natural errors
     (dirty != clean) with the native surprise score (AUROC).

Prints one block per dataset: the discovered rules (simple and composite) + AUROC.
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
CAP, TAU, LIFT, EP, LAM, DMODEL = 300, 0.90, 0.10, 250, 0.3, 24

from _common import DATA

DSETS = [
    ("Hospital", DATA + "hospital_dirty.csv", DATA + "hospital_errmask.npy"),
    ("Adult",    DATA + "adult.csv",          None),
    ("Tax",      DATA + "tax/dirty.csv",      DATA + "tax/clean.csv"),
    ("Beers",    DATA + "beers/dirty.csv",    DATA + "beers/clean.csv"),
    ("Flights",  DATA + "flights/dirty.csv",  DATA + "flights/clean.csv"),
]
norm = lambda df: df.fillna("").apply(lambda s: s.str.strip().str.lower())

def fd_conf(d, L, B, n):
    g = d.groupby(list(L) + [B]).size()
    return g.groupby(level=list(range(len(L)))).max().sum() / n

def avg_group(d, L):
    return len(d) / max(1, d.groupby(list(L)).ngroups)

def minimal_rule(d, order, B, n):
    """Smallest LHS (gate-ranked) reaching confidence TAU. A rule stays SIMPLE unless no
    single source suffices; composite growth requires a real jump AND non-fragmented
    groups (mean size >= 3), so confidence cannot be inflated by singleton conditioning."""
    cand = [A for A in order if A != B]
    singles = sorted(((A, fd_conf(d, [A], B, n)) for A in cand), key=lambda z: -z[1])
    if not singles:
        return (), 0.0, 0, 0.0
    bestA, bestc = singles[0]
    if bestc >= TAU:                                   # a single attribute already explains B
        return (bestA,), bestc, 1, bestc
    L, conf = [bestA], bestc                            # otherwise build a minimal composite LHS
    for A, _ in singles[1:]:
        if len(L) >= 3:
            break
        c = fd_conf(d, L + [A], B, n)
        if c > conf + 0.05 and avg_group(d, L + [A]) >= 3:
            L.append(A); conf = c
        if conf >= TAU:
            break
    return tuple(L), conf, len(L), bestc              # bestc = best single-source confidence

def run(name, dpath, cpath):
    dfull = norm(pd.read_csv(dpath, dtype=str))
    allcols = list(dfull.columns)
    keep = [c for c in allcols if 1 < dfull[c].nunique() <= CAP]
    keep_idx = [allcols.index(c) for c in keep]
    dirty = dfull[keep].reset_index(drop=True); cols = keep; nc = len(cols)
    # ground-truth error mask aligned to kept columns (npy mask, or aligned wide csv)
    err_full = None
    if cpath and os.path.exists(cpath):
        if cpath.endswith(".npy"):
            m = np.load(cpath)
            err_full = m[:, keep_idx] if m.shape[1] == len(allcols) else None
        else:
            cl = norm(pd.read_csv(cpath, dtype=str))
            if set(keep) <= set(cl.columns) and len(cl) == len(dirty):
                err_full = (dirty.values != cl[keep].reset_index(drop=True).values)
    if name == "Tax":
        si = np.random.default_rng(0).choice(len(dirty), 15000, replace=False)
        dirty = dirty.iloc[si].reset_index(drop=True)
        if err_full is not None: err_full = err_full[si]
    n = len(dirty)
    codes, cards = [], []
    for c in cols:
        code, uniq = pd.factorize(dirty[c]); codes.append(code); cards.append(len(uniq))
    X = torch.tensor(np.stack(codes, 1), dtype=torch.long)
    torch.manual_seed(0); np.random.seed(0)
    embs = nn.ModuleList([nn.Embedding(c, DMODEL) for c in cards])
    heads = nn.ModuleList([nn.Linear(DMODEL, c) for c in cards])
    gl = nn.Parameter(torch.zeros(nc, nc) - 2.0); eye = torch.eye(nc); ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(embs.parameters()) + list(heads.parameters()) + [gl], lr=0.01)
    gates = lambda: F.softplus(gl) * (1 - eye)
    def corrupt(xb, p=0.15):
        xc = xb.clone()
        for j in range(nc):
            m = torch.rand(len(xb)) < p; xc[m, j] = xb[torch.randperm(len(xb)), j][m]
        return xc
    for ep in range(EP):
        xin = corrupt(X); E = torch.stack([embs[j](xin[:, j]) for j in range(nc)], 1)
        G = gates(); R = torch.einsum('ba,nad->nbd', G, E)
        loss = sum(ce(heads[B](R[:, B, :]), X[:, B]) for B in range(nc)) + LAM * G.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()
    G = gates().detach().numpy()

    # native surprise score (for AUROC), full table
    with torch.no_grad():
        E = torch.stack([embs[j](X[:, j]) for j in range(nc)], 1)
        Gt = gates(); R = torch.einsum('ba,nad->nbd', Gt, E)
        viol = np.zeros((n, nc))
        for B in range(nc):
            p = torch.softmax(heads[B](R[:, B, :]), 1)
            viol[:, B] = 1.0 - p[torch.arange(n), X[:, B]].numpy()

    err = err_full if err_full is not None and len(err_full) == n else None

    rules = []
    for B in range(nc):
        order = [cols[a] for a in np.argsort(-G[B])]
        base = (dirty[cols[B]].value_counts() / n).max()
        L, conf, k, bestsingle = minimal_rule(dirty, order, cols[B], n)
        if L and conf >= TAU and conf - base >= LIFT:
            auc = ""
            if err is not None and err[:, B].sum() >= 5:
                auc = f"{roc_auc_score(err[:, B].astype(int), viol[:, B]):.3f}"
            rules.append((L, cols[B], conf, k, bestsingle, auc))
    return cols, rules

print(f"{'='*78}\nRègles apprises nativement par CPAD (modèle à portes), par jeu de données\n{'='*78}")
for name, dp, cp in DSETS:
    cols, rules = run(name, dp, cp)
    print(f"\n### {name}  ({len(cols)} colonnes exploitables) — {len(rules)} règles")
    print(f"{'LHS -> cible':<46}{'conf':>7}{'arité':>7}{'simpleFD':>9}{'AUROC':>7}")
    for L, B, conf, k, simple, auc in sorted(rules, key=lambda r: (-r[3], -r[2])):
        lhs = ", ".join(L)
        tag = "  (composée!)" if k > 1 else ""
        print(f"  ({lhs}) -> {B:<28}{conf:>7.3f}{k:>7}{simple:>9.3f}{auc:>7}{tag}")
