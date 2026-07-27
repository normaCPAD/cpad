"""Build a REAL Wikidata table and run CPAD detection + completion on it. Produces the numbers
the paper needs: size, dimensionality, sparsity, discovered FDs, detection AUROC (CPAD vs the
marginal density baseline vs Isolation Forest), and how many empty cells CPAD can fill.
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import wikidata_shacl as wd

CLASS = "Q486972"                                         # human settlement (strong admin/geo FDs, many props)
PROPS = ["P17", "P131", "P30", "P421", "P1376", "P206", "P47", "P361", "P138",
         "P706", "P2936", "P37", "P190", "P571", "P856", "P1082"]   # 16 properties -> high-dim, sparse
PROPS = list(dict.fromkeys(PROPS))                        # de-dup, keep order
N = 20000
rng = np.random.default_rng(0)

print(f"[wikidata] building film table ({N} items, {len(PROPS)} properties)...")
constraints = wd.fetch_property_constraints(PROPS)
table = wd.fetch_entity_table(CLASS, PROPS, N)
props = [c for c in table.columns if c.startswith("P") and not c.endswith("_cnt")]
flat = table[props].astype(str)
n = len(flat)

mask, vlog = wd.label_violations(table, constraints)
fds = wd.discover_fds(flat, props, tau=0.85, min_support=20)
sugg = wd.suggest_completions(flat, props, fds=fds, constraints=constraints)
fill = (flat != "").mean()
print(f"[wikidata] {n} rows x {len(props)} props | mean fill {fill.mean():.1%} (sparse)")
print(f"[wikidata] declared-constraint violations: {int(mask.sum())} | discovered FDs: {len(fds)} "
      f"| empty-cell completions suggested: {len(sugg)}")
for A, B, c in sorted(fds, key=lambda z: -z[2])[:6]:
    print(f"    FD  {A} -> {B}   conf={c:.2f}")

# ---- detection: inject DC violations into the governed columns (paper methodology) ----
ci = {c: j for j, c in enumerate(props)}
gov = sorted({B for _, B, _ in fds})
if not gov:
    print("[wikidata] no single-LHS FD with enough support -> detection not applicable (reported as such)")
    sys.exit(0)

dirty = flat.copy(); y = np.zeros(n, bool)
for _, B, _ in fds:
    nonempty = np.where(flat[B].values != "")[0]
    sel = nonempty[rng.random(len(nonempty)) < 0.05]
    pool = flat[B][flat[B] != ""].value_counts(normalize=True)
    repl = rng.choice(pool.index.values, size=len(sel), p=pool.values)
    keep = repl != flat[B].values[sel]; sel, repl = sel[keep], repl[keep]
    dirty.iloc[sel, ci[B]] = repl; y[sel] = True


def fd_row_score(df):
    cell = np.zeros((n, len(props)))
    for A, B, _ in fds:
        sub = df.groupby(A)[B]
        size = sub.transform("size").values
        own = df.groupby([A, B])[B].transform("size").values
        cell[:, ci[B]] = np.maximum(cell[:, ci[B]], 1.0 - np.where(size > 0, own / size, 1.0))
    return cell.max(1)


def marginal_row(df):
    m = np.zeros((n, len(props)))
    for c in props:
        m[:, ci[c]] = 1.0 - df[c].map(df[c].value_counts() / n).values
    return m.max(1)


cpad = roc_auc_score(y, fd_row_score(dirty))
marg = roc_auc_score(y, marginal_row(dirty))
X = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dirty).toarray()
ifr = roc_auc_score(y, -IsolationForest(random_state=0, n_estimators=150).fit(X).score_samples(X))
print(f"\n[wikidata] detection of injected DC violations ({int(y.sum())} errors), tuple AUROC:")
print(f"    CPAD={cpad:.3f}   Marginal={marg:.3f}   IForest={ifr:.3f}")
