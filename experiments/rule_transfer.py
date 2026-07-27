"""Can CPAD's auto-discovered rules supply the expertise NADEEF needs (its expert
DCs), and conversely does expert knowledge help CPAD? On Hospital we score with
three rule sets -- expert DCs, CPAD-discovered FDs, and their union -- under the
same conditional-violation scoring, and measure rule overlap (rediscovery + novel).
"""
import re, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
from _common import DATA as D
norm = lambda df: df.fillna("").map(lambda s: str(s).strip().lower())
d = norm(pd.read_csv(D + "hospital_dirty.csv", dtype=str)).reset_index(drop=True)
err = np.load(D + "hospital_errmask.npy")
cols = list(d.columns); n, ci = len(d), {c: i for i, c in enumerate(d.columns)}

# --- expert DCs (HoloClean): EQ(LHS)->IQ(RHS) ---
expert = []
for line in open(D + "hospital_constraints.txt"):
    eqs = re.findall(r"EQ\(t1\.(\w+),", line); iqs = re.findall(r"IQ\(t1\.(\w+),", line)
    for B in iqs:
        if eqs: expert.append((tuple(eqs), B))           # (LHS attrs, RHS)
expert = list(set(expert))

# --- CPAD-discovered FDs: single source A->B, strength>=tau, info-gain>=0.15 ---
def discover(tau=0.90):
    rules = []
    for B in cols:
        if d[B].nunique() < 2: continue
        base = (d[B].value_counts() / n).max()
        for A in cols:
            if A == B or n / d[A].nunique() < 5: continue
            mode = d.groupby(A)[B].transform(lambda s: s.value_counts().idxmax())
            strg = (d[B].values == mode.values).mean()
            if strg >= tau and strg - base >= 0.15: rules.append(((A,), B))
    return list(set(rules))
cpad = discover()

def cond_viol(lhs, B):
    size = d.groupby(list(lhs))[B].transform("size").values
    own = d.groupby(list(lhs) + [B])[B].transform("size").values
    return 1.0 - own / size

def score(ruleset):
    cell = np.zeros((n, len(cols)))
    for lhs, B in ruleset:
        if all(a in cols for a in lhs):
            cell[:, ci[B]] = np.maximum(cell[:, ci[B]], cond_viol(lhs, B))
    return cell

from scipy.stats import rankdata
rnk = lambda x: rankdata(x) / len(x)
def marginal_surprise():
    """type-aware per-column outlier (Hospital is categorical): max(rareté,longueur,format)."""
    S = np.zeros((n, len(cols)))
    for j, c in enumerate(cols):
        rar = 1.0 - d[c].map(d[c].value_counts() / n).values
        ln = d[c].str.len().values.astype(float); ln = np.abs(ln - np.median(ln)) / (np.median(np.abs(ln - np.median(ln))) + 1e-9)
        dg = (d[c].str.count(r"\d").values / (d[c].str.len().values + 1)); dg = np.abs(dg - np.median(dg)) / (np.median(np.abs(dg - np.median(dg))) + 1e-9)
        S[:, j] = np.max([rnk(rar), rnk(ln), rnk(dg)], axis=0)
    return S
MARG = marginal_surprise()

def complete(rs):
    """Routed: columns governed by the rule set -> rule violation; others -> marginal surprise."""
    rule_cell = score(rs); governed = {ci[B] for _, B in rs if B in cols}
    S = np.zeros((n, len(cols)))
    for j in range(len(cols)):
        S[:, j] = rnk(rule_cell[:, j]) if j in governed else MARG[:, j]
    return S

def report(name, rs):
    sr = score(rs); sc = complete(rs)
    def m(s): return (roc_auc_score(err.ravel(), s.ravel()), roc_auc_score(err.any(1), s.max(1)))
    ar, rr = m(sr); ac, rc = m(sc)
    print(f"{name:<24}{len(rs):>5}{ar:>9.3f}{rr:>9.3f}{'':>4}{ac:>9.3f}{rc:>9.3f}")

union = list(set(expert) | set(cpad))
# overlap (compare single-LHS rules)
exp_simple = set((lhs[0], B) for lhs, B in expert if len(lhs) == 1)
cpad_simple = set((lhs[0], B) for lhs, B in cpad)
redisc = exp_simple & cpad_simple; novel = cpad_simple - exp_simple; missed = exp_simple - cpad_simple

print(f"{'jeu de règles':<24}{'#règ':>5}{'  --- règles seules ---':>18}{'  --- complet (routé) ---':>22}")
print(f"{'':<24}{'':>5}{'cellAUROC tupleAUROC':>18}{'cellAUROC tupleAUROC':>26}")
print("-" * 74)
report("NADEEF (DC expertes)", expert)
report("CPAD (règles apprises)", cpad)
report("Union (expert + CPAD)", union)
print("-" * 74)
print(f"Expert FD simples : {len(exp_simple)} | CPAD redécouvre : {len(redisc)} "
      f"({100*len(redisc)/max(1,len(exp_simple)):.0f}%) | manquées : {len(missed)} | "
      f"nouvelles (CPAD seul) : {len(novel)}")
print("Exemples de règles NOUVELLES trouvées par CPAD (absentes des DC expertes) :")
for a, b in list(novel)[:5]: print(f"   {a} -> {b}")
