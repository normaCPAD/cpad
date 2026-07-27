"""
State-of-the-art comparison on Hospital (cell- and tuple-level error detection).

Faithful, runnable baselines spanning the related-work families:
  - NADEEF / HoloClean detection : violations of the EXPERT-GIVEN denial constraints
      (the official HoloClean hospital_constraints.txt). Isolates given vs learned.
  - DC discovery + count         : violations of FDs DISCOVERED from data (discrete).
  - HoloDetect-style             : synthetic augmentation + learned classifier (self-sup).
  - Isolation Forest             : statistical outlier detector (one-hot), tuple-level.
  - Marginal rarity              : per-cell value rarity (no relational structure).
  - CPAD-cat (ours)              : learned FDs, self-supervised tau, conditional violation.
All unsupervised w.r.t. real labels (HoloDetect/CPAD select via self-supervision).
"""
import re, numpy as np, pandas as pd, warnings
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

from _common import DATA as D
norm = lambda df: df.fillna("").map(lambda s: str(s).strip().lower())
dirty = norm(pd.read_csv(D + "hospital_dirty.csv", dtype=str)).reset_index(drop=True)
err = np.load(D + "hospital_errmask.npy")
cols = list(dirty.columns); ci = {c: i for i, c in enumerate(cols)}; n = len(dirty)

# ----------------------------------------------------------------------
# parse expert denial constraints: t1&t2&EQ(t1.A,t2.A)&...&IQ(t1.B,t2.B)
# EQ attrs = LHS (context), IQ attrs = RHS (must not differ) => FD LHS->RHS
# ----------------------------------------------------------------------
def parse_dcs(path):
    dcs = []
    for line in open(path):
        eqs = re.findall(r"EQ\(t1\.(\w+),", line)
        iqs = re.findall(r"IQ\(t1\.(\w+),", line)
        if eqs and iqs:
            dcs.append((eqs, iqs))
    return dcs
DCS = parse_dcs(D + "hospital_constraints.txt")

def conditional_viol(df, lhs, B):
    """1 - freq(t.B | t[lhs]) per row: minority of B within its lhs-group."""
    size = df.groupby(lhs)[B].transform("size").values
    own  = df.groupby(lhs + [B])[B].transform("size").values
    return 1.0 - own / size

def nadeef_given(df):
    """Cell scores from violations of the EXPERT-GIVEN DCs."""
    cell = np.zeros((n, len(cols)))
    for eqs, iqs in DCS:
        for B in iqs:
            v = conditional_viol(df, eqs, B)
            cell[:, ci[B]] = np.maximum(cell[:, ci[B]], v)
    return cell

# ----------------------------------------------------------------------
# discovered FDs (CPAD machinery) — used by CPAD-cat and as a HoloDetect feature
# ----------------------------------------------------------------------
def fd_cell_scores(df, tau):
    cell = np.zeros((n, len(cols)))
    for B in cols:
        for A in cols:
            if A == B: continue
            g = df.groupby(A)[B]
            mode = g.transform(lambda s: s.value_counts().idxmax())
            if (df[B].values == mode.values).mean() < tau: continue
            cell[:, ci[B]] = np.maximum(cell[:, ci[B]], conditional_viol(df, [A], B))
    return cell

def marginal(df):
    m = np.zeros((n, len(cols)))
    for c in cols:
        m[:, ci[c]] = 1.0 - df[c].map(df[c].value_counts() / n).values
    return m

# ----------------------------------------------------------------------
# corrupters for self-supervision
# ----------------------------------------------------------------------
def inject(df, rate=0.03):
    dc = df.copy(); mask = np.zeros((n, len(cols)), bool)
    for c in cols:
        vals = df[c].values; idx = rng.random(n) < rate
        repl = rng.permutation(vals); idx &= (repl != vals)
        dc.iloc[idx, ci[c]] = repl[idx]; mask[idx, ci[c]] = True
    return dc.reset_index(drop=True), mask

# ----------------------------------------------------------------------
# HoloDetect-style: features + classifier trained on augmented data (self-sup)
# ----------------------------------------------------------------------
def features(df, fd):
    f = []
    marg = marginal(df)
    for c in cols:
        vals = df[c].astype(str)
        f.append(np.stack([
            marg[:, ci[c]],                                   # marginal rarity
            vals.str.len().values,                            # value length
            vals.str.count(r"\d").values / (vals.str.len().values + 1),  # digit fraction
            np.full(n, df[c].nunique()),                      # column cardinality
            fd[:, ci[c]],                                     # FD-violation signal
        ], 1))
    return np.concatenate(f, 0)                               # (n*ncols, 5)

def holodetect(df):
    """HoloDetect-style, FULLY UNSUPERVISED (no labels): learn a cell classifier
    from synthetic augmentation only (clean=original, positive=injected value-swaps).
    Fair unsupervised setting: no label budget, the whole table is given."""
    dinj, M = inject(df, rate=0.05)
    Xtr, ytr = features(dinj, fd_cell_scores(dinj, 0.9)), M.T.ravel()
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0).fit(Xtr, ytr)
    s = clf.predict_proba(features(df, fd_cell_scores(df, 0.9)))[:, 1]
    return s.reshape(len(cols), n).T

def holodetect_fewshot(df, n_labels=200):
    """Particular case: HoloDetect's actual few-shot paradigm (a small label budget
    cleans the augmentation). Not a fair unsupervised baseline -- shown for reference."""
    fd_real = fd_cell_scores(df, 0.9); Xreal = features(df, fd_real); y = err.T.ravel()
    k = n_labels // 2
    lab = np.r_[rng.choice(np.where(y == 1)[0], k, replace=False),
                rng.choice(np.where(y == 0)[0], k, replace=False)]
    dinj, M = inject(df, 0.05); Xaug = features(dinj, fd_cell_scores(dinj, 0.9))
    aug = np.where(M.T.ravel() == 1)[0]
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(np.vstack([Xreal[lab], Xaug[aug]]), np.r_[y[lab], np.ones(len(aug))])
    return clf.predict_proba(Xreal)[:, 1].reshape(len(cols), n).T

# ----------------------------------------------------------------------
# run all detectors
# ----------------------------------------------------------------------
def cellm(s): return roc_auc_score(err.ravel(), s.ravel()), average_precision_score(err.ravel(), s.ravel())
def rowm(s):  return roc_auc_score(err.any(1), s.max(1))

# CPAD tau selection: self-supervised (no labels) vs oracle (real labels)
TAUS = [0.80, 0.85, 0.90, 0.95]
Dinj, M = inject(dirty, 0.03)
tau_self  = max(TAUS, key=lambda t: roc_auc_score(M.ravel(), fd_cell_scores(Dinj, t).ravel()))
tau_oracle = max(TAUS, key=lambda t: roc_auc_score(err.ravel(), fd_cell_scores(dirty, t).ravel()))

Xoh = OneHotEncoder(handle_unknown="ignore", max_categories=50).fit_transform(dirty).toarray()
if_row = np.mean([-IsolationForest(random_state=s, n_estimators=200).fit(Xoh).score_samples(Xoh)
                  for s in range(5)], 0)

# CAS GENERAL : non supervisé, comparaison juste (toute la table, aucun label ni
# contrainte fournie -- aucune méthode n'a de "maîtrise").
unsup = {
    "Marginal rarity":              marginal(dirty),
    "DC discovery + count":         (fd_cell_scores(dirty, 0.95) > 0.5).astype(float),
    "HoloDetect-style (auto-sup)":  holodetect(dirty),
    "CPAD-cat (tau auto-sup)":      fd_cell_scores(dirty, tau_self),
}
# CAS PARTICULIER : information supplémentaire (contraintes expertes ou labels).
# CPAD reçoit la MEME info que ses concurrents, pour comparer à armes égales.
extra = {
    "NADEEF (DC expertes)":          nadeef_given(dirty),
    "CPAD (expertes + apprises)":    np.maximum(nadeef_given(dirty), fd_cell_scores(dirty, tau_self)),
    "HoloDetect (few-shot 200 lab.)":holodetect_fewshot(dirty),
    "CPAD (labels : tau oracle)":    fd_cell_scores(dirty, tau_oracle),
}

def block(title, d, with_if=False):
    print(f"\n{title}")
    print(f"{'method':<32}{'cellAUROC':>10}{'cellAUPRC':>10}{'rowAUROC':>10}")
    print("-" * 74)
    for name, s in d.items():
        a, p = cellm(s); print(f"{name:<32}{a:>10.3f}{p:>10.3f}{rowm(s):>10.3f}")
    if with_if:
        print(f"{'Isolation Forest (one-hot)':<32}{'---':>10}{'---':>10}"
              f"{roc_auc_score(err.any(1), if_row):>10.3f}")

print(f"Hospital: {n} rows, {len(cols)} cols, {int(err.sum())} errors ({100*err.mean():.2f}%), "
      f"{len(DCS)} expert DCs.  CPAD self-sup tau={tau_self}")
print("=" * 74)
block("[CAS GENERAL] Non supervisé -- comparaison juste (aucun label, aucune contrainte fournie)",
      unsup, with_if=True)
block("[CAS PARTICULIER] Information supplémentaire disponible (contraintes expertes / labels)",
      extra)
print("=" * 74)
