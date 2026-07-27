"""Fair, single-protocol detection across ALL datasets (real + Wikidata), so the multi-dataset
table is apples-to-apples. EVERY dataset goes through the IDENTICAL pipeline:

  1. discover single-LHS FDs with a group-size guard (reject near-key sources);
  2. inject value-swap DC-violations into the NON-EMPTY governed cells at the same rate (5%);
  3. score each tuple with the same three detectors:
       - CPAD     : max over governed columns of the conditional violation 1 - freq(B | A);
       - Marginal : max over columns of the value rarity 1 - freq(value)  (density baseline);
       - IForest  : Isolation Forest on a one-hot encoding;
  4. report tuple-level AUROC.

Same rate, same scorer, same FD definition, same baselines -> a fair comparison. Hard datasets
(weak/absent relational structure) are kept and reported as such: the point is that CPAD's
accuracy tracks the structure, not that it wins everywhere.
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(0)
ROW_CAP = 20000                                           # same cap for every dataset (tractable, fair)
TAU, MIN_GROUP, MIN_SUPPORT, RATE = 0.90, 4.0, 30, 0.05
DATA = sys.argv[1] if len(sys.argv) > 1 else "../data_dq/"


# ---- the identical protocol ----------------------------------------------------------------
def discover_fds(df, cols):
    fds = []
    for B in cols:
        best = None
        for A in cols:
            if A == B:
                continue
            sub = df[(df[A] != "") & (df[B] != "")]
            if len(sub) < MIN_SUPPORT or len(sub) / max(1, sub[A].nunique()) < MIN_GROUP:
                continue
            conf = sub.groupby(A)[B].transform(lambda s: s.value_counts(normalize=True).max()).mean()
            if conf >= TAU and (best is None or conf > best[1]):
                best = (A, float(conf))
        if best:
            fds.append((best[0], B))
    return fds


def inject(df, fds, ci):
    dirty = df.copy(); n = len(df); y = np.zeros(n, bool)
    for A, B in fds:
        nonempty = np.where(df[B].values != "")[0]
        sel = nonempty[RNG.random(len(nonempty)) < RATE]
        freq = df[B][df[B] != ""].value_counts(normalize=True)
        repl = RNG.choice(freq.index.values, size=len(sel), p=freq.values)
        keep = repl != df[B].values[sel]; sel, repl = sel[keep], repl[keep]
        dirty.iloc[sel, ci[B]] = repl; y[sel] = True
    return dirty, y


def cpad_score(df, fds, ci, n, k):
    cell = np.zeros((n, k))
    for A, B in fds:
        size = df.groupby(A)[B].transform("size").values
        own = df.groupby([A, B])[B].transform("size").values
        cell[:, ci[B]] = np.maximum(cell[:, ci[B]], 1.0 - np.where(size > 0, own / size, 1.0))
    return cell.max(1)


def marginal_score(df, cols, ci, n, k):
    m = np.zeros((n, k))
    for c in cols:
        m[:, ci[c]] = 1.0 - df[c].map(df[c].value_counts() / n).values
    return m.max(1)


def evaluate(name, clean):
    clean = clean.astype(str).fillna("").reset_index(drop=True)
    if len(clean) > ROW_CAP:
        clean = clean.sample(ROW_CAP, random_state=0).reset_index(drop=True)
    cols = list(clean.columns); ci = {c: j for j, c in enumerate(cols)}
    n, k = len(clean), len(cols)
    fds = discover_fds(clean, cols)
    if not fds:
        print(f"{name:<16}{n:>7}{k:>4}{0:>6}   no FD-governed column (out of scope)")
        return
    dirty, y = inject(clean, fds, ci)
    cp = roc_auc_score(y, cpad_score(dirty, fds, ci, n, k))
    mg = roc_auc_score(y, marginal_score(dirty, cols, ci, n, k))
    sub = dirty.sample(min(n, 12000), random_state=0)
    X = OneHotEncoder(handle_unknown="ignore", max_categories=40).fit_transform(sub).toarray()
    sc = -IsolationForest(random_state=0, n_estimators=150).fit(X).score_samples(
        OneHotEncoder(handle_unknown="ignore", max_categories=40).fit(sub).transform(dirty).toarray())
    ifr = roc_auc_score(y, sc)
    fill = (clean != "").mean().mean()
    print(f"{name:<16}{n:>7}{k:>4}{len(fds):>6}{100*fill:>6.0f}%{cp:>9.3f}{mg:>9.3f}{ifr:>9.3f}")


# ---- datasets (all through `evaluate`) -----------------------------------------------------
def load(path):
    return pd.read_csv(DATA + path, dtype=str)


print(f"Fair single-protocol detection (rate {RATE:.0%}, tau {TAU}, min_group {MIN_GROUP}, cap {ROW_CAP})")
print(f"{'dataset':<16}{'n':>7}{'d':>4}{'#FD':>6}{'fill':>7}{'CPAD':>9}{'Marg':>9}{'IFor':>9}")
print("-" * 67)
for name, path in [("Hospital", "hospital_clean_wide.csv"), ("Flights", "flights/clean.csv"),
                   ("Tax", "tax/clean.csv"), ("Adult", "adult.csv"),
                   ("Beers", "beers/clean.csv"), ("Rayyan", "rayyan/clean.csv"),
                   ("Movies", "movies_1/clean.csv")]:
    try:
        evaluate(name, load(path))
    except Exception as e:
        print(f"{name:<16} load/error: {type(e).__name__}: {e}")

# Wikidata: build live through the same SHACL pipeline, then the same evaluate()
try:
    sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/benchmarks")
    import wikidata_shacl as wd
    for tag, cls, props, nN in [
        ("Wikidata-geo", "Q486972",
         ["P17", "P131", "P30", "P421", "P1376", "P206", "P47", "P361", "P138",
          "P706", "P2936", "P37", "P190", "P571", "P856", "P1082"], 20000),
        ("Wikidata-film", "Q11424",
         ["P57", "P58", "P162", "P272", "P495", "P364", "P344", "P86", "P750"], 5000)]:
        t = wd.fetch_entity_table(cls, props, nN)
        flat = t[[c for c in t.columns if c.startswith("P") and not c.endswith("_cnt")]]
        evaluate(tag, flat)
except Exception as e:
    print(f"Wikidata: {type(e).__name__}: {e}")
