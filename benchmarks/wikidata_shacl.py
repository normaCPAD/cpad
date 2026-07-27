"""Build a *constraint-labeled* error-detection benchmark from Wikidata property constraints.

Why this benchmark.  Wikidata curates ~30 constraint types on its properties through the
"property constraint" statement (P2302): single-value (Q19474404), distinct-values
(Q21502410), one-of (Q21510859), type (Q21503250), value-type (Q21510865), inverse
(Q21510855), item-requires-statement (Q21503247), conflicts-with (Q21502838), ... These are
real, human-maintained *denial constraints*, and the items that break them are real,
constraint-labeled errors -- not statistical outliers. That is exactly the regime CPAD
targets ("error != rare"), and, to our knowledge, no unsupervised data-quality detector has
been evaluated against it. The labels come from the constraints, so they are independent of
the value frequencies a density detector would key on.

Pipeline (no Wikidata dump needed; runs over the public Query Service via SPARQL/HTTPS):

    1. fetch_property_constraints(props)        -> the P2302 constraints declared on each prop
    2. fetch_entity_table(class_qid, props, n)  -> a flat relational table (rows=items, cols=props)
    3. label_violations(table, constraints)     -> per-cell ground-truth mask + the broken constraint
    4. emit_cpad_dataset(out_dir, ...)           -> clean.csv / dirty.csv / errmask.npy / constraints.json

The emitted files match the layout the CPAD experiments already load (DATA/<name>/{clean,dirty}.csv
plus an error mask), so a Wikidata benchmark drops straight into `broad_benchmarks.py`.

Usage:
    python -m cpad.benchmarks.wikidata_shacl --class Q11424 \
        --props P57 P58 P162 P272 P136 P495 --n 5000 --out DATA/wikidata_film
    (Q11424 = film; P57 director, P58 screenwriter, P162 producer, P272 production company,
     P136 genre, P495 country of origin)

Network is required for the live build; `OFFLINE_FIXTURE` lets the labeler be unit-tested
without it (see `selftest`).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:                                       # keep the module importable offline
    requests = None

WDQS = "https://query.wikidata.org/sparql"
UA = "CPAD-constraint-benchmark/1.0 (research; unsupervised data-quality detection)"

# constraint-type QIDs we materialize onto a flat table
C_SINGLE_VALUE = "Q19474404"
C_DISTINCT     = "Q21502410"
C_ONE_OF       = "Q21510859"
C_TYPE         = "Q21503250"
C_VALUE_TYPE   = "Q21510865"
C_REQUIRES     = "Q21503247"      # item-requires-statement: if prop set, another prop must be set
C_CONFLICTS    = "Q21502838"      # conflicts-with: if prop set, another prop must NOT be set


# --------------------------------------------------------------------------------------------
# SPARQL plumbing
# --------------------------------------------------------------------------------------------
def _sparql(query: str, retries: int = 4, pause: float = 1.0) -> list[dict]:
    if requests is None:
        raise RuntimeError("the `requests` package is required for the live build")
    for attempt in range(retries):
        try:                                              # POST avoids URL-length limits; tolerate
            r = requests.post(WDQS, data={"query": query, "format": "json"},   # truncated/slow replies
                              headers={"User-Agent": UA,
                                       "Accept": "application/sparql-results+json"}, timeout=180)
            if r.status_code == 200:
                return r.json()["results"]["bindings"]
        except Exception:
            pass
        time.sleep(pause * (attempt + 1))                 # back off on 429/503/truncation
    return []                                             # give up gracefully; caller handles empty


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------------------------
# 1. constraints declared on the chosen properties
# --------------------------------------------------------------------------------------------
def fetch_property_constraints(props: Iterable[str]) -> dict[str, list[dict]]:
    """For each property Pxx, read its P2302 constraints. Returns
    {prop: [{'type': <constraint QID>, 'allowed': [...], 'class': [...], 'other': <prop>}]}."""
    out: dict[str, list[dict]] = {p: [] for p in props}
    values = " ".join(f"wd:{p}" for p in props)
    # constraint type + the qualifiers we use: P2305 (item of allowed set / class), P2306 (property)
    q = f"""
    SELECT ?prop ?ctype ?allowed ?otherprop WHERE {{
      VALUES ?prop {{ {values} }}
      ?prop p:P2302 ?st .
      ?st ps:P2302 ?ctype .
      OPTIONAL {{ ?st pq:P2305 ?allowed . }}
      OPTIONAL {{ ?st pq:P2306 ?otherprop . }}
    }}"""
    rows = _sparql(q)
    agg: dict[tuple[str, str], dict] = {}
    for b in rows:
        p = _qid(b["prop"]["value"]); ct = _qid(b["ctype"]["value"])
        key = (p, ct)
        rec = agg.setdefault(key, {"type": ct, "allowed": [], "class": [], "other": None})
        if "allowed" in b:
            rec["allowed"].append(_qid(b["allowed"]["value"]))
            rec["class"].append(_qid(b["allowed"]["value"]))
        if "otherprop" in b:
            rec["other"] = _qid(b["otherprop"]["value"])
    for (p, _), rec in agg.items():
        out[p].append(rec)
    return out


# --------------------------------------------------------------------------------------------
# 2. a flat relational table of items x properties
# --------------------------------------------------------------------------------------------
def fetch_item_ids(class_qid: str, n: int, page: int = 10000) -> list[str]:
    """Paginate the items of a class (LIMIT/OFFSET) up to `n`. For very large n, partition the
    class instead (e.g. add a year/country filter); OFFSET past ~100k is slow on WDQS."""
    ids: list[str] = []
    offset = 0
    while len(ids) < n:
        take = min(page, n - len(ids))
        rows = _sparql(f"SELECT ?item WHERE {{ ?item wdt:P31 wd:{class_qid} . }} "
                       f"LIMIT {take} OFFSET {offset}")
        if not rows:
            break
        ids += [_qid(b["item"]["value"]) for b in rows]
        offset += len(rows)
        if len(rows) < take:
            break
    return ids[:n]


def fetch_item_properties(items: list[str], props: list[str], batch: int = 120
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batched long->wide fetch that scales to high n AND high d. For each batch of items we pull
    all (item, property, value) triples in one query, so the table is built from a few hundred
    cheap queries instead of one OPTIONAL cross-product. Returns (values, counts); the value
    table is naturally SPARSE -- most items declare only a few of the properties."""
    val: dict[str, dict] = defaultdict(dict)
    cnt: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    pset = " ".join(f"wdt:{p}" for p in props)

    def grab(chunk):
        vlist = " ".join(f"wd:{q}" for q in chunk)
        rows = _sparql(f"SELECT ?item ?p ?v WHERE {{ VALUES ?item {{ {vlist} }} "
                       f"VALUES ?p {{ {pset} }} ?item ?p ?v . }}")
        if not rows and len(chunk) > 12:                   # empty => likely truncated: split and retry
            mid = len(chunk) // 2
            grab(chunk[:mid]); grab(chunk[mid:]); return
        for b in rows:
            it = _qid(b["item"]["value"]); p = _qid(b["p"]["value"])
            v = b["v"]["value"]; v = _qid(v) if v.startswith("http") else v
            cnt[it][p] += 1
            val[it].setdefault(p, v)                        # keep the first value seen

    for s in range(0, len(items), batch):
        grab(items[s:s + batch])
    table = pd.DataFrame([{"item": it, **{p: val[it].get(p, "") for p in props}} for it in items])
    counts = pd.DataFrame([{"item": it, **{p: cnt[it].get(p, 0) for p in props}} for it in items])
    return table.fillna(""), counts


def fetch_entity_table(class_qid: str, props: list[str], n: int = 20000) -> pd.DataFrame:
    """Convenience wrapper: paginate item ids, then batch-fetch their property values. The result
    has one row per item, one column per property (sparse), plus per-property `_cnt` multiplicity
    columns used by the single-value constraint."""
    items = fetch_item_ids(class_qid, n)
    table, counts = fetch_item_properties(items, props)
    for p in props:
        table[p + "_cnt"] = counts[p].values
    dens = (table[props] != "").mean()
    print(f"[wikidata_shacl] {len(table)} items x {len(props)} props; "
          f"mean fill rate {dens.mean():.1%} (sparse: many empty cells to complete)")
    return table


# --------------------------------------------------------------------------------------------
# 3. label the constraint violations -> ground truth
# --------------------------------------------------------------------------------------------
def label_violations(table: pd.DataFrame, constraints: dict[str, list[dict]],
                     value_classes: dict[str, set] | None = None) -> tuple[np.ndarray, list[dict]]:
    """Return (mask, log). `mask[i, j]` is True iff cell (row i, property col j) breaks a
    constraint. `log` records, per violation, the item / property / constraint type, so the
    labels are explained by a *rule*, not by frequency.

    Implemented locally (no extra SPARQL): one-of (value not in allowed set), single-value
    (multiplicity > 1), requires/conflicts (cross-property presence DCs). Type / value-type are
    recorded and can be checked with `value_classes` (item -> set of P31 classes) when available.
    """
    props = [c for c in table.columns if c.startswith("P")]
    col = {p: j for j, p in enumerate(props)}
    mask = np.zeros((len(table), len(props)), bool)
    log: list[dict] = []

    def flag(i, p, ctype, detail=""):
        mask[i, col[p]] = True
        log.append({"row": int(i), "prop": p, "constraint": ctype, "detail": detail})

    for p in props:
        cnt_col = p + "_cnt"
        for rec in constraints.get(p, []):
            ct, allowed, other = rec["type"], set(rec.get("allowed", [])), rec.get("other")
            if ct == C_ONE_OF and allowed:
                for i, v in enumerate(table[p].values):
                    if v and v not in allowed:
                        flag(i, p, ct, f"{v} not in one-of set")
            elif ct == C_SINGLE_VALUE and cnt_col in table.columns:
                for i, c in enumerate(table[cnt_col].values):
                    if isinstance(c, (int, np.integer)) and c > 1:
                        flag(i, p, ct, f"{c} values (single-value)")
            elif ct == C_REQUIRES and other in col:
                for i in range(len(table)):
                    if table[p].iat[i] and not table[other].iat[i]:
                        flag(i, p, ct, f"requires {other}")
            elif ct == C_CONFLICTS and other in col:
                for i in range(len(table)):
                    if table[p].iat[i] and table[other].iat[i]:
                        flag(i, p, ct, f"conflicts with {other}")
            elif ct in (C_TYPE, C_VALUE_TYPE) and value_classes is not None:
                for i, v in enumerate(table[p].values):
                    cls = value_classes.get(v, set())
                    if v and allowed and not (cls & allowed):
                        flag(i, p, ct, f"{v} not of required class")

    if C_DISTINCT:                                        # key constraints: duplicate values
        for p in props:
            if any(r["type"] == C_DISTINCT for r in constraints.get(p, [])):
                dup = table[p].duplicated(keep=False) & (table[p] != "")
                for i in np.where(dup.values)[0]:
                    flag(i, p, C_DISTINCT, "duplicate value")
    return mask, log


# --------------------------------------------------------------------------------------------
# 3b. suggest values for EMPTY cells (constraint-guided completion)
# --------------------------------------------------------------------------------------------
def discover_fds(table: pd.DataFrame, props: list[str], tau: float = 0.9,
                 min_support: int = 30, min_group: float = 4.0) -> list[tuple[str, str, float]]:
    """Light single-LHS FD discovery on the *observed* (non-empty) cells: A -> B if, among rows
    where both A and B are present, B equals its A-group majority for >= tau of them. Returns
    (A, B, confidence). `min_group` rejects near-key sources (A almost unique), whose singleton
    groups make confidence trivially 1 but carry no detectable signal -- only FDs with real
    groups are kept, which is what makes both completion and violation detection meaningful."""
    fds = []
    for B in props:
        best = None
        for A in props:
            if A == B:
                continue
            sub = table[(table[A] != "") & (table[B] != "")]
            if len(sub) < min_support or len(sub) / max(1, sub[A].nunique()) < min_group:
                continue                                   # too small, or A is near-key (no groups)
            conf = sub.groupby(A)[B].transform(lambda s: s.value_counts(normalize=True).max()).mean()
            if conf >= tau and (best is None or conf > best[1]):
                best = (A, float(conf))
        if best:
            fds.append((best[0], B, best[1]))
    return fds


def suggest_completions(table: pd.DataFrame, props: list[str],
                        fds: list[tuple[str, str, float]] | None = None,
                        constraints: dict | None = None, tau: float = 0.9) -> list[dict]:
    """Propose a value for EMPTY governed cells. For every FD A -> B and every row with A present
    but B empty, suggest B = the majority value of the row's A-group, with the group's purity as
    confidence. Also flags empties that an item-requires-statement constraint says must be filled.
    Returns a ranked list of {row, col, suggest, confidence, basis}."""
    fds = discover_fds(table, props) if fds is None else fds
    sugg: list[dict] = []
    for A, B, _ in fds:
        sub = table[table[A] != ""]
        grp = sub.groupby(A)[B]
        mode = grp.agg(lambda s: s[s != ""].value_counts().idxmax() if (s != "").any() else "")
        purity = grp.agg(lambda s: s[s != ""].value_counts(normalize=True).max() if (s != "").any() else 0.0)
        empty = (table[B] == "") & (table[A] != "")
        for i in np.where(empty.values)[0]:
            a = table[A].iat[i]; v = mode.get(a, ""); c = float(purity.get(a, 0.0))
            if v and c >= tau:
                sugg.append({"row": int(i), "col": B, "suggest": v,
                             "confidence": round(c, 3), "basis": f"{A}->{B}"})
    if constraints:                                       # requires-statement: B empty but mandated
        for p in props:
            for rec in constraints.get(p, []):
                other = rec.get("other")
                if rec["type"] == C_REQUIRES and other in props:
                    need = (table[p] != "") & (table[other] == "")
                    for i in np.where(need.values)[0]:
                        sugg.append({"row": int(i), "col": other, "suggest": "",
                                     "confidence": 1.0, "basis": f"requires({p}->{other})"})
    sugg.sort(key=lambda d: -d["confidence"])
    return sugg


# --------------------------------------------------------------------------------------------
# 4. emit a CPAD-ready dataset
# --------------------------------------------------------------------------------------------
def emit_cpad_dataset(out_dir: str, table: pd.DataFrame, mask: np.ndarray,
                      constraints: dict, log: list[dict],
                      suggestions: list[dict] | None = None) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    props = [c for c in table.columns if c.startswith("P") and not c.endswith("_cnt")]
    flat = table[props].astype(str)
    flat.to_csv(f"{out_dir}/dirty.csv", index=False)      # the table as-is (contains the violations)
    clean = flat.copy()
    clean.values[mask[:, [list(table.columns).index(p) for p in props]]] = ""   # blank flagged cells
    clean.to_csv(f"{out_dir}/clean.csv", index=False)
    np.save(f"{out_dir}/errmask.npy", mask[:, [i for i, c in enumerate(table.columns) if c in props]])
    fill = (flat != "").mean()
    json.dump({"constraints": constraints, "n_violations": int(mask.sum()),
               "fill_rate_per_prop": {p: round(float(fill[p]), 3) for p in props},
               "violation_log": log[:1000]}, open(f"{out_dir}/constraints.json", "w"), indent=2)
    if suggestions is not None:
        json.dump(suggestions[:5000], open(f"{out_dir}/completions.json", "w"), indent=2)
    print(f"[wikidata_shacl] {out_dir}: {len(flat)} rows x {len(props)} props, "
          f"{int(mask.sum())} constraint-labeled errors, "
          f"{0 if suggestions is None else len(suggestions)} empty-cell completions suggested "
          f"(mean fill {fill.mean():.1%})")


# --------------------------------------------------------------------------------------------
# offline self-test of the labeler (no network)
# --------------------------------------------------------------------------------------------
def selftest() -> None:
    table = pd.DataFrame({
        "item": ["Q1", "Q2", "Q3", "Q4"],
        "P1": ["Qa", "Qb", "Qx", "Qa"],      # one-of {Qa,Qb,Qc}; Qx (row2) violates
        "P1_cnt": [1, 2, 1, 1],              # single-value; row1 has 2 -> violates
        "P2": ["", "v", "v", ""],            # P3 requires P2
        "P3": ["y", "y", "", "y"],           # rows 0 and 3 have P3 but no P2 -> requires-violation
    })
    constraints = {
        "P1": [{"type": C_ONE_OF, "allowed": ["Qa", "Qb", "Qc"], "class": [], "other": None},
               {"type": C_SINGLE_VALUE, "allowed": [], "class": [], "other": None}],
        "P2": [], "P3": [{"type": C_REQUIRES, "allowed": [], "class": [], "other": "P2"}],
    }
    mask, log = label_violations(table, constraints)
    got = {(d["row"], d["prop"], d["constraint"]) for d in log}
    expect = {(2, "P1", C_ONE_OF), (1, "P1", C_SINGLE_VALUE),
              (0, "P3", C_REQUIRES), (3, "P3", C_REQUIRES)}
    assert got == expect, f"labeler mismatch:\n got={got}\n exp={expect}"
    print("[wikidata_shacl] labeler OK -", len(log), "violations labeled by constraint:", sorted(got))

    # completion: an FD P4 -> P5 with one empty P5 cell to fill from its group majority
    comp = pd.DataFrame({
        "item": [f"Q{i}" for i in range(6)],
        "P4": ["a", "a", "a", "b", "b", "b"],
        "P5": ["x", "x", "", "y", "y", "y"],             # row 2: P4=a present, P5 empty -> suggest "x"
    })
    fds = discover_fds(comp, ["P4", "P5"], tau=0.9, min_support=2)
    sugg = suggest_completions(comp, ["P4", "P5"], fds=fds, tau=0.9)
    assert any(s["row"] == 2 and s["col"] == "P5" and s["suggest"] == "x" for s in sugg), \
        f"completion failed: {sugg}"
    print("[wikidata_shacl] completion OK - suggested", sugg[0]["suggest"],
          f"for empty cell (row 2, P5) via {sugg[0]['basis']} @ conf {sugg[0]['confidence']}")
    print("[wikidata_shacl] selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Wikidata constraint-labeled benchmark for CPAD.")
    ap.add_argument("--class", dest="cls", help="class QID, e.g. Q11424 (film)")
    ap.add_argument("--props", nargs="+", help="property PIDs, e.g. P57 P136 P495")
    ap.add_argument("--n", type=int, default=20000, help="number of items (paginated; >5000 ok)")
    ap.add_argument("--out", default="DATA/wikidata")
    ap.add_argument("--selftest", action="store_true", help="run the offline labeler test and exit")
    a = ap.parse_args()
    if a.selftest or not (a.cls and a.props):
        selftest(); return
    constraints = fetch_property_constraints(a.props)
    table = fetch_entity_table(a.cls, a.props, a.n)
    mask, log = label_violations(table, constraints)
    sugg = suggest_completions(table, a.props, constraints=constraints)   # fill the empty cells
    emit_cpad_dataset(a.out, table, mask, constraints, log, suggestions=sugg)


if __name__ == "__main__":
    main()
