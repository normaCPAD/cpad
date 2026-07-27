"""The 3-strata "error != rare" benchmark: the protocol that measures, on *real* data with
known denial constraints, whether a detector separates errors from rare-but-valid values.

Motivation.  CPAD's central thesis is that a data-quality error is a *constraint violation*,
not a statistical outlier. Today this is shown with injected value-swaps and a proxy (the
false-positive rate on rare-but-valid synthetic rows). The decisive test is a real table whose
cells are partitioned into three strata:

    (A) ERROR        - a cell edited to VIOLATE a denial constraint (BART-style injection)
    (B) RARE_VALID   - a cell holding a marginally RARE value that nonetheless SATISFIES every
                       constraint (the hard negative: rare but correct)
    (C) COMMON_VALID - a cell holding a common, conforming value

A density detector ranks (B) as high as (A) -- it cannot tell a rare error from a rare truth.
A constraint detector ranks (A) high and (B) low. The benchmark's headline metric is therefore
the AUROC/AUPRC that separates (A) from (B): the direct, real-data measurement of "error != rare".

This module scaffolds the generator (BART-style DC-violating injection + controlled rare-valid
sampling), the per-cell stratum labels, the evaluation, and the human-annotation schema used to
certify stratum (B) on real data (the one part that needs an expert, not code).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# per-cell stratum codes
COMMON_VALID, RARE_VALID, ERROR = 0, 1, 2
STRATUM_NAME = {COMMON_VALID: "common_valid", RARE_VALID: "rare_valid", ERROR: "error"}


@dataclass
class FD:
    """A functional dependency lhs -> rhs used both to inject errors and to define conformance."""
    lhs: tuple
    rhs: str


# --------------------------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------------------------
def _group_mode(df: pd.DataFrame, lhs: Sequence[str], rhs: str) -> pd.Series:
    return df.groupby(list(lhs))[rhs].transform(lambda s: s.value_counts().idxmax())


def build_3strata(clean: pd.DataFrame, fds: list[FD], *, error_rate: float = 0.04,
                  rare_freq: float = 0.005, seed: int = 0
                  ) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    """Return (dirty, strata, log).

    `strata[i, j]` in {COMMON_VALID, RARE_VALID, ERROR} is the ground-truth stratum of cell
    (row i, col j). `log` records every (A) and (B) cell with the constraint / frequency that
    justifies its label, so the benchmark is auditable rather than frequency-derived.

    (A) ERROR: for an FD lhs->rhs, replace the rhs of selected rows by another value that is
        valid *elsewhere* but breaks the FD in this group (a value-swap, marginally common).
    (B) RARE_VALID: cells whose value is in the bottom `rare_quantile` of column frequency yet
        equals its group's modal rhs (so it conforms). Optionally inject extra rare-valid cells
        by moving a row to a rare-but-internally-consistent (lhs, rhs) combination.
    """
    rng = np.random.default_rng(seed)
    df = clean.copy().reset_index(drop=True)
    cols = list(df.columns); ci = {c: j for j, c in enumerate(cols)}
    n = len(df)
    strata = np.zeros((n, len(cols)), dtype=np.int8)
    log: list[dict] = []
    governed = {fd.rhs for fd in fds}

    # ---- stratum (A): DC-violating edits ----
    for fd in fds:
        mode = _group_mode(df, fd.lhs, fd.rhs)
        pool = df[fd.rhs].value_counts()                  # valid values seen in the column
        common = pool.index.values
        sel = np.where(rng.random(n) < error_rate)[0]
        for i in sel:
            wrong = rng.choice(common)
            if wrong != mode.iat[i] and wrong != df.at[i, fd.rhs]:   # must break the FD
                df.at[i, fd.rhs] = wrong
                strata[i, ci[fd.rhs]] = ERROR
                log.append({"row": int(i), "col": fd.rhs, "stratum": "error",
                            "constraint": f"{'+'.join(fd.lhs)}->{fd.rhs}",
                            "expected": str(mode.iat[i]), "got": str(wrong)})

    # ---- stratum (B): rare-but-valid (absolute-rare values that still satisfy every FD) ----
    rare_count = max(2, int(rare_freq * n))               # a value is "rare" if seen <= this many times
    for c in cols:
        vc = df[c].value_counts()
        freq = df[c].map(vc / n)
        rare_vals = set(vc[vc <= rare_count].index)
        conform = np.ones(n, bool)
        for fd in fds:                                    # a cell conforms if it equals its group mode
            if fd.rhs == c:
                conform &= (df[c].values == _group_mode(df, fd.lhs, fd.rhs).values)
        rare_valid = df[c].isin(rare_vals).values & conform & (strata[:, ci[c]] == COMMON_VALID)
        for i in np.where(rare_valid)[0]:
            strata[i, ci[c]] = RARE_VALID
            log.append({"row": int(i), "col": c, "stratum": "rare_valid",
                        "freq": float(freq.values[i]), "conforms": True})

    return df, strata, log


# --------------------------------------------------------------------------------------------
# evaluation -- the "error != rare" separation
# --------------------------------------------------------------------------------------------
def evaluate_separation(scores: np.ndarray, strata: np.ndarray, governed_cols=None,
                        col_index=None) -> dict:
    """`scores[i, j]` is a detector's per-cell anomaly score. Returns the headline metrics:

      err_vs_rare  : AUROC separating ERROR from RARE_VALID  (== 1 means "error != rare" perfectly;
                     ~0.5 means the detector confuses a rare truth with an error -- a density detector)
      err_vs_all   : AUROC separating ERROR from everything valid (B + C)
      rare_fpr@k   : fraction of RARE_VALID cells in the top-k% scored cells (want it LOW)
    """
    s = scores.ravel(); z = strata.ravel()
    err, rare, common = z == ERROR, z == RARE_VALID, z == COMMON_VALID
    out = {}
    if err.any() and rare.any():
        m = err | rare
        out["err_vs_rare_auroc"] = float(roc_auc_score(err[m], s[m]))
        out["err_vs_rare_auprc"] = float(average_precision_score(err[m], s[m]))
    if err.any() and (rare | common).any():
        m = err | rare | common
        out["err_vs_all_auroc"] = float(roc_auc_score(err[m], s[m]))
    k = max(1, int(0.02 * len(s)))
    top = np.argsort(-s)[:k]
    out["rare_fpr@2pct"] = float(rare[top].mean())
    out["n_error"], out["n_rare_valid"] = int(err.sum()), int(rare.sum())
    return out


# --------------------------------------------------------------------------------------------
# annotation schema -- the human-certified part (stratum B on real data)
# --------------------------------------------------------------------------------------------
ANNOTATION_SCHEMA = {
    "task": "Certify the RARE_VALID stratum: for each flagged low-frequency cell, decide whether "
            "the rare value is an ERROR (violates a real-world constraint) or RARE_BUT_VALID "
            "(uncommon yet correct). This is the judgement a frequency model cannot make.",
    "unit": "one cell (row id, column)",
    "fields": {
        "row_id": "stable identifier of the tuple",
        "column": "attribute name",
        "value": "the cell value under review",
        "marginal_freq": "value frequency in the column (provided, for reference only)",
        "candidate_constraints": "DCs whose left-hand side governs this column (provided)",
        "label": "one of {ERROR, RARE_VALID, UNSURE}",
        "violated_constraint": "if ERROR, the constraint id it breaks (else empty)",
        "rationale": "one line: why it is rare-but-valid, or which constraint it breaks",
        "annotator": "annotator id",
        "double_checked": "true if adjudicated by a second annotator",
    },
    "protocol": [
        "Sample cells in the bottom 2% of column frequency (the RARE candidates).",
        "Show the annotator the tuple, the governing DCs, and sibling tuples in the same DC group.",
        "Annotator labels ERROR vs RARE_VALID using the constraints, not the frequency.",
        "Two annotators per cell; disagreements adjudicated; report inter-annotator agreement.",
        "Only cells labeled RARE_VALID by both enter stratum (B); ERROR cells enter stratum (A).",
    ],
    "headline_metric": "err_vs_rare_auroc on the certified (A) vs (B) cells.",
}


def emit_annotation_sheet(path: str, dirty: pd.DataFrame, strata: np.ndarray,
                          fds: list[FD]) -> None:
    """Write a CSV pre-filled with the RARE candidates for human review (stratum-B certification)."""
    cols = list(dirty.columns); n = len(dirty)
    gov = {fd.rhs: [f"{'+'.join(fd.lhs)}->{fd.rhs}" for fd in fds if fd.rhs == fd.rhs] for fd in fds}
    rows = []
    for j, c in enumerate(cols):
        freq = dirty[c].map(dirty[c].value_counts() / n)
        for i in np.where(strata[:, j] == RARE_VALID)[0]:
            rows.append({"row_id": i, "column": c, "value": dirty.at[i, c],
                         "marginal_freq": round(float(freq.values[i]), 5),
                         "candidate_constraints": ";".join(gov.get(c, [])),
                         "label": "", "violated_constraint": "", "rationale": "",
                         "annotator": "", "double_checked": ""})
    pd.DataFrame(rows).to_csv(path, index=False)
    json.dump(ANNOTATION_SCHEMA, open(path.replace(".csv", "_schema.json"), "w"), indent=2)
    print(f"[three_strata] annotation sheet -> {path} ({len(rows)} rare candidates to certify)")


# --------------------------------------------------------------------------------------------
# self-test: build strata on a tiny table with a known FD and check the separation works
# --------------------------------------------------------------------------------------------
def selftest() -> None:
    rng = np.random.default_rng(0)
    zc = rng.integers(0, 30, 1990)
    countries = [f"country{z}" for z in zc]; continents = [f"cont{z}" for z in zc]   # 30 common country->continent pairs
    for k in range(10):                                  # 10 GENUINE rare-but-valid: singleton country->unique continent
        countries.append(f"countryrare{k}"); continents.append(f"rarecont{k}")
    df = pd.DataFrame({"country": countries, "continent": continents,
                       "note": list(rng.choice(["a", "b", "c"], len(countries)))})

    fds = [FD(("country",), "continent")]                 # exact FD country -> continent
    dirty, strata, log = build_3strata(df, fds, error_rate=0.05, rare_freq=0.005, seed=1)

    # a constraint detector: per-cell FD violation 1 - freq(continent | country)
    viol = np.zeros(strata.shape)
    mode = _group_mode(dirty, ("country",), "continent")
    viol[:, list(dirty.columns).index("continent")] = (dirty["continent"].values != mode.values).astype(float)
    # a density detector: marginal rarity of the continent value
    dens = np.zeros(strata.shape)
    dens[:, list(dirty.columns).index("continent")] = (1 - dirty["continent"].map(
        dirty["continent"].value_counts() / len(dirty)).values)

    con = evaluate_separation(viol, strata)
    den = evaluate_separation(dens, strata)
    print(f"[three_strata] strata: {con['n_error']} ERROR, {con['n_rare_valid']} RARE_VALID")
    print(f"  constraint detector  err_vs_rare AUROC = {con.get('err_vs_rare_auroc', float('nan')):.3f}"
          f"   (should be ~1: error != rare)")
    print(f"  density detector     err_vs_rare AUROC = {den.get('err_vs_rare_auroc', float('nan')):.3f}"
          f"   (should be ~0.5: confuses rare with error)")
    assert con.get("err_vs_rare_auroc", 0) > 0.9, "constraint detector should separate error from rare"
    print("[three_strata] selftest OK")


if __name__ == "__main__":
    selftest()
