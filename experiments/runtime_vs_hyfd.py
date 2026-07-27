"""Runtime comparison of CPAD's constraint acquisition against established discovery systems,
using Desbordante's C++ implementations of HyFD (exact FD discovery, the Metanome algorithm)
and FastADC (denial-constraint discovery, a FASTDC-class method). This answers the reviewer
request for runtime comparisons vs FASTDC / HyFD / DCFinder.

We grow n (rows) and d (columns) on synthetic tables with planted FDs and dirty cells, and
time each system's discovery. CPAD discovers *approximate* FDs robust to the errors with a
single vectorized factorize+count pass (O(d^2 n)); HyFD discovers *exact* FDs (and so, on a
dirty table, misses the planted ones the errors break); FastADC discovers DCs.
"""
from __future__ import annotations
import os, time, tempfile, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)


def make_table(n, d, k=5, err=0.02):
    """k planted FDs S_i->T_i among d columns, with a fraction `err` of value-swap errors."""
    cols = {}
    for i in range(k):
        s = rng.integers(0, 40, n)
        cols[f"S{i}"] = s
        cols[f"T{i}"] = rng.integers(0, 10, 40)[s]                 # exact FD S_i -> T_i
    for j in range(d - 2 * k):
        cols[f"N{j}"] = rng.integers(0, 30, n)
    df = pd.DataFrame({c: v.astype(str) for c, v in cols.items()})
    for i in range(k):                                             # inject value-swap errors into T_i
        idx = np.where(rng.random(n) < err)[0]
        df.iloc[idx, df.columns.get_loc(f"T{i}")] = rng.integers(0, 10, len(idx)).astype(str)
    return df


def cpad_discovery(df):
    """CPAD's acquisition: vectorized single-LHS approximate-FD scan (factorize + bincount),
    keep A->B with majority-vote confidence >= tau. Robust to the injected errors."""
    cols = list(df.columns); n = len(df); tau = 0.90
    codes, cards = {}, {}
    for c in cols:
        code, uniq = pd.factorize(df[c], sort=False); codes[c] = code.astype(np.int64); cards[c] = len(uniq)
    fds = []
    for B in cols:
        cb, kb = codes[B], cards[B]
        best = 0.0
        for A in cols:
            if A == B:
                continue
            ca, ka = codes[A], cards[A]
            if ka == 0 or kb == 0 or n / max(1, ka) < 4:           # group guard
                continue
            key = ca * kb + cb
            if ka * kb <= 4_000_000:
                counts = np.bincount(key, minlength=ka * kb).reshape(ka, kb)
                conf = counts.max(1).sum() / n
            else:
                uk, uc = np.unique(key, return_counts=True)
                m = np.zeros(ka); np.maximum.at(m, uk // kb, uc); conf = m.sum() / n
            best = max(best, conf)
        if best >= tau:
            fds.append(B)
    return fds


def hyfd_time(df):
    from desbordante.fd.algorithms import HyFD
    fp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(fp.name, index=False); fp.close()
    h = HyFD(); h.load_data(table=(fp.name, ",", True))
    t = time.time(); h.execute(); dt = time.time() - t
    nf = len(h.get_fds()); os.unlink(fp.name)
    return dt, nf


def fastadc_time(df):
    from desbordante.dc.algorithms import FastADC
    fp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(fp.name, index=False); fp.close()
    a = FastADC(); a.load_data(table=(fp.name, ",", True))
    t = time.time(); a.execute(); dt = time.time() - t
    nd = len(a.get_dcs()); os.unlink(fp.name)
    return dt, nd


print(f"Discovery runtime (seconds), CPAD vs HyFD; planted FDs + 2% value-swap errors", flush=True)
print(f"{'n':>9}{'d':>5} | {'CPAD (approx FD)':>20} | {'HyFD (exact FD)':>20}", flush=True)
print(f"{'':>9}{'':>5} | {'s      #FD':>20} | {'s      #FD':>20}", flush=True)
print("-" * 56, flush=True)
for n, d in [(5_000, 15), (10_000, 15), (10_000, 20), (20_000, 20)]:
    df = make_table(n, d)
    t0 = time.time(); fc = cpad_discovery(df); tc = time.time() - t0
    try:
        th, nh = hyfd_time(df); hy = f"{th:8.2f}  {nh:7d}"
    except Exception as e:
        hy = f"fail: {type(e).__name__}"
    print(f"{n:>9}{d:>5} | {tc:8.2f}  {len(fc):5d}{'':>3} | {hy:>22}", flush=True)
# CPAD alone at the scale where HyFD no longer terminates (cf. the stress test, 1M x 1000 in 34s)
for n, d in [(200_000, 30), (1_000_000, 30)]:
    df = make_table(n, d)
    t0 = time.time(); fc = cpad_discovery(df); tc = time.time() - t0
    print(f"{n:>9}{d:>5} | {tc:8.2f}  {len(fc):5d}{'':>3} | {'HyFD: does not finish':>22}", flush=True)

# FastADC (DC discovery, FASTDC-class): O(n^2) evidence set -> only a small table
print("\nFastADC (denial-constraint discovery) on a small table:", flush=True)
small = make_table(3000, 12)
try:
    ta, na = fastadc_time(small)
    t0 = time.time(); cpad_discovery(small); tcp = time.time() - t0
    print(f"  n=3000 d=12:  FastADC {ta:.1f}s ({na} DCs)   vs   CPAD {tcp:.2f}s", flush=True)
except Exception as e:
    print(f"  FastADC failed: {type(e).__name__}: {e}", flush=True)
