"""Empirical complexity: wall-clock time and peak memory of CPAD's discrete engine as a
function of the number of rows n and columns d, confirming the O(d^2 n) analysis.
Addresses the "no empirical complexity / scalability" concern.
"""
from __future__ import annotations
import time
import tracemalloc
import numpy as np
import pandas as pd

from cpad.core.table import Table
from cpad.models import DiscreteCPAD

RNG = np.random.default_rng(0)


def synth(n, d, card=20):
    """A categorical table with some FD structure: a few columns are deterministic
    functions of a key column (so discovery has real work), the rest are random."""
    key = RNG.integers(0, max(card, n // 50), n)
    data = {}
    for j in range(d):
        if j % 3 == 0:                                   # governed: function of the key
            lut = RNG.integers(0, card, key.max() + 1)
            data[f"c{j}"] = lut[key]
        else:
            data[f"c{j}"] = RNG.integers(0, card, n)
    return pd.DataFrame({k: v.astype(str) for k, v in data.items()})


def measure(n, d):
    df = synth(n, d)
    t = Table(df)
    tracemalloc.start()
    t0 = time.perf_counter()
    m = DiscreteCPAD().fit(t); _ = m.score(t)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return dt, peak / 1e6, len(getattr(m, "rules_", []))


if __name__ == "__main__":
    print("CPAD discrete engine: fit + score wall-clock (s), peak memory (MB), #constraints")
    print("\n== scaling rows n (d=20) ==")
    print(f"{'n':>8} {'time(s)':>9} {'mem(MB)':>9} {'#cons':>6}")
    for n in [1000, 5000, 20000, 50000, 100000]:
        dt, mem, nc = measure(n, 20)
        print(f"{n:>8} {dt:>9.2f} {mem:>9.1f} {nc:>6}")
    print("\n== scaling columns d (n=10000) ==")
    print(f"{'d':>8} {'time(s)':>9} {'mem(MB)':>9} {'#cons':>6}")
    for d in [10, 20, 40, 80, 120]:
        dt, mem, nc = measure(10000, d)
        print(f"{d:>8} {dt:>9.2f} {mem:>9.1f} {nc:>6}")
