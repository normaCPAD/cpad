import numpy as np
import pandas as pd

from cpad.core.table import Table
from cpad.models import DiscreteCPAD, RoutedCPAD


def _toy_table(seed=0):
    """country -> continent is deterministic, with one planted violation; an independent noise column."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 5, size=200)
    continent = np.array([f"continent{z}" for z in codes])
    continent[7] = "continentX"                          # the planted error (wrong continent for its country)
    df = pd.DataFrame({
        "country": [f"c{z}" for z in codes],
        "continent": continent,
        "noise": [f"n{v}" for v in rng.integers(0, 50, size=200)],
    })
    return Table(df, name="toy"), 7


def test_discrete_finds_fd_and_flags_error():
    table, bad_row = _toy_table()
    model = DiscreteCPAD(max_lhs=1).fit(table)
    rhs = {fd.rhs for fd in model.rules()}
    assert "continent" in rhs                             # country -> continent discovered
    S = model.score(table)
    j = table.columns.index("continent")
    assert int(np.argmax(S[:, j])) == bad_row            # the violation is the top-scored continent cell


def test_routed_runs_and_scores_all_columns():
    table, _ = _toy_table()
    model = RoutedCPAD().fit(table)
    S = model.score(table)
    assert S.shape == (table.n, len(table.columns))
    assert np.isfinite(S).all()
    assert (S >= 0).all() and (S <= 1).all()
