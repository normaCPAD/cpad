import numpy as np
import pandas as pd

from cpad.core.table import Table
from cpad.core.constraint import DenialConstraint
from cpad.models.order import OrderCPAD


def _conditional_monotone_table(seed=0):
    """Within each context C, Y increases monotonically with X (noisy, so X->Y is not an
    exact FD). One planted tuple breaks the monotonic trend."""
    rng = np.random.default_rng(seed)
    n = 400
    C = rng.choice(["a", "b"], size=n)
    X = rng.integers(0, 40, size=n)
    offset = np.where(C == "a", 0, 100)
    Y = 2 * X + offset + rng.integers(0, 3, size=n)       # monotone in X within C, ties -> non-functional
    bad = 5
    X[bad], Y[bad] = 38, offset[bad] + 0                  # high X but lowest Y: breaks monotonicity
    df = pd.DataFrame({"ctx": C, "x": X.astype(str), "y": Y.astype(str)})
    return Table(df, name="ord"), bad


def test_order_cpad_discovers_order_dc():
    table, bad = _conditional_monotone_table()
    model = OrderCPAD(tau_threshold=0.7).fit(table)
    assert "y" in model.governed_                         # y is governed by an order dependency
    assert any(isinstance(r, DenialConstraint) for r in model.rules())
    # the discovered DC must use inequality predicates (not pure equality)
    dc = next(r for r in model.rules())
    assert any(p.op in ("<", ">") for p in dc.predicates)


def test_order_cpad_flags_monotonicity_violation():
    table, bad = _conditional_monotone_table()
    model = OrderCPAD(tau_threshold=0.7).fit(table)
    S = model.score(table)
    j = table.columns.index("y")
    assert int(np.argmax(S[:, j])) == bad                 # the trend-breaking cell scores highest
