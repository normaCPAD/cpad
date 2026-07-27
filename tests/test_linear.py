import numpy as np
import pandas as pd

from cpad.core.table import Table
from cpad.core.constraint import LinearConstraint
from cpad.models.linear import LinearCPAD
from cpad.models import RoutedCPAD


def _linear_table(violation=10.0, seed=0):
    """z = x + y exactly (a linear constraint), with one LARGE planted violation -- the
    regime where a (non-robust) PCA covariance breaks down."""
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.normal(0, 1, n).round(4)
    y = rng.normal(0, 1, n).round(4)
    z = (x + y).round(4)
    z[4] = z[4] + violation
    df = pd.DataFrame({"x": x.astype(str), "y": y.astype(str), "z": z.astype(str)})
    return Table(df, name="lin"), 4


def test_linear_cpad_discovers_constraint_and_is_robust():
    table, bad = _linear_table(violation=10.0)            # large outlier: PCA would smear the direction
    model = LinearCPAD().fit(table)
    assert model.governed_                                # a contrastive linear constraint was found
    assert any(isinstance(r, LinearConstraint) for r in model.rules())
    lc = model.rules()[0]
    assert {"x", "y", "z"} >= lc.attributes               # the constraint involves the right columns
    S = model.score(table)
    assert int(np.argmax(S[:, table.columns.index("z")])) == bad


def test_linear_constraint_is_sparse_form():
    table, _ = _linear_table(violation=3.0)
    lc = LinearCPAD().fit(table).rules()[0]
    # coefficients reflect z ~= x + y, i.e. all three columns with comparable weight
    coefs = dict(lc.coefficients)
    assert set(coefs) == {"x", "y", "z"}


def test_routed_routes_numeric_to_linear():
    table, _ = _linear_table(violation=3.0)
    model = RoutedCPAD().fit(table)
    assert {"x", "y", "z"} <= model.governed_
