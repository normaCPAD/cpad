import numpy as np
import pandas as pd

from cpad.core.table import Table
from cpad.models import DiscreteCPAD
from cpad.repair import repair_table, RepairConfig


def _typo_table(seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 5, size=200)
    continent = np.array([f"continent{z}" for z in codes], dtype=object)
    bad = 7
    continent[bad] = "continentXX_typo"                  # a rare, non-conforming value
    df = pd.DataFrame({"country": [f"c{z}" for z in codes], "continent": continent})
    return Table(df, name="toy"), bad


def test_fd_repair_fixes_typo():
    table, bad = _typo_table()
    fds = DiscreteCPAD(max_lhs=1).fit(table).rules()
    cfg = RepairConfig(min_confidence=0.8, min_group=3, max_rarity=0.5)
    res = repair_table(table.df, fds, cfg, table.kinds)
    assert res.n_edits >= 1
    z = table.df["country"].iat[bad]
    correct = f"continent{z[1:]}"                        # cN -> continentN
    assert res.repaired["continent"].iat[bad] == correct  # repaired to the group's true value
    assert any(e.row == bad and e.column == "continent" for e in res.edits)


def test_repair_audit_log():
    table, bad = _typo_table()
    fds = DiscreteCPAD(max_lhs=1).fit(table).rules()
    res = repair_table(table.df, fds, RepairConfig(min_confidence=0.8, min_group=3, max_rarity=0.5), table.kinds)
    e = res.edits[0]
    assert e.kind == "FD" and e.old != e.new and 0.0 <= e.confidence <= 1.0
    assert "continent" in res.by_column


def test_repair_respects_confidence_gate():
    table, _ = _typo_table()
    fds = DiscreteCPAD(max_lhs=1).fit(table).rules()
    res = repair_table(table.df, fds, RepairConfig(min_confidence=0.999), table.kinds)
    assert res.n_edits == 0                              # FD conf < 0.999 -> nothing repaired
