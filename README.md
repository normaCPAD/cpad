# CPAD — Constrained Predicates for Anomaly Detection

**Source code: https://github.com/normaCPAD/cpad**

CPAD learns the *denial constraints* (functional dependencies, order and linear
dependencies) that a relational table is supposed to satisfy, and scores each cell by
how strongly it violates them. Unlike statistical outlier detectors, CPAD targets
*relational inconsistency*: a value that is perfectly common on its own but impossible
in its row. Discovery is fully **unsupervised** and driven by a contrastive,
value-swap corrupter.

This repository contains the CPAD engine and the scripts that reproduce every
experiment in the paper. The companion product **[norma](https://github.com/normaCPAD/norma)** (interactive
desktop studio: discovery → 3NF/BCNF normalization → repair → clean database) is built
on this same engine.

## Install

```bash
git clone https://github.com/normaCPAD/cpad
cd cpad
pip install -e ".[baselines,gated]"     # core + pyod baselines + torch (gated variant)
```

The core engine needs only `numpy`, `pandas`, `scipy`, `scikit-learn`. `torch` is
optional (the differentiable `GatedCPAD` variant); `pyod` is optional (experiment
baselines).

## Quickstart

```python
import pandas as pd
from cpad.models import RoutedCPAD

df = pd.read_csv("data/hospital_dirty.csv", dtype=str).fillna("")
model = RoutedCPAD().fit(df)        # discover constraints, no labels
scores = model.score(df)            # per-cell violation scores
for rule in model.rules():
    print(rule)
```

## Layout

```
cpad/
├── cpad/                 # the engine (importable package)
│   ├── core/             # Table, constraints (FD / denial / linear), value-swap corrupter
│   ├── models/           # DiscreteCPAD, GatedCPAD, OrderCPAD, LinearCPAD, EnsembleCPAD, RoutedCPAD
│   ├── detect/           # marginal baseline scoring
│   ├── rules/            # rule extraction + confidence
│   ├── modeling/         # closure, candidate keys, 3NF/BCNF synthesis, reporting
│   └── repair.py         # constraint-guided cell repair
├── experiments/          # one script per paper table/figure  (see experiments/README.md)
├── data/                 # Hospital + Adult bundled; see data/README.md for the rest
└── tests/                # unit tests (pytest)
```

## Reproducing the paper

Every table and figure maps to a script in [`experiments/`](experiments/README.md):

```bash
python experiments/cpad_categorical.py     # Hospital detection
python experiments/repair.py               # constraint-guided repair
python experiments/value_swap.py           # the value-swap test
...
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

Released for research use. If you use CPAD, please cite the paper.
