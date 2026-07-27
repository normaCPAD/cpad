"""Shared helpers for the CPAD reproduction experiments.

`DATA` resolves the dataset root in this order:
  1. the CPAD_DATA environment variable, if set;
  2. a sibling `data_dq/` directory (the full benchmark, incl. Tax and the extra
     data-quality datasets), when present;
  3. the bundled `cpad/data/` directory (Hospital + adult ship with the repo).
See data/README.md for how to obtain the datasets not bundled with the repo.

Each experiment script is standalone and reproduces one table/figure of the paper;
run it directly, e.g. `python experiments/hospital_detection.py`.
"""
from __future__ import annotations
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get("CPAD_DATA"),
    os.path.join(_HERE, "..", "..", "data_dq"),
    os.path.join(_HERE, "..", "data"),
]


def _resolve_data_root() -> str:
    for d in _CANDIDATES:
        if d and os.path.isdir(d):
            return os.path.abspath(d) + os.sep
    raise FileNotFoundError(
        "No dataset directory found. Set CPAD_DATA, or place data under cpad/data/ "
        "(see data/README.md)."
    )


DATA = _resolve_data_root()
