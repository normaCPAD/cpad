# Datasets

Bundled with the repository (small, used by most experiments):

- `hospital_dirty.csv` / `hospital_clean.csv` — Hospital (HoloClean), the standard
  FD-violation data-cleaning benchmark; `hospital_*_wide.csv` and
  `hospital_errmask.npy` are the wide form and the ground-truth error mask.
- `adult.csv` — UCI Adult, used as a governed/ungoverned mix.
- `hospital_constraints.txt` — the expert denial constraints (NADEEF/HoloClean format).

Not bundled (larger; fetch separately, then place under this `data/` directory or point
`CPAD_DATA` at a directory that contains them):

- `tax/clean.csv`, `tax/dirty.csv` — the Tax dataset (~28 MB total).
- `beers/`, `flights/`, `movies_1/`, `rayyan/` — the extra data-quality datasets, each
  with `clean.csv` / `dirty.csv`.

These come from the standard error-detection benchmark collections (e.g. the
HoloClean / Raha / Baran repositories). The experiments that need them
(`composite_dc_tax.py`, `repair.py` for Tax, `rule_extraction.py`) will raise a clear
`FileNotFoundError` if a dataset is missing; the Hospital/Adult experiments run with the
bundled data alone.

Resolution order (see `experiments/_common.py`): `CPAD_DATA` env var → a sibling
`data_dq/` directory → this bundled `data/`.
