# Reproducing the paper's experiments

Each script is **standalone** and reproduces one table or figure of the paper. Run any
of them directly from the repository root:

```bash
pip install -e ".[baselines,gated]"     # engine + pyod baselines + torch (gated variant)
python experiments/cpad_categorical.py
```

Datasets are resolved by `experiments/_common.py` (`CPAD_DATA` env var, else a sibling
`data_dq/`, else the bundled `cpad/data/`). Hospital and Adult ship with the repo; see
[`../data/README.md`](../data/README.md) for Tax and the extra data-quality datasets.

| Script | Paper element | What it shows | Extra deps |
|---|---|---|---|
| `cpad_categorical.py` | Table (Hospital) | CPAD-categorical detection of FD violations vs marginal rarity / IsolationForest | — |
| `hospital_detection.py` | Table (Hospital) | State-of-the-art comparison (expert DCs, discovered DCs, HoloDetect-style, IForest) | — |
| `regularization.py` | Table (ablation) | Differentiable CPAD with learned gates + L1 sparsity (the role of regularization) | torch |
| `rule_extraction.py` | Table (rules) | Native gated rule discovery across the data-quality datasets | torch |
| `rule_transfer.py` | Table (transfer) | Expert DCs vs CPAD-discovered FDs vs their union; rule overlap | — |
| `value_swap.py` | Table (corrupter) | The value-swap test: valid-but-inconsistent errors vs marginal typos | — |
| `generic_detectors.py` | Table (generic) | Generic outlier detectors (pyod) on Hospital, tuple-level | pyod |
| `hybrid.py` | Table (hybrid) | Union of CPAD conditional violation + type-aware outlier (Tax and Hospital) | — |
| `contamination.py` | Robustness section | Degradation under a fraction eps of contaminated training rows; robust variants | — |
| `unsupervised_tau.py` | Unsupervised selection | Self-supervised tau selection (inject value-swaps) matches the oracle | — |
| `repair.py` | Appendix (repair) | Constraint-guided repair: precision/recall and error-rate reduction | cpad engine |
| `composite_dc_tax.py` | Appendix (Tax schema) | Native composite (multi-LHS) denial-constraint discovery on Tax | torch |
| `numerical_constraints.py` | Appendix (numerical) | Linear + monotone constraints, contamination sweep; no FP on rare-but-valid | cpad engine |
| `label_budget.py` | Appendix (label budget) | CPAD (0 labels) vs HoloDetect-style at increasing label budgets (Hospital) | — |
| `broad_benchmarks.py` | Appendix (broad benchmarks) | Tuple-level detection on real datasets (Beers/Flights/Rayyan/Hospital) vs IForest | cpad engine |
| `generality_valueswap.py` | Appendix (controlled generality) | Value-swap DC violations injected into FD-governed columns of clean schemas | cpad engine |
| `generic_detectors.py` | Appendix (deep detectors) | Generic + deep tabular detectors (DeepSVDD/autoencoder/VAE) on Hospital vs CPAD | pyod, cpad engine |
| `significance.py` | Appendix (significance) | 10-seed mean+/-std and Wilcoxon test, CPAD vs IForest on Hospital | cpad engine |
| `scalability.py` | Appendix (complexity) | Fit+score time and memory vs n and d, confirming O(d^2 n) | cpad engine |
| `ablations.py` | Appendix (ablations) | Gated model: no-corruption, no-L1, and lambda sweep | torch, cpad engine |

Numbers vary slightly with the random seed; the reported conclusions are stable.
Verified locally: `cpad_categorical.py` (row-level AUROC 0.95 vs IForest 0.53),
`repair.py` (Hospital error rate -90% at precision ~1.0), `numerical_constraints.py`
(AUROC >= 0.94 up to 30% contamination) and `broad_benchmarks.py` (Hospital 0.955 vs 0.544).
