# Naming Correction Report

## Spelling

| Location | Before | After (documented) |
|:---|:---|:---|
| Local Windows folder | `battery SOH predications` | Recommend `battery_soh_prediction` |
| Historical GitHub name | `PREDICATION` | `PREDICTION` |

## Repository branding

| Layer | Current | Recommended |
|:---|:---|:---|
| GitHub repo slug | `MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--` | `battery-soh-paper-reproduction` |
| README title | Hybrid Deep Learning for Battery SOH Prediction | ✅ Aligned with paper |
| Python package | `experiments/` | ✅ Stable (no rename required) |

## Figure naming (standardised)

| File | Purpose |
|:---|:---|
| `fig01_soh_trajectories` | Validation SOH vs cycle |
| `fig02_soh_scatter` | Predicted vs true SOH |
| `fig03_soh_rmse_comparison` | RMSE vs published baselines |
| `fig04_training_convergence` | Validation RMSE per epoch |

Removed: `fig02_rul_*`, `fig05`–`fig07`, legacy `fig03`/`fig04` scatter/RMSE names.

## JSON / result files

| File | Naming |
|:---|:---|
| Primary report | `paper_experiment_report.json` |
| Benchmark | `computational_benchmark.json` |
| Removed | `experiment_comparison_report.json`, `nasa_real_experiment_report.json` |

## Consistency actions taken

- README uses paper-only terminology (no RUL in scope).
- `run_paper_experiment.py` payload field `experiment`: `paper_reproduction`.
- GitHub clone URLs documented in `docs/GITHUB.md` with rename guidance.
