# Results — Paper reproduction

**Paper:** Rahman et al., *Scientific Reports* (2026) — [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)  
**Protocol:** Stratified 5-fold CV (`cv5`), seed 42, real data  
**Source:** `results/paper_experiment_report.json`

## SOH RMSE

| Dataset | RMSE (mean ± std) | R² | Paper hybrid | Transformer |
|:---|:---:|:---:|:---:|:---:|
| **Oxford** | **0.0215 ± 0.0050** | 0.951 | 0.021 | 0.038 |
| NASA | 0.0385 ± 0.0048 | 0.915 | 0.021 | 0.038 |
| CALCE | 0.0673 ± 0.0101 | 0.950 | — | — |

## Interpretation

- **Oxford** matches the published hybrid target within cross-validation variance.  
- **NASA** did not reach 0.021; error is close to the published Transformer baseline (0.038).  
- **CALCE** is an additional public benchmark, not the paper’s primary Table 4 focus.

## Figures

| File | Content |
|:---|:---|
| `fig01_soh_trajectories` | Validation SOH vs cycle |
| `fig02_soh_scatter` | Predicted vs true SOH |
| `fig03_soh_rmse_comparison` | RMSE vs published baselines |
| `fig04_training_convergence` | Validation RMSE per epoch |

## Regenerate

```powershell
python run_paper_experiment.py --require-real --cpu
python scripts/sanitize_paper_report.py
python generate_figures.py
```
