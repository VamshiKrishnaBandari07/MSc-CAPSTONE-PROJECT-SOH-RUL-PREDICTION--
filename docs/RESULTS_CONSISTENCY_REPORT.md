# Results Consistency Report

**Date:** June 2026  
**Canonical metrics source:** `results/paper_experiment_report.json`  
**Figures:** `results/figures/fig01`–`fig04`  
**Benchmark:** `results/computational_benchmark.json`

## Consistency matrix

| Artifact | NASA RMSE | Oxford RMSE | CALCE RMSE | Protocol |
|:---|:---:|:---:|:---:|:---|
| `paper_experiment_report.json` | 0.0385 ± 0.0048 | 0.0215 ± 0.0050 | 0.0673 ± 0.0101 | 5-fold CV |
| `docs/RESULTS.md` | ✅ Match | ✅ Match | ✅ Match | ✅ |
| `README.md` § Results | ✅ Match | ✅ Match | ✅ Match | ✅ |
| `fig03_soh_rmse_comparison` | ✅ Bars from JSON | ✅ | ✅ | CV means |
| `computational_benchmark.json` | N/A | N/A | N/A | Params **0.385 M** (paper ~0.35 M) |

## Stale metadata — resolved

| Issue | Status |
|:---|:---|
| `experiment_b_msc` in JSON | ✅ Removed via `scripts/sanitize_paper_report.py` |
| `next_step` Phase 2 | ✅ Removed |
| `msc_proposed` in benchmark | ✅ Removed |
| MSc figures fig02_rul, fig05–07 | ✅ Removed from `results/figures/` |

## Honest reporting statement

> **Methodology reproduced successfully; exact numerical replication was not fully achieved** on NASA (our 0.0385 vs paper hybrid 0.021). Oxford mean RMSE aligns with the published target within fold variance. NASA RMSE is close to the published Transformer baseline (0.038).

## Figure vs metric protocol

| Figure | Data source | Protocol note |
|:---|:---|:---|
| fig01, fig02 | Checkpoints + 80/20 hold-out | Visualisation only; see `eval_note` in inference |
| fig03 | JSON CV means | Primary comparison figure |
| fig04 | Best-fold training history | Illustrative convergence |

## Regeneration commands

```powershell
python run_paper_experiment.py --require-real --cpu
python scripts/sanitize_paper_report.py
python generate_figures.py
python benchmark.py
python scripts/sync_results_docs.py
```
