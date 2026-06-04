# Experimental Results Summary

**Author:** Vamshi Krishna Bandari  
**Last verified run:** 2026-06-04 00:26 UTC — auto-synced from JSON  
**Report files:** `results/experiment_comparison_report.json`, `results/paper_experiment_report.json`  
**Figures:** `results/figures/fig01`–`fig07` (PNG + PDF)

> **Evaluation:** Experiment A uses **cv5** (default paper protocol). See `docs/PAPER_METHODOLOGY.md`.

---

## Experiment A — Paper reproduction (SOH RMSE)

| Dataset | Our RMSE | Our R² | Paper hybrid | Transformer | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| **NASA** | **0.0385 ± 0.0048** | 0.9152 | 0.021 | 0.038 | (stratified_5fold_cv) |
| **Oxford** | **0.0215 ± 0.0050** | 0.9512 | 0.021 | 0.038 | (stratified_5fold_cv) |
| **CALCE** | **0.0673 ± 0.0101** | 0.9497 | 0.021 | 0.038 | (stratified_5fold_cv) |

---

## Experiment B — MSc extension (joint SOH + RUL)

| Dataset | SOH RMSE | SOH R² | RUL RMSE (cycles) | Mono. violation |
|:---|:---:|:---:|:---:|:---:|
| NASA | 0.1116 | -0.9525 | 44.3 | 0.535 |
| Oxford | 0.0409 | 0.6409 | 14.2 | 0.447 |
| CALCE | 0.2297 | -0.1203 | 1.5 | 0.494 |

---

## Experiment C — Ablation (physics monotonicity loss)

| Dataset | SOH RMSE (no physics) | SOH RMSE (with physics) |
|:---|:---:|:---:|
| NASA | 0.0781 | **0.1116** |
| Oxford | 0.0342 | **0.0409** |
| CALCE | 0.3549 | **0.2297** |

---

## Regenerate results

```powershell
python run_paper_experiment.py --require-real --cpu
python run_experiments.py --msc-only --require-real --cpu
python generate_figures.py
python scripts/sync_results_docs.py
```
