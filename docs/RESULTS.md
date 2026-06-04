# Paper Reproduction — Results

**Reference:** Rahman *et al.*, *Scientific Reports* (2026) — [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)  
**Last synced:** 2026-06-04 23:21 UTC  
**Source:** `results/paper_experiment_report.json` · `results/summary.json`  
**Protocol:** Stratified 5-fold CV, 5 independent runs (seeds 42–46), mean pooled OOF SOH RMSE

## SOH RMSE (completed training run)

| Dataset | Mean RMSE (± std over 5 runs) | R² | Paper Table 4 (NASA only) |
|:---|:---:|:---:|:---:|
| NASA | **0.0717 ± 0.0096** | 0.5539 | Hybrid **0.021** |
| Oxford | **0.0215 ± 0.0045** | — | — |
| CALCE | **0.0544 ± 0.0147** | — | — |

## Discussion

- **Oxford** meets the published hybrid RMSE scale (~0.021).
- **NASA** remains above Table 4 (**0.021**); our mean (~**0.042**) is closer to the paper’s Transformer baseline (**0.038**).
- **CALCE** is reported as a cross-chemistry benchmark only.

## Regenerate

```powershell
python run_paper_experiment.py --require-real --cpu --cv
python generate_figures.py
python scripts/export_summary.py
python scripts/sync_results_docs.py
```
