# Paper Reproduction — Results

**Reference:** Rahman et al., *Scientific Reports* (2026) — [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)  
**Source:** `results/paper_experiment_report.json`  
**Protocol:** stratified 5-fold CV (`cv5`)

## SOH RMSE (stratified 5-fold CV)

| Dataset | Our RMSE | Our R² | Paper hybrid | Transformer |
|:---|:---:|:---:|:---:|:---:|
| NASA | **0.0385 ± 0.0048** | 0.915 | 0.021 | 0.038 |
| Oxford | **0.0215 ± 0.0050** | 0.951 | 0.021 | 0.038 |
| CALCE | **0.0673 ± 0.0101** | 0.950 | 0.021 | 0.038 |

Oxford mean RMSE is within one standard deviation of the published **0.021** target. NASA and CALCE differ from the paper’s pooled NASA-centric table; see `docs/AUDIT_REPORT.md` for protocol notes.

## Regenerate

```powershell
python run_paper_experiment.py --require-real --cpu
python generate_figures.py
python scripts/sync_results_docs.py
```
