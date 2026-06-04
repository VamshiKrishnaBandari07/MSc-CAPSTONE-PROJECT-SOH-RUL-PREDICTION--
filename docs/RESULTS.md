# Paper Reproduction — Results

**Reference:** Rahman *et al.*, *Scientific Reports* (2026) — [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)  
**Protocol:** Stratified 5-fold CV, five independent runs (seeds 42–46), mean pooled OOF SOH RMSE  
**Source:** `results/paper_experiment_report.json` · `results/summary.json`

## SOH RMSE (completed training run)

| Dataset | Mean RMSE (± std over 5 runs) | Paper Table 4 (NASA only) |
|:---|:---:|:---:|
| NASA PCoE | **0.0417 ± 0.0023** | Hybrid **0.021** |
| Oxford | **0.0215 ± 0.0045** | — (matches paper magnitude) |
| CALCE | **0.0544 ± 0.0147** | — |

### Per independent run (pooled OOF RMSE)

| Run | NASA | Oxford | CALCE |
|:---:|:---:|:---:|:---:|
| 1 | 0.0404 | 0.0268 | 0.0502 |
| 2 | 0.0392 | **0.0154** | 0.0454 |
| 3 | 0.0428 | 0.0183 | 0.0463 |
| 4 | 0.0407 | 0.0266 | 0.0835 |
| 5 | 0.0456 | 0.0205 | 0.0464 |

## Discussion

- **Oxford** meets the published hybrid RMSE scale (~0.021).
- **NASA** remains above Table 4 (**0.021**); our mean (~**0.042**) is closer to the paper’s Transformer baseline (**0.038**). Likely factors: cycle-level CV across cells, NASA Q-profile construction, and early stopping on non-finite validation epochs during training.
- **CALCE** is reported as a cross-chemistry benchmark only.

## Regenerate

```powershell
python run_paper_experiment.py --require-real --cpu --cv
python generate_figures.py
python scripts/sync_results_docs.py
```
