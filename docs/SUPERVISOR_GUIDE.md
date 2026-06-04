# Supervisor verification guide

**Student:** Vamshi Krishna Bandari · MSc AI, University of Roehampton  
**Paper:** Rahman et al., *Scientific Reports* (2026) — [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

## What this repository is

A **paper-only** reproduction of the published hybrid CNN–TCN–LSTM–attention model for **SOH** prediction. No joint RUL or custom MSc experiments are on GitHub.

## Verify in 10 minutes

```powershell
git lfs pull
pip install -r requirements.txt
python scripts/verify_setup.py
python -m pytest tests/ -v --tb=short
```

Committed results are in `results/paper_experiment_report.json` and `results/figures/fig01`–`fig04`.

## Primary results (5-fold stratified CV, real data, seed 42)

| Dataset | SOH RMSE (mean ± std) | vs paper hybrid 0.021 |
|:---|:---:|:---|
| **Oxford** | **0.0215 ± 0.0050** | Aligned |
| **NASA** | 0.0385 ± 0.0048 | Not replicated (near Transformer 0.038) |
| **CALCE** | 0.0673 ± 0.0101 | Supplementary benchmark |

## Reproducibility statement

**Methodology reproduced successfully** (features, grid, model, CV, training hyperparameters). **Exact NASA RMSE 0.021 was not replicated** with public data and this implementation; Oxford supports validity of the pipeline.

## Full reproduction (CPU, several hours)

```powershell
python run_paper_experiment.py --require-real --cpu
python scripts/sanitize_paper_report.py
python generate_figures.py
```

## Further reading

- [`PAPER_METHODOLOGY.md`](PAPER_METHODOLOGY.md) — code ↔ paper mapping  
- [`RESULTS.md`](RESULTS.md) — metrics table  
- [`REPRODUCIBILITY_CHECKLIST.md`](REPRODUCIBILITY_CHECKLIST.md) — step list  
