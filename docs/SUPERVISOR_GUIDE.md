# Supervisor guide — paper reproduction only

**Student:** Vamshi Krishna Bandari · MSc AI, University of Roehampton  
**Paper:** [Scientific Reports 16, 9871 (2026)](https://www.nature.com/articles/s41598-026-39911-8) · DOI [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

## Scope

GitHub holds **only** Rahman et al. (2026) **SOH** reproduction. MSc joint SOH+RUL work is in `local_archive/` (not pushed).

## Point-by-point checklist

1. Three datasets loaded — NASA, Oxford, CALCE  
2. SOH = Q_k / Q_BoL per cell  
3. ICA, DV, DC + SG(15,3) + 300-point grid  
4. Hybrid CNN–TCN–LSTM–attention  
5. Stratified 5-fold CV, seed 42  
6. Report JSON lists all three datasets  
7. Figures fig01–fig04 present  
8. No MSc scripts in repo root  

Run: `python scripts/verify_repo.py`

## Results vs Table 4 (NASA PCoE)

| Metric | Paper | Reproduction |
|:---|:---:|:---:|
| RMSE | 0.021 | 0.0385 ± 0.0048 |
| R² | 0.983 | 0.915 |

Oxford RMSE **0.0215 ± 0.0050** matches paper magnitude.

## Reproduce

```powershell
git lfs pull
pip install -r requirements.txt
python run_paper_experiment.py --require-real --cpu
python generate_figures.py
```
