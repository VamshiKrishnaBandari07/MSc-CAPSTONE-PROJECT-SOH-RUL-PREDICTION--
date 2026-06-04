# Supervisor guide — paper reproduction only

**Author:** Vamshi Krishna Bandari · MSc Artificial Intelligence, University of Roehampton  
**Paper:** Rahman et al., *Scientific Reports* (2026) — [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

## What this repository contains

| Included | Excluded (local only) |
|:---|:---|
| Hybrid SOH model (CNN–TCN–LSTM–attention) | Joint SOH+RUL MSc model |
| NASA, Oxford, CALCE datasets (Git LFS) | `local_archive/` |
| 5-fold CV experiment + figures | Thesis `.docx` |

## Point-by-point verification

Run:

```powershell
python scripts/verify_repo.py
python -m pytest tests/ -v
```

| # | Check | Expected |
|:---:|:---|:---|
| 1 | Entry script | `run_paper_experiment.py` |
| 2 | Three datasets in JSON | NASA, Oxford, CALCE |
| 3 | Evaluation | `eval_protocol`: cv5 |
| 4 | Figures | `fig01`–`fig04` only |
| 5 | No MSc scripts | No `run_experiments.py`, `model.py` |
| 6 | Oxford RMSE | ≈ **0.0215** (paper target 0.021) |
| 7 | NASA RMSE | ≈ **0.0385** (not 0.021 — documented) |

## Results (committed)

| Dataset | SOH RMSE (5-fold CV) | vs paper 0.021 |
|:---|:---:|:---|
| Oxford | **0.0215 ± 0.0050** | Aligned |
| NASA | 0.0385 ± 0.0048 | Not replicated |
| CALCE | 0.0673 ± 0.0101 | Supplementary |

**Statement:** Methodology reproduced; NASA numerical target not fully matched.

## Clone and reproduce

```powershell
git lfs install
git clone https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--
git lfs pull
pip install -r requirements.txt
python run_paper_experiment.py --require-real --cpu
python generate_figures.py
```

Full pipeline: `powershell -File scripts/run_paper_pipeline.ps1`

## Code map

| Paper step | File |
|:---|:---|
| Features ICA/DV/DC | `experiments/paper_preprocessing.py` |
| Loaders | `experiments/nasa_loader.py`, `oxford_loader.py`, `calce_loader.py` |
| Model | `model_paper.py` |
| Training + CV | `experiments/trainer.py`, `experiments/cv.py` |
| Hyperparameters | `experiments/paper_config.py` |

See [`PAPER_METHODOLOGY.md`](PAPER_METHODOLOGY.md) and [`RESULTS.md`](RESULTS.md).
