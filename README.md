# Battery SOH Paper Reproduction

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports%202026-2ea44f)](https://doi.org/10.1038/s41598-026-39911-8)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproduction of the hybrid deep learning model for lithium-ion **state-of-health (SOH)** estimation (Rahman et al., *Scientific Reports*, 2026).

| | |
|:---|:---|
| **Author** | [Vamshi Krishna Bandari](https://github.com/VamshiKrishnaBandari07) |
| **Programme** | MSc Artificial Intelligence, University of Roehampton (UK) |
| **Repository** | https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION-- |

---

## Summary for examiners

| Item | Status |
|:---|:---|
| Paper methodology (ICA/DV/DC, 300-pt grid, hybrid model, 5-fold CV) | Implemented |
| Oxford SOH RMSE vs published **0.021** | **0.0215 ± 0.0050** (aligned) |
| NASA SOH RMSE vs published **0.021** | **0.0385 ± 0.0048** (not matched; ~Transformer baseline) |
| Figures & JSON in repo | `results/figures/fig01`–`fig04`, `paper_experiment_report.json` |

> **Methodology reproduced successfully; exact numerical replication on NASA was not fully achieved.** See [`docs/SUPERVISOR_GUIDE.md`](docs/SUPERVISOR_GUIDE.md).

---

## Reference

**Rahman et al.** Hybrid deep learning approach for battery state-of-health prediction. *Scientific Reports* **16**, 9871 (2026).  
DOI: [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

## Quick start

```powershell
git lfs install
git clone https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--
git lfs pull
pip install -r requirements.txt
python scripts/verify_setup.py
```

**Reproduce results (CPU, ~2–8 h all datasets):**

```powershell
python run_paper_experiment.py --require-real --cpu
python scripts/sanitize_paper_report.py
python generate_figures.py
```

**NASA only (faster):** `python run_paper_experiment.py --require-real --cpu --dataset NASA`

## Results (stratified 5-fold CV)

| Dataset | SOH RMSE (mean ± std) | SOH R² |
|:---|:---:|:---:|
| Oxford | **0.0215 ± 0.0050** | 0.951 |
| NASA | 0.0385 ± 0.0048 | 0.915 |
| CALCE | 0.0673 ± 0.0101 | 0.950 |

Source: `results/paper_experiment_report.json` · Plots: `results/figures/`

## Methodology

1. Datasets: NASA, Oxford, CALCE (real `.mat` / `.xlsx`, Git LFS)  
2. Features: ICA (dQ/dV), DV (dV/dQ), DC (dI/dV) on 300-point grid (2.5–4.2 V)  
3. Model: CNN → TCN → LSTM → attention (~0.39M parameters)  
4. Training: MSE, Adam, augmentation, early stopping (see `experiments/paper_config.py`)  
5. Evaluation: **stratified 5-fold CV** (default), seed **42**

Details: [`docs/PAPER_METHODOLOGY.md`](docs/PAPER_METHODOLOGY.md)

## Repository layout

```
├── data/                      # NASA, Oxford, CALCE (LFS)
├── experiments/               # loaders, CV, training, metrics
├── model_paper.py             # hybrid architecture
├── preprocess_paper.py        # feature pipeline
├── run_paper_experiment.py    # main experiment
├── generate_figures.py
├── results/                   # JSON + fig01–fig04
├── tests/
├── scripts/
└── docs/                      # methodology, results, supervisor guide
```

## Documentation

| Document | Purpose |
|:---|:---|
| [`docs/SUPERVISOR_GUIDE.md`](docs/SUPERVISOR_GUIDE.md) | Examiner verification (start here) |
| [`docs/PAPER_METHODOLOGY.md`](docs/PAPER_METHODOLOGY.md) | Paper ↔ code mapping |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Results table |
| [`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md) | Full checklist |
| [`docs/DATA_AND_GIT.md`](docs/DATA_AND_GIT.md) | Data & LFS |

## Citation

```bibtex
@article{rahman2026hybrid,
  title   = {Hybrid deep learning approach for battery state-of-health prediction},
  journal = {Scientific Reports},
  volume  = {16},
  pages   = {9871},
  year    = {2026},
  doi     = {10.1038/s41598-026-39911-8}
}
```

## License

MIT — see [LICENSE](LICENSE).
