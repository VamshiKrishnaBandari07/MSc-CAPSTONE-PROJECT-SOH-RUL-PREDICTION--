# Battery SOH Paper Reproduction

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports%202026-2ea44f)](https://doi.org/10.1038/s41598-026-39911-8)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![CI](https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--/actions/workflows/ci.yml/badge.svg)](https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Academic reproduction of **Rahman et al. (2026)** — hybrid deep learning for lithium-ion **state-of-health (SOH)** on **NASA**, **Oxford**, and **CALCE**.

| | |
|:---|:---|
| **Author** | [Vamshi Krishna Bandari](https://github.com/VamshiKrishnaBandari07) |
| **Institution** | University of Roehampton — MSc Artificial Intelligence |
| **Repository** | https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION-- |

---

## Results (paper experiment — 5-fold CV, seed 42)

| Dataset | SOH RMSE (mean ± std) | SOH R² |
|:---|:---:|:---:|
| **Oxford** | **0.0215 ± 0.0050** | 0.951 |
| NASA | 0.0385 ± 0.0048 | 0.915 |
| CALCE | 0.0673 ± 0.0101 | 0.950 |

Published hybrid target: **0.021** (Oxford aligns; NASA does not — see note below).

Artifacts: `results/paper_experiment_report.json` · Figures: `results/figures/fig01`–`fig04`

---

## Experimental workflow

```mermaid
flowchart TB
  D1[NASA] --> F[ICA / DV / DC]
  D2[Oxford] --> F
  D3[CALCE] --> F
  F --> G[300-pt voltage grid]
  G --> M[CNN - TCN - LSTM - Attention]
  M --> CV[5-fold stratified CV]
  CV --> R[RMSE and R2]
  R --> OUT[JSON + figures]
```

---

## Quick start

```powershell
git lfs install
git clone https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--
git lfs pull
pip install -r requirements.txt
python scripts/verify_repo.py
```

**Run paper experiment (all 3 datasets):**

```powershell
python run_paper_experiment.py --require-real --cpu
python generate_figures.py
```

Or: `powershell -File scripts/run_paper_pipeline.ps1` (~2–8 h CPU)

---

## Repository structure

```
run_paper_experiment.py    # main experiment (NASA + Oxford + CALCE)
model_paper.py             # hybrid architecture
preprocess_paper.py        # ICA / DV / DC pipeline
generate_figures.py        # fig01–fig04
experiments/               # loaders, CV, training, metrics
data/                      # datasets (Git LFS)
results/                   # paper_experiment_report.json + figures
tests/
docs/                      # methodology, results, supervisor guide
```

---

## Reproducibility note

**Methodology reproduced successfully** (features, model, 5-fold CV, hyperparameters). **Exact NASA RMSE 0.021 was not achieved** with this public-data pipeline; Oxford matches the published hybrid metric.

For examiner review: [`docs/SUPERVISOR_GUIDE.md`](docs/SUPERVISOR_GUIDE.md)

| Document | Content |
|:---|:---|
| [`docs/PAPER_METHODOLOGY.md`](docs/PAPER_METHODOLOGY.md) | Paper ↔ code |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Metrics table |

---

## Reference & citation

Rahman et al., *Scientific Reports* **16**, 9871 (2026). https://doi.org/10.1038/s41598-026-39911-8

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

MIT License — see [LICENSE](LICENSE).
