# Hybrid Deep Learning for Battery State-of-Health Prediction

**Paper reproduction — Rahman et al., *Scientific Reports* (2026)**

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports%202026-green)](https://doi.org/10.1038/s41598-026-39911-8)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9+-3776ab)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-pytest-blue)](tests/)
[![CI](https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--/actions/workflows/ci.yml/badge.svg)](https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--/actions/workflows/ci.yml)

**Author:** [Vamshi Krishna Bandari](https://github.com/VamshiKrishnaBandari07) · MSc Artificial Intelligence, University of Roehampton (UK)

This repository reproduces the **published hybrid CNN–TCN–LSTM–Attention model** for lithium-ion **SOH** estimation on **NASA**, **Oxford**, and **CALCE** data. It does **not** include MSc extension experiments (joint RUL, ablation) on GitHub — those live in `local_archive/` on your machine only.

```powershell
git lfs install
git clone git@github.com:VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--
git lfs pull
pip install -r requirements.txt
python run_paper_experiment.py --require-real --cpu
```

---

## 1. Research motivation

Accurate **state-of-health (SOH)** prediction supports battery management, warranty analytics, and second-life grading. Deep models that fuse electrochemical curve features with temporal encoders can outperform hand-crafted capacity thresholds alone. This project provides a **reproducible implementation** of a recent *Scientific Reports* hybrid architecture for examiner and community verification.

## 2. Paper information

| Field | Value |
|:---|:---|
| **Title** | Hybrid deep learning approach for battery state-of-health prediction |
| **Authors** | Rahman et al. |
| **Journal** | *Scientific Reports* **16**, 9871 (2026) |
| **DOI** | [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8) |

## 3. Problem statement

Estimate normalised **SOH** from charge/discharge voltage–current curves using **ICA** (dQ/dV), **DV** (dV/dQ), and **DC** (dI/dV) features on a fixed voltage grid, without manual feature selection per chemistry.

## 4. Dataset description

| Dataset | Source | Format | Role |
|:---|:---|:---|:---|
| **NASA** | NASA Prognostics Center of Excellence | `.mat` | Primary degradation cycles |
| **Oxford** | Oxford Battery Degradation Dataset | `.mat` | High-quality cycling lab data |
| **CALCE** | University of Maryland CALCE | `.xlsx` | Additional chemistry / protocol diversity |

Raw files are stored under `data/` (Git LFS). See [`docs/DATA_AND_GIT.md`](docs/DATA_AND_GIT.md).

## 5. Methodology overview

1. Load raw cycles per battery cell.  
2. Extract ICA, DV, DC on a **300-point** voltage grid (2.5–4.2 V).  
3. Savitzky–Golay smoothing (window 15, order 3).  
4. Train **CNN → TCN → LSTM → attention** with MSE and sigmoid SOH output.  
5. Evaluate with **stratified 5-fold cross-validation** (paper protocol).

Details: [`docs/PAPER_METHODOLOGY.md`](docs/PAPER_METHODOLOGY.md)

## 6. Architecture diagram (conceptual)

```mermaid
flowchart LR
  A[Raw V,I,Q curves] --> B[ICA / DV / DC features]
  B --> C[1D CNN]
  C --> D[TCN dilated blocks]
  D --> E[LSTM]
  E --> F[Attention]
  F --> G[SOH head sigmoid]
```

Implementation: [`model_paper.py`](model_paper.py) (~0.39M trainable parameters).

## 7. Experimental workflow

```
download_data.py --all
        ↓
run_paper_experiment.py  (5-fold CV, seed 42)
        ↓
paper_experiment_report.json
        ↓
generate_figures.py  →  fig01–fig04
```

## 8. Installation guide

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Or Conda:

```powershell
conda env create -f environment.yml
conda activate paper-soh-reproduction
```

## 9. Environment setup

| Requirement | Version |
|:---|:---|
| Python | 3.9 – 3.11 |
| PyTorch | ≥ 2.0 |
| NumPy, SciPy, pandas, matplotlib, openpyxl | see `requirements.txt` |

Verify:

```powershell
python scripts/verify_setup.py
```

## 10. Data preparation

```powershell
git lfs pull
# or
python download_data.py --all
```

## 11. Training instructions

**Full paper protocol (CPU, all datasets, ~2–8 h):**

```powershell
python run_paper_experiment.py --require-real --cpu
```

**Single dataset / fast split:**

```powershell
python run_paper_experiment.py --require-real --cpu --dataset Oxford
python run_paper_experiment.py --require-real --cpu --dataset NASA --chrono
```

Wrapper entry: `python paper_reproduction/run.py --require-real --cpu`

## 12. Evaluation instructions

- Primary metrics: **SOH RMSE**, **R²**, fold mean ± std in `results/paper_experiment_report.json`.  
- Default evaluation: **`--cv`** stratified 5-fold.  
- Supplementary: **`--chrono`** chronological 80/20.

## 13. Reproduction steps (checklist)

1. `git lfs pull`  
2. `pip install -r requirements.txt`  
3. `python run_paper_experiment.py --require-real --cpu`  
4. `python generate_figures.py`  
5. `python -m pytest tests/ -v`  

Full checklist: [`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md)

**One command:**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_paper_pipeline.ps1
```

## 14. Folder structure

```
├── data/                  # NASA, Oxford, CALCE (LFS)
├── experiments/           # loaders, CV, training, metrics
├── model_paper.py         # hybrid architecture
├── preprocess_paper.py    # paper feature pipeline
├── run_paper_experiment.py
├── paper_reproduction/    # thin wrapper
├── models/                # documentation pointer
├── results/               # JSON + figures
├── tests/
├── scripts/
└── docs/                  # methodology, audit, results
```

See [`docs/FOLDER_STRUCTURE.md`](docs/FOLDER_STRUCTURE.md).

## 15. Results

Committed 5-fold CV run (`results/paper_experiment_report.json`):

| Dataset | SOH RMSE (mean ± std) | SOH R² |
|:---|:---:|:---:|
| NASA | **0.0385 ± 0.0048** | 0.915 |
| Oxford | **0.0215 ± 0.0050** | 0.951 |
| CALCE | **0.0673 ± 0.0101** | 0.950 |

Oxford aligns with the paper’s **0.021** NASA-centric target within fold variance. See [`docs/RESULTS.md`](docs/RESULTS.md).

## 16. Performance metrics

| Metric | Definition |
|:---|:---|
| **RMSE** | Root mean squared error on normalised SOH |
| **R²** | Coefficient of determination |
| **Mono. violation rate** | Diagnostic for non-increasing SOH predictions |

Published baselines (Transformer vs paper hybrid) are referenced in `experiments/config.py` → `PAPER_REFERENCE`.

## 17. Limitations

- NASA pooled preprocessing may differ slightly from the original study → RMSE gap vs 0.021.  
- Training on CPU is slow; GPU recommended for replication studies.  
- No RUL head in this repository (MSc work archived locally).  
- CALCE protocols vary; results are benchmark supplementary, not always in paper tables.

## 18. Future work

- Publish pre-trained checkpoints per fold.  
- LFS-aware CI and Docker image with CUDA.  
- Harmonise NASA cell pooling with paper supplementary code if released.

## 19. References

1. Rahman, M. M. et al. Hybrid deep learning approach for battery state-of-health prediction. *Sci. Rep.* **16**, 9871 (2026). https://doi.org/10.1038/s41598-026-39911-8  
2. NASA Prognostics Data Repository — battery datasets.  
3. Oxford Battery Degradation Dataset 1.  
4. CALCE battery research group datasets.

## 20. Citation

If you use this reproduction code, please cite the **original paper** and optionally this repository:

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

```bibtex
@misc{bandari2026sohrepo,
  author       = {Vamshi Krishna Bandari},
  title        = {SOH Paper Reproduction --- MSc Capstone},
  year         = {2026},
  howpublished = {\url{https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--}}
}
```

## 21. Acknowledgements

University of Roehampton MSc Artificial Intelligence programme; dataset providers (NASA, Oxford, CALCE); PyTorch community.

## 22. Audit and quality

- [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md) — full review, score **87/100**, removal log  
- [`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md)  
- [`docs/GITHUB.md`](docs/GITHUB.md) — contribution and LFS notes  

**License:** MIT — see [`LICENSE`](LICENSE).

---

*Maintained for dissertation supervision, reproducibility review, and portfolio showcase. MSc extension code: `local_archive/` (local only, not on GitHub).*
