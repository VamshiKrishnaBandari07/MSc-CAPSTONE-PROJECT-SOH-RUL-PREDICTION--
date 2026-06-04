# Battery State-of-Health Prediction — Paper Reproduction

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports%202026-2ea44f)](https://doi.org/10.1038/s41598-026-39911-8)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Academic reproduction** of the hybrid deep learning model for lithium-ion **state-of-health (SOH)** estimation published in *Scientific Reports* (2026).

| | |
|:---|:---|
| **Author** | [Vamshi Krishna Bandari](https://github.com/VamshiKrishnaBandari07) |
| **Affiliation** | MSc Artificial Intelligence, University of Roehampton (UK) |
| **GitHub** | [`MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--`](https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--) |
| **Recommended rename** | `battery-soh-paper-reproduction` — see [`docs/GITHUB.md`](docs/GITHUB.md) |

> This repository contains **only** the published paper methodology (SOH). Joint RUL or custom MSc experiments are **not** on GitHub; they are stored locally under `local_archive/` (gitignored).

---

## Research motivation

Battery SOH quantifies remaining useful capacity relative to a fresh cell. Accurate SOH supports battery management systems, warranty analytics, and second-life deployment. The reference paper combines electrochemical curve features (ICA, DV, DC) with a **CNN–TCN–LSTM–attention** hybrid for robust SOH regression across chemistries and cycling protocols.

## Problem statement

Given voltage and current profiles from charge/discharge cycles, predict normalised **SOH ∈ [0, 1]** without hand-crafted capacity thresholds alone.

## Research paper information

| Field | Detail |
|:---|:---|
| **Title** | Hybrid deep learning approach for battery state-of-health prediction |
| **Authors** | Rahman et al. |
| **Journal** | *Scientific Reports* **16**, 9871 (2026) |
| **DOI** | [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8) |

## Methodology overview

1. Load NASA, Oxford, and CALCE cycling data.  
2. Extract **ICA** (dQ/dV), **DV** (dV/dQ), **DC** (dI/dV) on a **300-point** voltage grid (2.5–4.2 V).  
3. Savitzky–Golay smoothing (window 15, order 3).  
4. Train hybrid **CNN → TCN → LSTM → attention** with MSE.  
5. Evaluate with **stratified 5-fold cross-validation** (primary protocol).

Full mapping: [`docs/PAPER_METHODOLOGY.md`](docs/PAPER_METHODOLOGY.md)

## Dataset description

| Dataset | Format | Role |
|:---|:---|:---|
| NASA PCoE | `.mat` | Degradation cycling |
| Oxford Battery Degradation Dataset 1 | `.mat` | Characterisation cycles |
| CALCE (CS2 cells) | `.xlsx` | Additional protocols |

Raw data: **Git LFS** (~450 MB). See [`docs/DATA_AND_GIT.md`](docs/DATA_AND_GIT.md).

## Data preprocessing pipeline

- Per-cycle alignment to fixed voltage grid  
- ICA / DV / DC from interpolated Q(V), V(Q), I(V)  
- NaN sanitisation and channel normalisation  
- Implementation: `preprocess_paper.py`, `experiments/paper_preprocessing.py`

## Feature extraction pipeline

| Channel | Definition |
|:---|:---|
| ICA | dQ/dV |
| DV | dV/dQ |
| DC | dI/dV |

Grid: **300 points**, 2.5 V – 4.2 V.

## Model architecture

```mermaid
flowchart LR
  curves[Raw V I Q] --> feat[ICA DV DC]
  feat --> cnn[1D CNN]
  cnn --> tcn[TCN]
  tcn --> lstm[LSTM]
  lstm --> attn[Attention]
  attn --> soh[SOH sigmoid head]
```

Code: [`model_paper.py`](model_paper.py) (~**0.39M** trainable parameters).

## Training strategy

- Optimiser: Adam (lr 1e-3, weight decay 1e-5)  
- Loss: MSE on SOH  
- Max epochs: 200, early stopping patience 20  
- Gradient clipping: max norm 5  
- LR scheduler: ReduceLROnPlateau (×0.5)  
- Augmentation: ±10 mV voltage jitter + feature noise (train only)  
- Seed: **42**

## Evaluation strategy

| Protocol | Flag | Use |
|:---|:---|:---|
| **Stratified 5-fold CV** | default | **Primary** — paper comparison |
| Chronological 80/20 | `--chrono` | Fast supplementary / figure hold-out |

Metrics: SOH **RMSE**, **R²**, monotonicity violation rate (diagnostic).

## Reproduction workflow

```text
git lfs pull  →  run_paper_experiment.py  →  sanitize JSON  →  generate_figures.py  →  benchmark.py
```

## Installation guide

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Conda: `conda env create -f environment.yml`

## Quick start

```powershell
git lfs install
git clone git@github.com:VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--
git lfs pull
pip install -r requirements.txt
python run_paper_experiment.py --require-real --cpu
python scripts/sanitize_paper_report.py
python generate_figures.py
```

One command: `powershell -File scripts/run_paper_pipeline.ps1`

## Repository structure

```
├── data/                    # NASA, Oxford, CALCE (LFS)
├── experiments/             # loaders, CV, training, metrics
├── model_paper.py           # hybrid architecture
├── preprocess_paper.py      # feature pipeline
├── run_paper_experiment.py  # main experiment
├── paper_reproduction/      # wrapper entry
├── results/                 # JSON + fig01–fig04
├── tests/
├── scripts/
└── docs/                    # audit, reproducibility, results
```

Details: [`docs/FOLDER_STRUCTURE.md`](docs/FOLDER_STRUCTURE.md)

## Results table

**Protocol:** stratified 5-fold CV, real data, seed 42  
**Source:** `results/paper_experiment_report.json`

| Dataset | SOH RMSE (mean ± std) | SOH R² | Paper hybrid | Transformer |
|:---|:---:|:---:|:---:|:---:|
| NASA | 0.0385 ± 0.0048 | 0.915 | 0.021 | 0.038 |
| Oxford | **0.0215 ± 0.0050** | 0.951 | 0.021 | 0.038 |
| CALCE | 0.0673 ± 0.0101 | 0.950 | — | — |

### Reproducibility statement

**Methodology reproduced successfully; exact numerical replication was not fully achieved** on NASA (our RMSE ≈ Transformer baseline, not the paper hybrid 0.021). Oxford mean RMSE aligns with the published hybrid target within fold variance.

Figures: [`results/figures/`](results/figures/) — `fig01` trajectories, `fig02` scatter, `fig03` RMSE comparison, `fig04` training convergence.

## Discussion

- **Oxford** provides the closest match to the paper’s reported hybrid RMSE.  
- **NASA** gap likely reflects cell pooling, train/validation construction, or preprocessing differences vs the original study’s NASA-centric Table 4 setup.  
- **CALCE** extends evaluation beyond the paper’s primary table; interpret as supplementary.

## Limitations

- NASA RMSE does not reach 0.021.  
- Figure trajectories use a chronological hold-out; reported RMSE uses 5-fold CV.  
- Training on CPU is slow (~2–8 h all datasets).  
- No author-provided reference weights — models are retrained from seed.

## Future work

- Align NASA cell aggregation with paper supplementary details if released.  
- Publish fold checkpoints; LFS-aware CI.  
- Rename GitHub repository to `battery-soh-paper-reproduction`.

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

```bibtex
@misc{bandari2026sohrepro,
  author       = {Bandari, Vamshi Krishna},
  title        = {Battery SOH Paper Reproduction},
  year         = {2026},
  howpublished = {\url{https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--}}
}
```

## Acknowledgements

University of Roehampton; NASA, Oxford, and CALCE dataset providers; PyTorch community.

## License

MIT — see [LICENSE](LICENSE).

---

## Documentation index

| Document | Purpose |
|:---|:---|
| [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md) | Full file audit |
| [`docs/RESULTS_CONSISTENCY_REPORT.md`](docs/RESULTS_CONSISTENCY_REPORT.md) | Metrics vs figures |
| [`docs/REPRODUCIBILITY_CHECKLIST.md`](docs/REPRODUCIBILITY_CHECKLIST.md) | Step-by-step verification |
| [`docs/ACADEMIC_ASSESSMENT.md`](docs/ACADEMIC_ASSESSMENT.md) | Quality scores |
| [`docs/FINAL_FILE_MANIFEST.md`](docs/FINAL_FILE_MANIFEST.md) | GitHub vs local files |
| [`local_archive/LOCAL_ARCHIVE_CONTENTS.md`](local_archive/LOCAL_ARCHIVE_CONTENTS.md) | Excluded MSc work |
