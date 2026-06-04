# MSc Capstone — Project Overview

**Author:** Vamshi Krishna Bandari  
**Programme:** MSc Computing / Data Science / AI (UK)  
**Institution:** University of Roehampton  
**Artefact:** Reproducible deep learning pipeline for battery health prognostics  

---

## Problem statement

Electric vehicle battery packs degrade with cycling. Accurate **State of Health (SOH)** and **Remaining Useful Life (RUL)** estimation from routine charge–discharge data supports safe battery management, warranty decisions, and fleet optimisation without destructive testing.

---

## Research approach (two phases — mandatory order)

| Phase | Type | Contribution |
|:---:|:---|:---|
| **1** | **Experiment A** | Faithful reproduction of Rahman et al. (*Scientific Reports*, 2026): hybrid CNN–TCN–LSTM–Attention, ICA+DV+DC features, SOH-only prediction |
| **2** | **Experiment B** | **Original MSc extension:** joint SOH+RUL prediction with physics-informed monotonicity regularisation |
| **2** | **Experiment C** | Ablation study — Experiment B without monotonicity penalty |

Phase 1 establishes credibility against a peer-reviewed baseline. Phase 2 demonstrates novel research contribution beyond the published paper.

---

## Datasets (real public data)

| Dataset | Cycles | Source |
|:---|:---:|:---|
| NASA PCoE (B0005–B0018) | 636 | [NASA Open Data](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets) |
| Oxford Battery Degradation 1 | 519 | [Oxford ORA](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) |
| CALCE CS2 (33, 35, 36) | 2,703 | [CALCE UMD](https://calce.umd.edu/battery-data) |

Raw files (~500 MB) are downloaded locally via `python download_data.py --all` — not stored in git.

---

## Key results (verified — stratified 5-fold CV, Experiment A)

See [`RESULTS.md`](RESULTS.md) for full tables.

| Experiment | Oxford SOH RMSE | NASA SOH RMSE | vs paper 0.021 | vs Transformer 0.038 |
|:---|:---:|:---:|:---:|:---:|
| **A — Paper repro** | **0.0215 ± 0.0050** | 0.0385 ± 0.0048 | **matches (Oxford)** | **beats (Oxford −43%)** |
| **B — MSc joint** | 0.041 (80/20) | 0.112 (80/20) | — | — |

Experiment A uses **stratified 5-fold CV** (paper protocol). Experiments B and C use chronological 80/20 split (MSc default).

---

## Reproducibility checklist

```powershell
pip install -r requirements.txt
python download_data.py --all
python scripts/verify_setup.py
python run_paper_experiment.py --require-real --cpu
python run_experiments.py --msc-only --require-real --cpu
python generate_figures.py
python scripts/sync_results_docs.py
python -m pytest tests/ -v
```

---

## Repository map

| Path | Purpose |
|:---|:---|
| `run_paper_experiment.py` | **Phase 1 entry point** |
| `run_experiments.py` | Phase 2 (or full A+B+C) |
| `experiments/` | Config, CV, trainer, loaders, metrics |
| `results/` | JSON reports + thesis figures (in git) |
| `docs/` | Methodology, results, data policy |
| `tests/` | Unit tests |

---

## References

1. Rahman, T. et al. Deep learning-based battery health prediction for enhancing electric vehicle performance. *Sci. Rep.* **16**, 9871 (2026). [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

---

*MSc Capstone Software Artefact — Vamshi Krishna Bandari*
