# Experiment A — Paper Methodology Alignment

**Reference:** Rahman et al., *Scientific Reports* **16**, 9871 (2026).  
**DOI:** [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

## Two-phase MSc capstone (required order)

| Phase | Experiment | Script | When to run |
|:---:|:---|:---|:---|
| **1** | **A — Paper reproduction** | `run_paper_experiment.py` | **First** — validate against published 0.021 RMSE |
| **2** | **B — MSc extension** | `run_experiments.py --msc-only` | **After A** — joint SOH+RUL + physics loss |
| **2** | **C — Ablation** | included in `--msc-only` / full suite | Compare with/without monotonicity penalty |

```powershell
python download_data.py --all
python run_paper_experiment.py --require-real --cpu          # PHASE 1 (5-fold CV)
python run_experiments.py --msc-only --require-real --cpu    # PHASE 2
```

Fast supplementary run (not paper Table 4 protocol): add `--chrono` to Experiment A.

---

## Paper methodology ↔ code

| Paper requirement | Implementation |
|:---|:---|
| ICA \\(dQ/dV\\) | `experiments/paper_preprocessing.py` |
| DV \\(dV/dQ\\) | same |
| DC \\(dI/dV\\) | same |
| SG filter window=15, order=3 | `smooth_curve()` |
| Voltage grid 2.5–4.2 V, **300 points** | `PAPER_SEQ_LEN=300` |
| 1D-CNN k=5 + BatchNorm + ReLU | `model_paper.py` |
| TCN dilated causal + residual | `TemporalConvNet` |
| LSTM + attention | `BatterySOHPredictorPaper` |
| MSE loss, SOH ∈ [0,1] | sigmoid head + `train_paper_experiment` |
| Gradient clip max norm 5 | `PAPER_GRAD_CLIP_NORM` |
| LR reduce on plateau (×0.5) | `ReduceLROnPlateau` |
| ~180–220 epochs, early stopping patience 20 | `PAPER_MAX_EPOCHS=200` |
| ±10 mV augmentation + feature noise | `PAPER_VOLTAGE_JITTER_V`, `PAPER_FEATURE_NOISE` |
| **Stratified 5-fold CV** | `experiments/cv.py` — **default** (`--cv` / no flag) |
| NASA, Oxford, CALCE | `download_data.py --all` |
| ~0.35 M parameters | `build_paper_model()` ≈ 0.39 M |

---

## Experiment B (MSc — not in paper)

| | Experiment A | Experiment B |
|:---|:---|:---|
| Script | `run_paper_experiment.py` | `train.py` / `run_experiments.py --msc-only` |
| Model | `model_paper.py` | `model.py` |
| Features | ICA, DV, DC (300-pt grid) | ICA, DVA, DCA (100-pt grid) |
| Outputs | SOH only | **Joint SOH + RUL** |
| Loss | MSE | MSE + RUL + **monotonicity penalty** |

---

## Evaluation protocols

| Protocol | Flag | Use case |
|:---|:---|:---|
| **Stratified 5-fold CV** | default / `--cv` | **Paper comparison** (target RMSE 0.021) |
| Chronological 80/20 | `--chrono` | Fast local debugging on CPU |

State in thesis: primary metrics use **5-fold CV**; chronological results are supplementary.

---

## Commands

```powershell
# PHASE 1 — paper (real data, 5-fold CV)
python run_paper_experiment.py --require-real --cpu

# PHASE 1 — NASA only, fast chronological
python run_paper_experiment.py --require-real --cpu --dataset NASA --chrono

# PHASE 2 — MSc after paper
python run_experiments.py --msc-only --require-real --cpu

# Full A+B+C (long on CPU with 5-fold)
python run_experiments.py --require-real --cpu
```

Reports: `results/paper_experiment_report.json`, `results/experiment_comparison_report.json`

---

*Author: Vamshi Krishna Bandari — MSc Capstone*
