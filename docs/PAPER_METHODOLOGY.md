# Experiment A — Paper Methodology Alignment

**Reference:** Rahman et al., *Scientific Reports* **16**, 9871 (2026).  
**DOI:** [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

This document maps the published methodology to this repository’s **Experiment A** implementation and states where the MSc **Experiment B** extension begins.

---

## Two-experiment structure (professional layout)

| | Experiment A — Paper reproduction | Experiment B — MSc extension |
|:---|:---|:---|
| **Script** | `run_paper_experiment.py` | `train.py` / `run_experiments.py` (MSC blocks) |
| **Model** | `model_paper.py` | `model.py` |
| **Features** | ICA, DV, **DC** (dI/dV) | ICA, DV, **DCA** (dI/dV) + joint RUL |
| **Grid** | 300 pts, 2.5–4.2 V | 100 pts (faster MSc demo) |
| **Loss** | MSE (SOH only) | MSE + RUL + physics monotonicity |
| **Target** | RMSE ≈ **0.021**, R² ≈ **0.983** | Joint SOH + RUL capability |

---

## Paper methodology implemented in code

| Paper requirement | Implementation |
|:---|:---|
| ICA \\(dQ/dV\\) | `experiments/paper_preprocessing.py` |
| DV \\(dV/dQ\\) | same |
| DC \\(dI/dV\\) | same (was voltage channel in older repo — **fixed**) |
| SG filter window=15, order=3 | `smooth_curve()` |
| Voltage grid 2.5–4.2 V, 300 points | `PAPER_SEQ_LEN=300`, `interpolate_on_voltage_grid` |
| 1D-CNN k=5 + BatchNorm + ReLU | `model_paper.py` |
| TCN dilated causal + residual | `TemporalConvNet` |
| LSTM + attention | `BatterySOHPredictorPaper` |
| MSE loss, SOH ∈ [0,1] | `train_paper_experiment`, sigmoid head |
| Gradient clip max norm 5 | `PAPER_GRAD_CLIP_NORM` |
| LR reduce on plateau (×0.5) | `ReduceLROnPlateau` |
| ~180–220 epochs, early stopping | `PAPER_MAX_EPOCHS=200`, patience 20 |
| ±10 mV augmentation | `PAPER_VOLTAGE_JITTER_V` + feature noise in trainer |
| NASA, Oxford, CALCE data | `download_data.py --all` + loaders |
| ~0.35 M parameters | `build_paper_model()` ≈ 0.39 M |

---

## Commands

```bash
# Paper experiment only (Experiment A)
python download_data.py --all
python run_paper_experiment.py

# Paper + MSc + ablation (full capstone)
python run_experiments.py
```

Reports: `results/paper_experiment_report.json`, `results/experiment_comparison_report.json`

---

## Known protocol differences (state in thesis)

1. **Cross-validation:** Paper reports stratified 5-fold CV; this repo uses **80/20 chronological split** per dataset for reproducibility on cycle-ordered aging data.
2. **Pooled cells:** NASA B0005–B0018 are pooled; paper also reports per-cell figures (B0005–B0008).
3. **Training time:** Full paper schedule (~200 epochs) takes longer than the previous 25-epoch demo; use GPU when available.
4. **Experiment B** is explicitly **not** in the published paper — it is the MSc contribution (joint SOH+RUL + physics loss).

---

*Author: Vamshi Krishna Bandari — MSc Capstone*
