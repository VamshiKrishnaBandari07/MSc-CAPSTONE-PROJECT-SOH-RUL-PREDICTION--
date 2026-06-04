# Paper Methodology Alignment

**Reference:** Rahman et al., *Scientific Reports* **16**, 9871 (2026).  
**DOI:** [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

This repository implements **only** the published hybrid SOH prediction pipeline (no joint RUL or MSc extensions on GitHub).

## Reproduction command

```powershell
git lfs pull
python download_data.py --all
python run_paper_experiment.py --require-real --cpu
python generate_figures.py
```

Fast supplementary run (not paper Table 4 protocol): add `--chrono` or `--dataset NASA`.

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
| **Stratified 5-fold CV** | `experiments/cv.py` — default (`--cv`) |
| NASA, Oxford, CALCE | `download_data.py --all` |
| ~0.35 M parameters | `build_paper_model()` ≈ 0.39 M |

---

## Evaluation protocols

| Protocol | Flag | Use case |
|:---|:---|:---|
| **Stratified 5-fold CV** | default / `--cv` | **Paper comparison** (target RMSE 0.021 on NASA pooled) |
| Chronological 80/20 | `--chrono` | Fast local debugging on CPU |

Primary reported metrics in `results/paper_experiment_report.json` use **5-fold CV**.

---

## Module map (logical `src/` layout)

| Logical area | Path |
|:---|:---|
| Data preprocessing | `preprocess_paper.py`, `experiments/paper_preprocessing.py`, `experiments/*_loader.py` |
| Training | `experiments/trainer.py`, `run_paper_experiment.py` |
| Evaluation | `experiments/cv.py`, `experiments/metrics.py` |
| Utils | `experiments/io_utils.py`, `experiments/runtime.py`, `experiments/config.py` |

---

*Academic paper reproduction artefact — Vamshi Krishna Bandari*
