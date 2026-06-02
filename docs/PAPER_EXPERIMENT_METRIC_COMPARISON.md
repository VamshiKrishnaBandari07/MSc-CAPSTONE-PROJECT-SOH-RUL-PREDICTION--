# Paper Experiment — Metric Comparison and Reproducibility

This document clarifies **which paper-experiment runs use real data**, how local metrics compare to the published Scientific Reports (2026) target (**SOH RMSE = 0.021**), and what was changed to improve results.

---

## Is the paper experiment “real”?

| Question | Answer |
| :--- | :--- |
| Is the **architecture** paper-faithful? | **Yes** — `model_paper.py`, ICA + DVA + voltage features, SOH-only MSE loss (`train_paper.py`). |
| Is **NASA data** real when `.mat` files are present? | **Yes** — B0005, B0006, B0007, B0018 discharge cycles (636 cycles combined) via `experiments/nasa_loader.py`. |
| Is **Oxford / CALCE** real in `train_paper.py`? | **No** — synthetic fallback only (real parsers not implemented). |
| Is the **training schedule** identical to the paper? | **No** — this repo uses **25 max epochs + early stopping (patience 5)**, not the paper’s longer schedule (often **~300 epochs** in similar work). |
| Is the **hardware** identical? | **No** — local CPU timings; paper values were measured on their setup. |
| Is the **full published protocol** replicated? | **Partially** — see gaps below. |

**Authoritative real-data paper reproduction command:**

```bash
python download_data.py --nasa
python run_nasa_real.py
```

Results: `results/nasa_real_experiment_report.json`  
Checkpoints: `checkpoints/paper_nasa_real.pt`

---

## Metric comparison table

Published reference (Scientific Reports, 2026):

| Model | SOH RMSE | Parameters | Latency | Notes |
| :--- | :---: | :---: | :---: | :--- |
| Transformer (paper baseline) | 0.038 | 1.25 M | 12.4 ms | Published table |
| Paper hybrid CNN-TCN-LSTM-Attn | **0.021** | 0.35 M | 6.1 ms | **Reproduction target** |

Local runs (this repository):

| Run ID | Command | NASA data | Best val SOH RMSE | vs paper 0.021 | R² | Epoch | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **A1 — Real NASA (best)** | `run_nasa_real.py` | **Real `.mat`** (636 cycles) | **0.0220** | **+4.9%** | 0.924 | 7 | **Use for thesis** |
| A2 — Synthetic cross-dataset | `run_experiments.py` / `train_paper.py` (no `.mat`) | Synthetic (150 cycles) | 0.0935 (NASA) | +345% | -15.3 | 1 | Demo only |
| A3 — Synthetic Oxford | `train_paper.py` | Synthetic | 0.0703 | +235% | -15.3 | 1 | Demo only |
| A4 — Synthetic CALCE | `train_paper.py` | Synthetic | 0.1461 | +596% | -47.4 | 1 | Demo only |

> **Important:** `results/experiment_comparison_report.json` may show **A2–A4** if it was generated **before** NASA `.mat` files were downloaded or before the NASA loader was fixed. Re-run after `download_data.py --nasa` to refresh NASA rows in `run_experiments.py`.

### What “real” means for run **A1**

- **Real:** NASA PCoE `.mat` files, discharge capacity SOH labels, ICA/DVA/voltage features, paper model + MSE loss.
- **Not identical to paper:** Combined 4 cells (not single-cell protocol), 80/20 chronological split, 25-epoch cap with early stopping, Adam lr=1e-3, local CPU, ~0.065 M parameters (lighter than paper’s 0.35 M reported count).

Despite these gaps, **SOH RMSE 0.022 vs 0.021** supports that the implementation is a credible reproduction on real NASA data.

---

## Changes tried (improvement history)

| Change | Purpose | Effect on NASA paper SOH RMSE |
| :--- | :--- | :--- |
| Initial synthetic-only pipeline | Quick demo without downloads | ~0.09–0.15 (not comparable to paper) |
| NASA `.mat` auto-download (`download_data.py --nasa`) | Enable real data | Required for real evaluation |
| Fixed nested `.mat` schema (`B0005` key) | Parse official NASA files | Loader stopped returning `None` |
| Discharge-only SOH from `Capacity` scalar | Correct NASA label definition | Stable SOH range 0.57–1.0 |
| Shared `experiments/nasa_loader.py` | One parser for paper + MSc paths | Consistent 636-cycle load |
| Sigmoid SOH output | Bound predictions to [0, 1] | Improved validation stability |
| LR scheduler + early stopping | Reduce overfitting | Best epoch 7, RMSE **0.022** |
| 25 epochs vs paper ~300 | Faster MSc iteration | May under-train vs full paper schedule |

---

## Best stable local metric (paper experiment)

| Metric | Value | Source file |
| :--- | :---: | :--- |
| **SOH RMSE** | **0.0220** | `results/nasa_real_experiment_report.json` |
| SOH MAE | 0.0182 | same |
| SOH R² | 0.9239 | same |
| Val cycles | 127 (20% of 636) | same |
| Train cycles | 508 | same |
| Checkpoint | `checkpoints/paper_nasa_real.pt` | epoch 7 |

---

## Script guide — which command for what?

| Goal | Command | Data |
| :--- | :--- | :--- |
| **Paper repro on real NASA** | `python run_nasa_real.py` | Real NASA |
| Paper repro all datasets | `python train_paper.py` | NASA real if `.mat` present; Oxford/CALCE synthetic |
| Full suite (paper + MSc + ablation) | `python run_experiments.py` | Same as above |
| **MSc extension demo** | `python train.py` | NASA real if `.mat` in `data/NASA/`; else synthetic |
| Thesis figures (real NASA) | `python generate_figures.py --nasa-real-only` | Uses `paper_nasa_real.pt` |

---

## Remaining gaps for full paper-level reproduction

1. **300-epoch training schedule** (or match paper’s exact epoch count and batch settings).
2. **Single-cell evaluation** matching paper’s train/test split (B0005 only, etc.).
3. **Real Oxford and CALCE** parsers and experiments.
4. **Hyperparameter parity** with the paper (learning rate schedule, batch size, window length).
5. **Parameter count alignment** (local ~0.065 M vs paper 0.35 M — verify channel widths / TCN depth).
6. **Hardware-matched latency** measurement on target edge device.

---

## Reproducibility checklist

```bash
git clone git@github.com:VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICATION-.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICATION-
pip install -r requirements.txt
python download_data.py --nasa          # ~200 MB, not in git
python run_nasa_real.py                 # paper + MSc on real NASA
python generate_figures.py --nasa-real-only
```

Expected paper reproduction SOH RMSE: **~0.022** (seed 42, CPU, early stopping).

---

*Last verified: local run with B0005–B0018 `.mat` files present in `data/NASA/`.*
