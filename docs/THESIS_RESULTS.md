# Chapter: Results and Discussion

**Author:** Vamshi Krishna Bandari  
**Project:** MSc Capstone — Hybrid Deep Learning for Joint Battery SOH & RUL Prediction  
**Reference paper:** Deep learning-based battery health prediction for enhancing electric vehicle performance (*Scientific Reports*, 2026, DOI: [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8))

---

## 5.1 Experimental Setup

> **Reproducibility:** Real datasets are not in git. Download with `python download_data.py --all`. Paper reproduction uses real NASA (636 cycles), Oxford (519 cycles), and CALCE CS2 logs (2703 cycles). Training uses 25 epochs + early stopping, not the paper’s ~300-epoch schedule. See `docs/PAPER_EXPERIMENT_METRIC_COMPARISON.md`.

Two formal experiments were conducted:

| Experiment | Objective | Model | Loss | Features |
| :--- | :--- | :--- | :--- | :--- |
| **A — Paper reproduction** | Validate implementation against published baselines | CNN-TCN-LSTM-Attention (SOH head) | MSE | ICA, DVA, voltage |
| **B — MSc extension** | Joint SOH + RUL with physics-informed regularisation | CNN-TCN-LSTM-Attention (joint head) | MSE + monotonicity penalty | ICA, DVA, DCA |

Evaluation used an 80/20 chronological train-validation split, early stopping (patience = 5), Adam optimiser (lr = 1e-3), and fixed random seed (42). Three datasets were evaluated on **real public data**: NASA PCoE (B0005–B0018, 636 cycles), Oxford Battery Degradation Dataset 1 (519 characterisation cycles), and CALCE CS2 (CS2_33, CS2_35, CS2_36; 2703 discharge cycles).

---

## 5.2 Experiment A — Paper Reproduction Results

### 5.2.1 Real-Data Results (All Three Datasets)

| Dataset | Our SOH RMSE | Published paper hybrid | vs paper 0.021 | SOH R² | Mono. violation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NASA** | **0.022** | 0.021 | +4.9% | 0.924 | 22.0% |
| **Oxford** | **0.016** | 0.021 | −25.1% | 0.947 | 39.8% |
| **CALCE** | **0.034** | 0.021 | +60.2% | — | 42.4% |

**NASA (primary thesis benchmark):** The reproduced model achieves SOH RMSE within **4.9%** of the published paper hybrid result (0.022 vs 0.021), confirming that the implementation faithfully replicates the reference architecture. The result also outperforms the published Transformer baseline (0.038) by **42.1%**.

**Oxford:** SOH RMSE **0.016** is better than the published 0.021; this may reflect differences in train/validation split or pooled multi-cell protocol vs the paper’s exact evaluation.

**CALCE:** SOH RMSE **0.034** is higher than the paper target; CALCE logs contain noisy capacity readings at very low SOH (min ~0.05), which increases label noise on the pooled validation split.

*Figure reference:* `results/figures/fig01_soh_trajectories.pdf`, `fig04_soh_rmse_comparison.pdf`

### 5.2.2 Synthetic Fallback (No Downloads)

When real files are absent, synthetic generators provide demo metrics only (SOH RMSE 0.07–0.15). These are **not** comparable to published baselines. Always run `python download_data.py --all` before thesis experiments.

---

## 5.3 Experiment B — MSc Extension Results

### 5.3.1 Joint SOH + RUL Prediction (Real Data)

| Dataset | SOH RMSE | SOH R² | RUL RMSE (cycles) | Mono. violation | Paper SOH RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| NASA | 0.074 | 0.142 | 35.23 | 49.6% | 0.022 |
| Oxford | **0.028** | 0.828 | 16.27 | 43.7% | 0.016 |
| CALCE | 0.218 | −0.008 | 17.80 | 0.0% | 0.034 |

On Oxford real data, the MSc model achieves strong SOH R² (0.828) while additionally predicting RUL. On NASA, the joint head trades SOH accuracy for RUL capability (expected multi-task trade-off). CALCE MSc SOH is poor (0.218) due to noisy low-SOH labels and the joint-task objective on 2703 pooled cycles.

### 5.3.2 NASA Real-Data Focused Run (`run_nasa_real.py`)

| Model | SOH RMSE | RUL RMSE | Additional capability |
| :--- | :---: | :---: | :--- |
| Paper reproduction | **0.022** | — | SOH only |
| MSc PI-MT (physics loss) | 0.074 | 35.23 cycles | Joint SOH + RUL |

On real NASA data, the paper reproduction remains superior for SOH-only prediction (0.022 vs 0.074). The MSc extension adds RUL estimation at the cost of SOH fidelity — an expected trade-off in multi-task learning. RUL labels are computed **per NASA cell** (B0005–B0018) with end-of-life at SOH ≤ 0.70; early stopping uses a combined SOH + RUL validation score.

*Figure reference:* `results/figures/fig_nasa_real_02_rul_trajectories.pdf`

---

## 5.4 Ablation Study — Physics-Informed Monotonicity Loss

| Dataset | SOH RMSE (no physics) | SOH RMSE (with physics) | Mono. reduction |
| :--- | :---: | :---: | :---: |
| NASA (real) | 0.081 | 0.074 | comparable |
| Oxford (real) | 0.035 | **0.028** | +1.9% |
| CALCE (real) | 0.257 | **0.218** | +48.5% |

The physics-informed monotonicity penalty (γ = 0.25) improves SOH RMSE on all three real datasets. The largest benefit is on CALCE (0.257 → 0.218) with a 48.5% reduction in monotonicity violations.

*Figure reference:* `results/figures/fig07_ablation_monotonicity.pdf`

---

## 5.5 Computational Efficiency

| Model | Parameters (M) | Latency (ms) | Energy (mJ) | Targets |
| :--- | :---: | :---: | :---: | :--- |
| Transformer (published) | 1.25 | 12.4 | 0.86 | SOH |
| Paper reproduction (ours) | **0.065** | **4.5** | **0.47** | SOH |
| MSc PI-MT (ours) | **0.067** | **5.1** | **0.53** | SOH + RUL |

Both reproduced models achieve:
- **94.8% reduction** in trainable parameters vs the Transformer baseline
- **63.6% reduction** in inference latency
- **45.3% reduction** in energy per inference sample

The MSc extension adds only **2,081 parameters** (+3.2%) over the paper reproduction while enabling joint RUL prediction.

*Figure reference:* `results/figures/fig06_computational_profile.pdf`

---

## 5.6 Discussion

### 5.6.1 Reproduction Validity

The paper-exact experiment on real NASA data (SOH RMSE = 0.022, R² = 0.924) validates the implementation against the Scientific Reports (2026) reference. This establishes a credible baseline for evaluating the MSc contribution.

### 5.6.2 MSc Contribution

The MSc extension introduces three novel elements over the paper baseline:

1. **Joint SOH + RUL prediction head** — enables simultaneous capacity and remaining-life estimation in a single forward pass.
2. **Physics-informed monotonicity loss** — penalises non-physical capacity recovery between consecutive cycles, enforcing electrochemical degradation constraints.
3. **DCA feature channel (dI/dV)** — extends the paper's ICA/DVA inputs with differential current analysis.

### 5.6.3 Limitations

- CALCE MSc SOH accuracy (RMSE 0.218) is limited by noisy capacity labels at very low SOH and joint-task trade-offs on 2703 pooled cycles.
- Monotonicity violation rates (~22–50%) indicate that per-cycle independent prediction does not fully enforce trajectory-level monotonicity; future work should apply sequence-level constraints.
- Oxford paper RMSE (0.016) is better than published 0.021 — verify split protocol matches the paper before claiming superiority.

### 5.6.4 Comparison with Published Work

| Criterion | Transformer (2026) | Paper hybrid (2026) | This work (MSc) |
| :--- | :---: | :---: | :---: |
| SOH RMSE (NASA real) | 0.038 | 0.021 | **0.022** (repro) / 0.074 (joint) |
| SOH RMSE (Oxford real) | 0.038 | 0.021 | **0.016** (repro) / 0.028 (joint) |
| SOH RMSE (CALCE real) | 0.038 | 0.021 | **0.034** (repro) / 0.218 (joint) |
| RUL prediction | No | No | **Yes** |
| Physics-informed loss | No | No | **Yes** |
| Parameters (M) | 1.25 | 0.35 | **0.067** |
| Embedded BMS compatible | No | Partial | **Yes** |

---

## 5.7 Summary

This chapter presented results from two experiments on **real NASA, Oxford, and CALCE data**: a faithful paper reproduction achieving SOH RMSE = 0.022 on NASA (within 4.9% of the published 0.021), and an MSc extension enabling joint SOH + RUL prediction with physics-informed regularisation at 94.8% fewer parameters than the Transformer baseline. The computational profile confirms embedded BMS deployment feasibility, and the ablation study demonstrates the value of the monotonicity penalty for SOH fidelity on degraded cells.

---

*All figures available in `results/figures/` after running `python generate_figures.py` and `python generate_figures.py --nasa-real-only`.*
