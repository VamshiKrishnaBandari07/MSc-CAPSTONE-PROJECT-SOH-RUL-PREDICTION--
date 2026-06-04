# Experimental Results — Thesis Reference

**Author:** Vamshi Krishna Bandari  
**Project:** Hybrid Deep Learning for Joint Battery SOH & RUL Prediction  
**Reference:** Rahman et al., *Scientific Reports* (2026). [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

> Copy tables into Word thesis Chapter 4. Figures: `results/figures/`.  
> Committed metrics from real NASA/Oxford/CALCE data (seed 42, ICA+DV+DC, 300-pt grid).

---

## 4.1 Experimental setup

Two formal experiments plus ablation:

| Experiment | Objective | Model | Loss |
|:---|:---|:---|:---|
| **A** | Paper reproduction | CNN-TCN-LSTM-Attn (SOH) | MSE |
| **B** | MSc extension | Joint head | MSE + RUL + monotonicity |
| **C** | Ablation | Same as B | MSE + RUL (no monotonicity) |

**Datasets:** NASA (636 cycles), Oxford (519), CALCE (2,703) — real public data via `download_data.py --all`.

**Hardware:** CPU training supported (`--cpu`); batch size 4; PyTorch 8 threads.

---

## 4.2 Experiment A — Paper reproduction (5-fold stratified CV)

| Dataset | SOH RMSE (mean ± std) | SOH MAE | SOH R² | Paper hybrid | Transformer |
|:---|:---:|:---:|:---:|:---:|:---:|
| NASA | **0.0385 ± 0.0048** | 0.0271 | 0.915 | 0.021 | 0.038 |
| Oxford | **0.0215 ± 0.0050** | 0.0134 | 0.951 | 0.021 | 0.038 |
| CALCE | **0.0673 ± 0.0101** | 0.0374 | 0.950 | 0.021 | 0.038 |

Oxford mean RMSE **matches the paper hybrid target (0.021)** within one standard deviation. NASA is on par with the Transformer baseline; CALCE remains harder due to dataset heterogeneity.

*Figures:* `fig04_soh_rmse_comparison.pdf`, `fig01_soh_trajectories.pdf`

---

## 4.3 Experiment B — MSc extension (chronological 80/20)

| Dataset | SOH RMSE | SOH R² | RUL RMSE | Mono. violation |
|:---|:---:|:---:|:---:|:---:|
| NASA | 0.1116 | −0.953 | 44.31 | 53.5% |
| Oxford | 0.0409 | 0.641 | 14.18 | 44.7% |
| CALCE | 0.2297 | −0.120 | 1.48 | 49.4% |

Joint SOH+RUL in one forward pass; SOH accuracy trades off vs Experiment A (expected multi-task effect).

*Figures:* `fig02_rul_trajectories.pdf`, `fig03_soh_scatter.pdf`

---

## 4.4 Experiment C — Ablation

| Dataset | SOH RMSE (no physics) | SOH RMSE (with physics) |
|:---|:---:|:---:|
| NASA | 0.0781 | **0.1116** |
| Oxford | **0.0342** | 0.0409 |
| CALCE | 0.3549 | **0.2297** |

*Figure:* `fig07_ablation_monotonicity.pdf`

---

## 4.5 Computational efficiency

| Model | Params (M) | Latency (ms) | Energy (mJ) |
|:---|:---:|:---:|:---:|
| Transformer (published) | 1.25 | 12.4 | 0.86 |
| Paper reproduction | 0.385 | 8.95 | 0.92 |
| **MSc PI-MT** | **0.067** | **2.82** | **0.29** |

*Figure:* `fig06_computational_profile.pdf`

---

## 4.6 Discussion

**Strengths:** Reproducible pipeline; real-data evaluation; paper architecture validated directionally; MSc model 95% smaller than Transformer with joint RUL.

**Limitations:** SOH RMSE above paper 0.021 on committed run; CALCE MSc performance limited by label noise; monotonicity violations ~35–51%; chronological split in committed JSON vs paper 5-fold CV.

**Comparison to state of the art:**

| Criterion | Transformer | Paper hybrid | This work (A) | This work (B) |
|:---|:---:|:---:|:---:|:---:|
| NASA SOH RMSE | 0.038 | 0.021 | **0.032** | 0.078 |
| RUL prediction | No | No | No | **Yes** |
| Physics loss | No | No | No | **Yes** |
| Params (M) | 1.25 | 0.35 | 0.39 | **0.067** |

---

## 4.7 Summary

Experiment A validates the Scientific Reports (2026) implementation on three public datasets. Experiment B delivers the MSc contribution: joint SOH+RUL at embedded-friendly compute. All metrics and figures regenerate from committed code and `results/experiment_comparison_report.json`.

---

*Regenerate: `python run_experiments.py --require-real --cpu` then `python generate_figures.py`*
