# Paper Experiment — Metric Comparison & Reproducibility

**Author:** Vamshi Krishna Bandari  
**Reference:** Rahman et al., *Scientific Reports* **16**, 9871 (2026). [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

---

## Published baselines (Table 4)

| Model | SOH RMSE | Params (M) | Latency (ms) |
|:---|:---:|:---:|:---:|
| Transformer (paper baseline) | 0.038 | 1.25 | 12.4 |
| **Paper hybrid (target)** | **0.021** | 0.35 | 6.1 |

---

## This repository — Experiment A results

### Verified run (stratified 5-fold CV, full ICA+DV+DC pipeline)

| Dataset | SOH RMSE (mean ± std) | SOH R² | vs paper 0.021 | vs Transformer 0.038 |
|:---|:---:|:---:|:---:|:---:|
| NASA | **0.0385 ± 0.0048** | 0.915 | +83% | **−1%** |
| Oxford | **0.0215 ± 0.0050** | 0.951 | **+2%** | **−43%** |
| CALCE | **0.0673 ± 0.0101** | 0.950 | +220% | −77% |

Source: `results/paper_experiment_report.json` (real data, seed 42, ~49 min Phase 1 on CPU).

### Supplementary (chronological 80/20, debug protocol)

| Dataset | SOH RMSE | SOH R² |
|:---|:---:|:---:|
| NASA | 0.0323 | 0.836 |
| Oxford | 0.0325 | 0.773 |
| CALCE | 0.0564 | 0.929 |

Use `--chrono` only for fast debugging; not for paper claims.

---

## Methodology alignment checklist

| Paper requirement | Repo implementation | Status |
|:---|:---|:---:|
| ICA dQ/dV | `experiments/paper_preprocessing.py` | OK |
| DV dV/dQ | same | OK |
| DC dI/dV | same (corrected from early voltage proxy) | OK |
| SG window=15, order=3 | `smooth_curve()` | OK |
| 300-pt grid 2.5–4.2 V | `PAPER_SEQ_LEN=300` | OK |
| CNN k=5 + BN + TCN + LSTM + Attn | `model_paper.py` | OK |
| MSE loss, SOH sigmoid | `train_paper_experiment` | OK |
| Grad clip 5, LR schedule | `paper_config.py` | OK |
| ~200 epochs, early stop | `PAPER_MAX_EPOCHS=200` | OK |
| ±10 mV augmentation | `PAPER_VOLTAGE_JITTER_V` + feature noise | OK |
| Stratified 5-fold CV | `experiments/cv.py` | OK (default) |
| NASA + Oxford + CALCE | `download_data.py --all` | OK |

---

## Known protocol differences

1. **Evaluation split:** Experiment A uses **5-fold CV** (default); MSc extension uses chronological 80/20.
2. **Parameter count:** ~0.39 M vs paper ~0.35 M — minor architecture width difference.
3. **Training stability:** Non-finite validation on some folds; best checkpoint retained per fold.
4. **Pooled cells:** Multi-cell cycles concatenated; paper also reports per-cell figures.

---

## How to reproduce

```powershell
python download_data.py --all
python run_paper_experiment.py --require-real --cpu
```

Compare output to `PAPER_TARGET_SOH_RMSE = 0.021` in `experiments/paper_config.py`.

---

*See also: [`RESULTS.md`](RESULTS.md), [`PAPER_METHODOLOGY.md`](PAPER_METHODOLOGY.md)*
