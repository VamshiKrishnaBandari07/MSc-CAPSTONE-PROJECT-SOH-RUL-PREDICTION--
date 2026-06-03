# Examiner / supervisor review checklist — MSc Capstone SOH & RUL Prediction
#
# Use this document when preparing for viva, supervisor meetings, or external review.
# Each section maps to what an MSc AI examiner typically verifies.

## 1. Research question and contribution clarity

| Check | Status | Evidence |
| :--- | :--- | :--- |
| Baseline paper identified with DOI | Pass | README, `docs/THESIS_RESULTS.md` |
| Experiment A (reproduction) clearly separated from Experiment B (extension) | Pass | `train_paper.py` vs `train.py`, `model_paper.py` vs `model.py` |
| MSc contribution stated (joint SOH+RUL + physics monotonicity loss) | Pass | README §Physics-informed joint loss |
| Claims match what code actually does | Pass | See limitations below |

## 2. Reproducibility

| Check | Status | Evidence |
| :--- | :--- | :--- |
| Fixed random seed | Pass | `experiments/config.py` → `RANDOM_SEED = 42` |
| Requirements pinned (major versions) | Pass | `requirements.txt` |
| One-command setup documented | Pass | README Setup + Quick start |
| Real data not in git (documented) | Pass | `.gitignore`, README §Reproducibility |
| Data download script | Pass | `python download_data.py --all` (NASA, Oxford, CALCE) |
| Config snapshot in experiment JSON | Pass | `data_sources`, `experiment_config` in reports |
| Which runs use real vs synthetic | Pass | Real when files in `data/`; synthetic fallback otherwise |

**Authoritative real-data results:** `python download_data.py --all` then `python run_experiments.py`

| Dataset | Paper SOH RMSE | vs published 0.021 |
| :--- | :---: | :---: |
| NASA | 0.022 | +4.9% ✓ |
| Oxford | 0.016 | −25% (verify split) |
| CALCE | 0.034 | +60% (noisy labels) |

## 3. Experiment A — Paper reproduction

| Check | Status | Notes |
| :--- | :--- | :--- |
| SOH-only head, MSE loss | Pass | `model_paper.py`, `train_paper_experiment` |
| Paper features: ICA, DVA, voltage (not DCA) | Pass | `preprocess_paper.py` |
| Real NASA `.mat` parsing | Pass | `experiments/nasa_loader.py` |
| Real Oxford `.mat` parsing | Pass | `experiments/oxford_loader.py` (519 cycles) |
| Real CALCE `.xlsx` parsing | Pass | `experiments/calce_loader.py` (2703 cycles) |
| SOH RMSE within ~5% of published 0.021 on real NASA | Pass | 0.022 local best |
| Oxford/CALCE real data | Pass | Real loaders implemented; download required |
| 300-epoch paper schedule | **Not matched** | 25 epochs + early stopping — state in thesis |
| Parameter count vs paper 0.35M | **Gap** | Local ~0.065M — lighter architecture variant |

## 4. Experiment B — MSc extension

| Check | Status | Notes |
| :--- | :--- | :--- |
| Joint SOH + RUL heads | Pass | `model.py` |
| Physics monotonicity penalty | Pass | `JointPhysicsInformedLoss` in `experiments/trainer.py` |
| Ablation (Experiment C) without physics | Pass | `run_experiments.py`, `run_nasa_real.py` |
| Per-cell RUL labels on real NASA | Pass | `preprocess.py` — RUL computed per `.mat` cell |
| Early stopping respects both SOH and RUL | Pass | Combined validation score in trainer |
| Multi-task trade-off documented | Pass | MSc SOH RMSE higher than paper-only on same data |

## 5. Evaluation methodology

| Check | Status | Notes |
| :--- | :--- | :--- |
| Chronological 80/20 train/val split | Pass | `split_indices` in trainer |
| RMSE, MAE, R² reported | Pass | `experiments/metrics.py` |
| Monotonicity violation rate | Pass | Lower is better; report alongside SOH |
| Published baselines in comparison table | Pass | `PAPER_REFERENCE` in config |
| Computational benchmark (params, latency, energy) | Pass | `benchmark.py` |
| Energy is estimated, not measured | Pass | Documented in README and JSON |

## 6. Software quality

| Check | Status | Notes |
| :--- | :--- | :--- |
| Modular experiment package | Pass | `experiments/` |
| Checkpoints saved on best validation | Pass | `experiments/io_utils.py` |
| JSON reports for thesis tables | Pass | `results/*.json` |
| Thesis figures (PNG + PDF) | Pass | `generate_figures.py` |
| Unit tests | Partial | `tests/test_metrics.py` — run `pytest tests/` |
| No spurious co-authors in git history | Pass | Author: Vamshi Krishna Bandari only |

## 7. Honest limitations (state in thesis Discussion)

1. Training budget is **25 epochs + early stopping**, not the paper’s longer schedule.
2. NASA/Oxford/CALCE cells are **pooled**; not single-cell hold-out as in some paper protocols.
3. **CALCE MSc SOH** (0.218) is poor due to noisy low-SOH labels and joint-task trade-off.
4. **Latency and energy** are hardware-dependent estimates.
5. **Parameter count** differs from the published 0.35M figure.
6. Physics monotonicity penalty operates **within mini-batches** (chronological order preserved).

## 8. Recommended commands before submission

```bash
pip install -r requirements.txt
pytest tests/ -v
python download_data.py --all
python run_experiments.py
python generate_figures.py
python benchmark.py
```

## 9. Key files for viva preparation

| File | Purpose |
| :--- | :--- |
| `docs/PAPER_EXPERIMENT_METRIC_COMPARISON.md` | Paper repro metric history |
| `docs/THESIS_RESULTS.md` | Results chapter draft |
| `results/experiment_comparison_report.json` | Full real-data metrics (all datasets) |
| `checkpoints/paper_nasa_real.pt` | Best paper reproduction weights |
| `checkpoints/msc_nasa_real.pt` | Best MSc extension weights |

---

*This checklist is maintained as part of the MSc capstone software artefact.*
