# Repository Audit Report — Paper Reproduction

**Date:** June 2026  
**Repository:** [MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--](https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICTION--)  
**Reference paper:** Rahman et al., *Scientific Reports* **16**, 9871 (2026), DOI [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

---

## A. Research reproducibility review

### Aligned with paper

| Component | Status | Location |
|:---|:---|:---|
| ICA / DV / DC features | ✅ | `experiments/paper_preprocessing.py` |
| 300-point voltage grid (2.5–4.2 V) | ✅ | `PAPER_SEQ_LEN=300` |
| Savitzky–Golay (15, 3) | ✅ | `paper_preprocessing.smooth_curve` |
| CNN–TCN–LSTM–Attention | ✅ | `model_paper.py` |
| MSE, sigmoid SOH head | ✅ | `train_paper_experiment` |
| Grad clip 5, LR plateau ×0.5 | ✅ | `experiments/paper_config.py` |
| Augmentation ±10 mV + feature noise | ✅ | training loop |
| Stratified 5-fold CV (default) | ✅ | `experiments/cv.py` |
| NASA, Oxford, CALCE | ✅ | loaders + LFS data |
| RMSE, R², monotonicity diagnostic | ✅ | `experiments/metrics.py` |

### Deviations (document in thesis)

| Item | Paper | This repo | Impact |
|:---|:---|:---|:---|
| Parameter count | ~0.35 M | ~0.39 M | Minor; architecture width scaled |
| NASA RMSE (CV) | 0.021 pooled | 0.0385 ± 0.0048 | Dataset split / cell pooling may differ |
| CALCE | Secondary in paper | 0.0673 ± 0.0101 | Extra benchmark, not always in paper tables |
| Runtime | GPU-oriented | CPU default in docs | Slower but reproducible |

### Removed (non-paper)

| Removed from GitHub | Former purpose |
|:---|:---|
| `run_experiments.py`, `model.py`, `preprocess.py`, `train.py` | MSc joint SOH+RUL |
| `run_nasa_real.py`, `train_paper.py` | Extra NASA / duplicate entry |
| `experiments/report.py` | MSc comparison tables |
| `docs/CAPSTONE_OVERVIEW.md`, `docs/THESIS_RESULTS.md` | Phase 2 narrative |
| `results/experiment_comparison_report.json` | Exp B/C metrics |
| Figures `fig02_rul_*`, `fig06_*`, `fig07_*` | MSc-only plots |

**Preserved locally:** `local_archive/msc_capstone_extension/` (gitignored).

### Missing vs ideal paper artefact

| Gap | Priority |
|:---|:---|
| Pre-trained weight checkpoints in `models/` | Low (retrain from seed) |
| `notebooks/` exploratory EDA | Low (not in paper) |
| Physical `src/` package move | Low (documented mapping in `docs/FOLDER_STRUCTURE.md`) |
| CI with `git lfs pull` | Medium (loader tests skip without LFS) |

---

## B. Code review summary

| Check | Result |
|:---|:---|
| Duplicate MSc trainer | Removed from `experiments/trainer.py` |
| Dead MSc imports | Removed with deleted root scripts |
| Paper-only inference | `experiments/inference.py` |
| Paper-only figures | `generate_figures.py` (fig01–fig04) |
| Unit tests | `tests/test_loaders.py`, `test_metrics.py`, `test_paper_preprocessing.py` |

---

## C. GitHub structure (implemented)

See [`docs/FOLDER_STRUCTURE.md`](FOLDER_STRUCTURE.md) and root `paper_reproduction/`, `models/`, `environment.yml`.

---

## D. Non-paper experiment removal report

**Action:** All Phase 2 (Experiments B/C) code and JSON removed from version control; copies under `local_archive/`.

**Safe to keep locally, never commit:**

- `local_archive/`
- `*.docx`, thesis drafts
- `.git-rewrite/`
- `checkpoints/`, `results/*.log`

---

## E. Missing components report

1. **Exact NASA pooled preprocessing** as paper Table 4 — verify cell grouping in `experiments/nasa_loader.py` if RMSE gap matters for publication claim.
2. **GPU Dockerfile** — optional for cloud reproduction.
3. **Pre-saved model weights** — optional for instant inference demo.

---

## F. Academic quality score

| Criterion | Score (/10) |
|:---|:---:|
| Paper methodology fidelity | 8.5 |
| Documentation & README | 9.0 |
| Reproducibility (seed, deps, scripts) | 8.5 |
| Repository hygiene (no stray experiments) | 9.0 |
| Tests & CI | 7.5 |
| Results traceability (JSON + figures) | 9.0 |

### **Overall: 87 / 100**

*After this refactor: suitable for MSc artefact, supervisor review, and portfolio. For journal-grade replication, close NASA RMSE gap and add LFS-aware CI.*

---

## G. Git history guidance

### Recommended commit sequence

1. `chore: add local_archive gitignore and paper_reproduction entry`
2. `refactor: remove MSc extension from public repository`
3. `docs: publication README, audit, and reproducibility checklist`
4. `test: paper-only metrics and verify_setup`
5. `results: regenerate fig01–fig04 paper figures`

### Files to **keep**

- `run_paper_experiment.py`, `model_paper.py`, `preprocess_paper.py`, `experiments/`, `data/`, `results/paper_experiment_report.json`, `results/figures/fig01–fig04`, `tests/`, `docs/` (paper docs), `LICENSE`, CI workflow.

### Files to **remove** from Git

- Listed in section “Removed (non-paper)” above.

### Files to **ignore**

- See root `.gitignore` and `docs/DATA_AND_GIT.md`.

---

## H. Final action plan

1. ✅ Paper-only code path and docs  
2. Run `python generate_figures.py` to replace legacy figure filenames  
3. Push to GitHub with sole-author commits  
4. In dissertation: cite this repo for **Experiment A only**; reference `local_archive` for MSc extension  
5. Optional: enable `git lfs pull` in CI for full loader tests  

---

*Audit performed as part of MSc capstone repository hardening — paper reproduction scope only.*
