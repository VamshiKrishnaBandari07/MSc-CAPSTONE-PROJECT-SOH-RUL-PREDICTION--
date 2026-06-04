# Repository Audit Report (Phase 1)

**Auditor role:** Academic reproducibility review  
**Reference:** Rahman et al., *Scientific Reports* **16**, 9871 (2026), DOI [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

---

## A. Files supporting paper reproduction

| Category | Files |
|:---|:---|
| Entry | `run_paper_experiment.py`, `paper_reproduction/run.py` |
| Model | `model_paper.py` |
| Preprocessing | `preprocess_paper.py`, `experiments/paper_preprocessing.py` |
| Loaders | `experiments/nasa_loader.py`, `oxford_loader.py`, `calce_loader.py` |
| Training | `experiments/trainer.py`, `experiments/cv.py`, `experiments/paper_config.py` |
| Evaluation | `experiments/metrics.py`, `generate_figures.py` |
| Data | `data/**`, `download_data.py` |
| Results | `results/paper_experiment_report.json`, `computational_benchmark.json`, `figures/fig01–04` |
| Tests | `tests/test_paper_preprocessing.py`, `test_loaders.py`, `test_metrics.py` |

## B. Files unrelated to the paper (removed from Git)

Listed in `docs/FILE_REMOVAL_REPORT.md`; copies in `local_archive/`.

## C. Stale metadata (fixed)

| Location | Issue | Fix |
|:---|:---|:---|
| `paper_experiment_report.json` | `experiment_b_msc`, Phase 2 `next_step` | `scripts/sanitize_paper_report.py` |
| `computational_benchmark.json` | `msc_proposed` block | Regenerated via `benchmark.py` |

## D. Spelling mistakes

| Item | Correction |
|:---|:---|
| Local folder `predications` | Document → `prediction` |
| GitHub `PREDICATION` history | Document → `PREDICTION` |

## E. MSc-era references (cleaned in public tree)

- Removed from Python entry points and JSON tails.
- Remain only in `local_archive/` (gitignored).

## F. RUL references not in paper

- Removed from GitHub code paths.
- GitHub repo name still contains `RUL` — rename recommended (`docs/NAMING_CORRECTION_REPORT.md`).

## G. Duplicate files

| Duplicate | Resolution |
|:---|:---|
| `train_paper.py` vs `run_paper_experiment.py` | Archived / removed |
| `scripts/run_full_pipeline.ps1` vs `run_paper_pipeline.ps1` | Removed old script |

## H. Dead code (removed)

- `experiments/trainer.py` non-paper `else` branch (`msc_batch_size`, undefined `MAX_EPOCHS`)
- `experiments/runtime.py` `msc_batch_size()`

## I. Unused imports

Cleaned in refactored `trainer.py`, `inference.py`, `runtime.py`.

## J. Obsolete figures (removed)

`fig02_rul_*`, `fig05`–`fig07`; standard set `fig01`–`fig04` regenerated.

---

## Methodology vs paper

| Requirement | Implementation | Match |
|:---|:---|:---:|
| ICA / DV / DC | `paper_preprocessing.py` | ✅ |
| 300-pt grid 2.5–4.2 V | `PAPER_SEQ_LEN=300` | ✅ |
| SG (15, 3) | ✅ | ✅ |
| CNN–TCN–LSTM–Attention | `model_paper.py` | ✅ |
| MSE + sigmoid SOH | ✅ | ✅ |
| 5-fold stratified CV | `experiments/cv.py` default | ✅ |
| Seed 42 | `RANDOM_SEED` | ✅ |

## Numerical replication

| Dataset | Paper hybrid RMSE | This repo (CV) | Match |
|:---|:---:|:---:|:---:|
| Oxford | 0.021 | 0.0215 ± 0.0050 | ✅ ~ |
| NASA | 0.021 | 0.0385 ± 0.0048 | ❌ |
| CALCE | (supplementary) | 0.0673 ± 0.0101 | N/A |

**Conclusion:** Methodology reproduced; **full numerical replication not achieved** on NASA.
