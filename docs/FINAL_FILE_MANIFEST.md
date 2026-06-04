# Final File Manifest

## Files that should remain on GitHub

### Root

- `README.md`, `LICENSE`, `requirements.txt`, `environment.yml`, `.gitignore`, `.gitattributes`
- `run_paper_experiment.py`, `model_paper.py`, `preprocess_paper.py`
- `generate_figures.py`, `benchmark.py`, `download_data.py`

### Code

- `experiments/` (all modules except removed `report.py`)
- `paper_reproduction/`
- `models/README.md`
- `tests/`
- `scripts/` (`verify_setup`, `sync_results_docs`, `sanitize_paper_report`, `run_paper_pipeline`, `git_commit_sole_author`)

### Data (Git LFS)

- `data/NASA/*.mat`, `data/Oxford/*.mat`, `data/CALCE/**/*.xlsx`

### Results

- `results/paper_experiment_report.json`
- `results/computational_benchmark.json`
- `results/figures/fig01_soh_trajectories.{png,pdf}`
- `results/figures/fig02_soh_scatter.{png,pdf}`
- `results/figures/fig03_soh_rmse_comparison.{png,pdf}`
- `results/figures/fig04_training_convergence.{png,pdf}`

### Documentation

- `docs/PAPER_METHODOLOGY.md`, `docs/RESULTS.md`, `docs/DATA_AND_GIT.md`
- `docs/REPRODUCIBILITY_CHECKLIST.md`, `docs/GITHUB.md`, `docs/FOLDER_STRUCTURE.md`
- `docs/REPOSITORY_AUDIT.md`, `docs/RESULTS_CONSISTENCY_REPORT.md`
- `docs/NAMING_CORRECTION_REPORT.md`, `docs/FILE_REMOVAL_REPORT.md`
- `docs/FOLDER_RESTRUCTURING_REPORT.md`, `docs/ACADEMIC_ASSESSMENT.md`
- `docs/PAPER_EXPERIMENT_METRIC_COMPARISON.md`

### CI

- `.github/workflows/ci.yml`

---

## Files that should remain local only

- `local_archive/` (entire tree — see `LOCAL_ARCHIVE_CONTENTS.md`)
- `*.docx`, thesis drafts
- `checkpoints/` (optional: commit fold weights only if required by examiner)
- `results/validation_predictions.json`, `results/*.log`
- `.git-rewrite/`, `.pytest_cache/`
