# Reproducibility Checklist

**Paper:** Rahman et al., *Scientific Reports* (2026) — [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

## Environment

- [ ] Python 3.9–3.11
- [ ] `pip install -r requirements.txt` or `conda env create -f environment.yml`
- [ ] `python scripts/verify_setup.py`

## Data

- [ ] `git lfs pull` **or** `python download_data.py --all`
- [ ] NASA `.mat`, Oxford `.mat`, CALCE `.xlsx` present

## Reproducibility controls

- [ ] Global seed **42** (`experiments/config.py`)
- [ ] `--require-real` (no synthetic fallback for final numbers)
- [ ] Default **5-fold stratified CV** (`--cv`)

## Primary experiment

```powershell
python run_paper_experiment.py --require-real --cpu
python scripts/sanitize_paper_report.py
```

- [ ] Output: `results/paper_experiment_report.json`
- [ ] `experiment`: `paper_reproduction` (no Phase 2 fields)

## Figures and benchmarks

```powershell
python generate_figures.py
python benchmark.py
python scripts/sync_results_docs.py
```

- [ ] `fig01`–`fig04` under `results/figures/`
- [ ] `computational_benchmark.json` has **paper_reproduction** only

## Quality gates

- [ ] `python -m pytest tests/ -v`

## One command

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_paper_pipeline.ps1
```

## Local only (not on GitHub)

- [ ] MSc extension: `local_archive/msc_capstone_extension/`
- [ ] Thesis `.docx` gitignored

## Honest reporting

- [ ] README states NASA RMSE ≠ 0.021
- [ ] Dissertation cites methodology reproduction, not full NASA numerical match
