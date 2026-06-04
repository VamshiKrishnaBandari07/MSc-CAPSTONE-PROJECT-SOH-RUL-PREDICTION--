# Reproducibility Checklist

Use this list for supervisor review, dissertation appendix, or GitHub reproducibility audit.

## Environment

- [ ] Python 3.9–3.11 installed
- [ ] `pip install -r requirements.txt` or `conda env create -f environment.yml`
- [ ] `python scripts/verify_setup.py` passes imports
- [ ] `git lfs pull` completed (or `python download_data.py --all`)

## Determinism

- [ ] Global seed **42** (`experiments/config.py` → `set_seed()`)
- [ ] Same `torch`, `numpy`, `scipy` versions as `requirements.txt`
- [ ] `--require-real` flag used (no synthetic fallback)

## Data

- [ ] NASA `.mat` files present (`data/NASA/`)
- [ ] Oxford `.mat` present (`data/Oxford/`)
- [ ] CALCE `.xlsx` present (`data/CALCE/`)

## Paper protocol (primary)

- [ ] `python run_paper_experiment.py --require-real --cpu` (default **5-fold CV**)
- [ ] Output: `results/paper_experiment_report.json`
- [ ] Oxford SOH RMSE near **0.021** (paper target; expect ± fold variance)

## Figures and docs

- [ ] `python generate_figures.py` → `results/figures/fig01`–`fig04`
- [ ] `python scripts/sync_results_docs.py` → `docs/RESULTS.md`

## Quality gates

- [ ] `python -m pytest tests/ -v`
- [ ] Optional: `python benchmark.py` → `results/computational_benchmark.json`

## One-command pipeline

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_paper_pipeline.ps1
```

## Not in this repository (local only)

- [ ] MSc joint SOH+RUL extension → `local_archive/msc_capstone_extension/`
- [ ] Final `.docx` thesis → gitignored
