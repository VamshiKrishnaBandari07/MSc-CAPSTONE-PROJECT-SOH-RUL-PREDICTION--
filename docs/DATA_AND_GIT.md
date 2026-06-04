# What Is in Git vs What You Download Locally

**Policy:** Publish **code + verified results + reproduction instructions**. Keep **raw datasets** and **thesis report** local.

---

## In the repository (examiner-visible)

| Item | Location |
|:---|:---|
| Source code | `*.py`, `experiments/` |
| Phase 1 entry | `run_paper_experiment.py` |
| Phase 2 entry | `run_experiments.py` |
| Documentation | `docs/`, `README.md` |
| Verified metrics | `results/experiment_comparison_report.json`, `results/paper_experiment_report.json` |
| Thesis figures | `results/figures/fig01`–`fig07` (PNG + PDF) |
| Unit tests | `tests/` |
| Data download | `download_data.py` |
| Setup check | `scripts/verify_setup.py` |

---

## Not in the repository

| Item | Reason | Obtain |
|:---|:---|:---|
| NASA `.mat` (~56 MB) | Large binary | `python download_data.py --nasa` |
| Oxford `.mat` (~266 MB) | Large binary | `python download_data.py --oxford` |
| CALCE `.xlsx` (~50 MB) | Large binary | `python download_data.py --calce` |
| Checkpoints | Regenerated on train | Run experiments |
| MSc report `.docx` | Local thesis submission | Author machine only |
| `validation_predictions.json` | Large intermediate | `generate_figures.py` |

Listed in `.gitignore`.

---

## Reproduce verified results

```powershell
pip install -r requirements.txt
python download_data.py --all
python scripts/verify_setup.py

# Phase 1 — Paper
python run_paper_experiment.py --require-real --cpu

# Phase 2 — MSc
python run_experiments.py --msc-only --require-real --cpu

python generate_figures.py
python scripts/sync_results_docs.py
python -m pytest tests/ -v
```

Expected Experiment A SOH RMSE (5-fold CV): **Oxford ~0.0215**, NASA ~0.0385, CALCE ~0.0673 (see committed JSON).

---

## Verify real data was used

1. `results/experiment_comparison_report.json` → `data_sources` shows `real_nasa_mat`, `real_oxford_mat`, `real_calce_xlsx`.
2. `python scripts/verify_setup.py` — all datasets `[OK]`.
3. Use `--require-real` to block synthetic fallback.

---

## Dataset citations

| Dataset | URL |
|:---|:---|
| NASA PCoE | https://data.nasa.gov/dataset/li-ion-battery-aging-datasets |
| Oxford | https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac |
| CALCE | https://calce.umd.edu/battery-data |

---

*Author: Vamshi Krishna Bandari — MSc Capstone*
