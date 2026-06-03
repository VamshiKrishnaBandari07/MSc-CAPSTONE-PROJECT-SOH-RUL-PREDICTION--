# Supervisor Review Guide — MSc Capstone SOH & RUL Prediction

**Student:** Vamshi Krishna Bandari  
**Repository:** [MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICATION-](https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICATION-)

Use this guide for a **5-minute desk review** or a **full reproduction session**.

---

## 1. Research question (30 seconds)

> *Does a physics-informed joint SOH + RUL deep learning model extend the Scientific Reports (2026) hybrid architecture while remaining deployable on embedded BMS hardware?*

| Track | What it proves |
| :--- | :--- |
| **Experiment A** | Implementation faithfully reproduces the published paper (SOH-only, real datasets) |
| **Experiment B** | MSc contribution — joint SOH + RUL + monotonicity physics loss |
| **Experiment C** | Ablation — physics penalty effect |

Baseline paper: [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

---

## 2. Five-minute review (no training required)

Your student has committed **verified metrics and thesis figures** so you can review without re-running ~14 minutes of GPU/CPU training.

| Step | Action | Time |
| :--- | :--- | :--- |
| 1 | Read this file | 2 min |
| 2 | Open `results/experiment_comparison_report.json` | 1 min |
| 3 | View `results/figures/fig04_soh_rmse_comparison.pdf` | 1 min |
| 4 | Skim `docs/THESIS_RESULTS.md` (Results chapter draft) | 2 min |

### Key results (real data, seed 42)

**Experiment A — Paper reproduction (target SOH RMSE = 0.021)**

| Dataset | Our SOH RMSE | vs paper | Real cycles |
| :--- | :---: | :---: | :---: |
| NASA | **0.022** | +4.9% | 636 |
| Oxford | **0.016** | −25% | 519 |
| CALCE | **0.034** | +60% | 2703 |

**Experiment B — MSc extension**

| Dataset | SOH RMSE | RUL RMSE | SOH R² |
| :--- | :---: | :---: | :---: |
| NASA | 0.074 | 35.2 cycles | 0.14 |
| Oxford | 0.028 | 16.3 cycles | 0.83 |
| CALCE | 0.218 | 17.8 cycles | −0.01 |

**Primary thesis claim:** NASA paper reproduction SOH RMSE **0.022 vs published 0.021** — credible validation of the baseline implementation.

---

## 3. Is the data “real” or synthetic?

| Dataset | In GitHub? | Locally after download? | Loader |
| :--- | :---: | :---: | :--- |
| NASA B0005–B0018 | No (gitignored) | Yes — official `.mat` | `experiments/nasa_loader.py` |
| Oxford Dataset 1 | No (gitignored) | Yes — official `.mat` | `experiments/oxford_loader.py` |
| CALCE CS2_33/35/36 | No (gitignored) | Yes — official `.xlsx` | `experiments/calce_loader.py` |

Raw datasets are **not in git** (standard practice — size + licensing).  
Code, download script, metrics JSON, and figures **are in git**.

See `docs/DATA_AND_GIT.md` for full policy.

---

## 4. Full reproduction (~20 minutes)

```bash
git clone https://github.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICATION-.git
cd MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICATION-
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/verify_setup.py                      # check environment
python download_data.py --all                       # ~500 MB download
python run_experiments.py                           # ~14 min on CPU
python generate_figures.py
pytest tests/ -v
```

Expected NASA paper SOH RMSE after full run: **0.022 ± 0.002**.

---

## 5. What I would ask in a supervision meeting

### Strengths to acknowledge

- Clear separation of Experiment A (reproduction) vs B (extension)
- Real public datasets with automated download — not synthetic-only demos
- NASA reproduction within 5% of published hybrid baseline
- Honest documentation of limitations (25-epoch schedule, pooled cells, CALCE noise)
- JSON provenance (`data_sources`, `experiment_config`) in experiment reports
- Thesis-ready figures exported as PNG + PDF

### Limitations the student should articulate

1. **Training budget:** 25 epochs + early stopping, not paper’s ~300 epochs
2. **Cell pooling:** All cells pooled; not single-cell hold-out like some paper protocols
3. **Oxford 0.016:** Better than paper — may reflect different split; verify before claiming superiority
4. **CALCE MSc SOH (0.218):** Noisy low-SOH labels + joint-task trade-off
5. **Parameter count:** Local ~0.065 M vs paper’s reported 0.35 M
6. **Energy/latency:** Estimated from CPU timing, not lab instrument measurement

### Files to open together

| File | Why |
| :--- | :--- |
| `model_paper.py` vs `model.py` | Shows reproduction vs extension |
| `preprocess_paper.py` vs `preprocess.py` | ICA/DVA/voltage vs ICA/DVA/DCA |
| `experiments/trainer.py` | Physics monotonicity loss |
| `docs/EXAMINER_CHECKLIST.md` | Viva preparation checklist |

---

## 6. Grading rubric alignment (typical MSc AI capstone)

| Criterion | Evidence | Rating guidance |
| :--- | :--- | :--- |
| Literature grounding | DOI cited, Experiment A reproduces published architecture | Strong |
| Novel contribution | Joint SOH+RUL + physics loss (Experiment B) | Adequate–Strong |
| Experimental rigour | 3 real datasets, ablation, computational benchmark | Strong |
| Reproducibility | Download script, seed, config snapshot, committed results | Strong |
| Critical analysis | Limitations documented honestly | Strong |
| Software quality | Modular package, tests, JSON reports, figures | Adequate–Strong |

---

## 7. Quick health check

```bash
python scripts/verify_setup.py
```

Prints: Python version, dependency status, which datasets are present, whether result files exist.

---

*Last updated after verified real-data run — NASA SOH RMSE 0.022, Oxford 0.016, CALCE 0.034 (Experiment A).*
