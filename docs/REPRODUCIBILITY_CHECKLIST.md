# Reproducibility checklist

**Paper:** [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

- [ ] Python 3.9+ and `pip install -r requirements.txt`
- [ ] `git lfs pull` or `python download_data.py --all`
- [ ] `python scripts/verify_setup.py`
- [ ] `python run_paper_experiment.py --require-real --cpu`
- [ ] `python scripts/sanitize_paper_report.py`
- [ ] `python generate_figures.py`
- [ ] `python -m pytest tests/ -v`

**Expected:** Oxford RMSE ≈ **0.021**; NASA may remain ≈ **0.038** with this public pipeline.
