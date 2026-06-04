# Repository layout

Paper reproduction uses a **flat Python layout** (imports stable across clones). Logical mapping to a classic `src/` tree:

```
project-root/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/raw/          → data/NASA, data/Oxford, data/CALCE
├── data/processed/    → (features built in-memory; no separate cache)
├── notebooks/         → (none — scripts only)
├── src/               → see mapping below
├── models/            → model_paper.py (documented in models/README.md)
├── results/
│   ├── figures/
│   ├── tables/        → (metrics in JSON)
│   └── metrics/       → paper_experiment_report.json
├── paper_reproduction/
└── docs/
```

| `src/` subfolder | Actual paths |
|:---|:---|
| `data_preprocessing/` | `preprocess_paper.py`, `experiments/paper_preprocessing.py`, `experiments/*_loader.py` |
| `training/` | `experiments/trainer.py`, `run_paper_experiment.py`, `model_paper.py` |
| `evaluation/` | `experiments/cv.py`, `experiments/metrics.py`, `generate_figures.py` |
| `utils/` | `experiments/io_utils.py`, `experiments/runtime.py`, `experiments/config.py` |
