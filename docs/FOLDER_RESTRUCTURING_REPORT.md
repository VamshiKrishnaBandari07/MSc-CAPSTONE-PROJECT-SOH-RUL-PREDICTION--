# Folder Restructuring Report

## Design principle

Keep a **flat, import-stable layout** suitable for academic reproduction (no breaking `from experiments...` paths). Logical `src/` mapping is documented, not physically moved.

## Target structure (implemented)

```
battery-soh-paper-reproduction/     # recommended GitHub name
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│   ├── NASA/
│   ├── Oxford/
│   └── CALCE/
├── experiments/          # preprocessing, training, evaluation, utils
├── model_paper.py
├── preprocess_paper.py
├── run_paper_experiment.py
├── generate_figures.py
├── benchmark.py
├── download_data.py
├── paper_reproduction/
├── models/               # documentation → model_paper.py
├── results/
│   ├── paper_experiment_report.json
│   ├── computational_benchmark.json
│   └── figures/fig01–fig04
├── tests/
├── scripts/
└── docs/
```

## Logical `src/` map

| Logical module | Path |
|:---|:---|
| `data_preprocessing/` | `preprocess_paper.py`, `experiments/paper_preprocessing.py`, `*_loader.py` |
| `training/` | `experiments/trainer.py`, `model_paper.py` |
| `evaluation/` | `experiments/cv.py`, `experiments/metrics.py` |
| `utils/` | `experiments/io_utils.py`, `experiments/runtime.py`, `experiments/config.py` |

## New additions

- `paper_reproduction/run.py` — wrapper entry point
- `scripts/sanitize_paper_report.py` — JSON metadata cleaner
- `scripts/run_paper_pipeline.ps1` — one-command reproduction
