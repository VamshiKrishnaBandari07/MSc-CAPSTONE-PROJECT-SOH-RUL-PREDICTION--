# `paper_exp`: Scientific Reports paper reproduction experiment

This folder contains a self-contained experiment for the paper:

> **Deep learning-based battery health prediction for enhancing electric vehicle performance**  
> Scientific Reports, DOI: `10.1038/s41598-026-39911-8`

The goal is to keep the paper-aligned method separate from the thesis extensions in the repository root.

## What is implemented

- DV/DC/ICA feature extraction:
  - Incremental Capacity Analysis: `dQ/dV`
  - Differential Voltage: `dV/dQ`
  - Differential Current: `dI/dV`
- Savitzky-Golay denoising and alignment to a standard voltage grid.
- Lightweight CNN-TCN-LSTM model with additive attention.
- SOH-only regression with MSE loss.
- Adam optimizer with paper hyperparameters:
  - learning rate `0.001`
  - betas `(0.9, 0.999)`
  - batch size `64`
  - dropout `0.2`
  - 300 epochs with early stopping patience `20`
  - `ReduceLROnPlateau` scheduler with factor `0.5` and patience `5`
- 5-fold stratified cycle segmentation over NASA, Oxford, and CALCE.
- RMSE, MAE, and R2 reporting.
- Parameter-count, latency, and estimated edge-energy reporting.
- Optional attention tensor export for interpretability inspection.

## Files

```text
paper_exp/
├── __init__.py
├── config.py       # Paper hyperparameters and reported targets
├── model.py        # CNN-TCN-LSTM-Attention SOH regressor (~0.35M params)
├── preprocess.py   # DV/DC/ICA extraction and dataset loading/fallback synthesis
├── prepare_data.py       # Converts NASA/Oxford/CALCE/Kaggle raw files into processed NPZ tensors
├── train.py              # Paper experiment: SOH-only training, metrics, efficiency report
├── modified_experiment.py # Existing repo modified/thesis SOH+RUL experiment wrapper
├── compare_results.py    # Builds JSON/Markdown comparison report
├── run_comparison.py     # Runs paper first, modified second, then comparison
└── README.md
```


## Recommended full workflow

Use this when you want the repo to look and run in the correct paper-first order.

### 1. Prepare the real Kaggle dataset

```bash
python3 -m paper_exp.prepare_data --download-kaggle --datasets KaggleSDG7 --raw-dir data --output-dir data/processed --seq-len 128
```

If you manually downloaded the Kaggle ZIP, extract it into `data/KaggleSDG7/` and run without `--download-kaggle`:

```bash
python3 -m paper_exp.prepare_data --datasets KaggleSDG7 --raw-dir data --output-dir data/processed --seq-len 128
```

### 2. Run the full comparison suite

This runs the paper experiment first, then the existing modified repo experiment, then creates a comparison report.

```bash
python3 -m paper_exp.run_comparison \
  --raw-dir data \
  --paper-datasets KaggleSDG7 \
  --modified-datasets NASA Oxford CALCE \
  --paper-epochs 300 \
  --modified-epochs 5
```

For a quick verification run:

```bash
python3 -m paper_exp.run_comparison --smoke --raw-dir data --paper-datasets KaggleSDG7
```

Generated files:

```text
paper_exp/outputs/full_comparison/
├── 01_paper_experiment/metrics.json
├── 02_modified_experiment/metrics.json
├── 03_comparison/comparison.json
├── 03_comparison/comparison.md
└── logs/
```

## Dataset input

The paper uses NASA PCoE, Oxford, and CALCE battery degradation datasets. The user-provided Kaggle dataset is also supported as `KaggleSDG7`: https://www.kaggle.com/datasets/drtawfikrrahman/deep-learning-ev-battery-pack-diagnostics-sdg-7. This repo does not ship those raw public datasets.

Place the raw files downloaded from the paper-mentioned repositories under:

```text
data/
├── NASA/      # NASA PCoE battery .mat files such as B0005.mat, B0006.mat, ...
├── Oxford/    # Oxford Battery Degradation .mat files
├── CALCE/     # CALCE battery CSV/XLS/XLSX files
└── KaggleSDG7/ # extracted files from the user-provided Kaggle dataset
```

Then convert NASA/Oxford/CALCE into the aligned DV/DC/ICA tensors used by the experiment:

```bash
python3 -m paper_exp.prepare_data --raw-dir data --output-dir data/processed
```

Convert the user-provided Kaggle dataset after manually extracting it under `data/KaggleSDG7/`:

```bash
python3 -m paper_exp.prepare_data --datasets KaggleSDG7 --raw-dir data --output-dir data/processed
python3 -m paper_exp.train --datasets KaggleSDG7 --require-real-data
```

Or download it through the Kaggle API if credentials are configured (`KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`):

```bash
python3 -m paper_exp.prepare_data --download-kaggle --datasets KaggleSDG7 --raw-dir data --output-dir data/processed
```

If you already processed the raw datasets, place files here:

```text
data/processed/NASA_paper_exp.npz
data/processed/Oxford_paper_exp.npz
data/processed/CALCE_paper_exp.npz
data/processed/KaggleSDG7_paper_exp.npz  # when using the Kaggle dataset
```

Each `.npz` file must contain:

- `features`: shape `[cycles, 3, seq_len]`
  - channel 0: ICA (`dQ/dV`)
  - channel 1: DV (`dV/dQ`)
  - channel 2: DC (`dI/dV`)
- `soh`: shape `[cycles]`

Optional arrays:

- `dataset_names`
- `cycle_indices`
- `cell_ids`

When these files are missing, the experiment uses a calibrated synthetic fallback that follows the same preprocessing and feature alignment pipeline. Use the real datasets for paper-level results. To force real paper data and prevent fallback:

```bash
python3 -m paper_exp.train --require-real-data
```

### Kaggle dataset link

The user-provided Kaggle dataset is:

```text
https://www.kaggle.com/datasets/drtawfikrrahman/deep-learning-ev-battery-pack-diagnostics-sdg-7
```

Kaggle metadata describes it as a CC0 dataset combining NASA cell-level degradation data and EV fleet pack-level telemetry. The converter accepts `.mat`, `.csv`, `.txt`, `.xls`, and `.xlsx` files and looks for voltage/current/capacity or ampere-hour-throughput columns, plus optional SOH, cycle/session, cell, vehicle, or pack identifiers.

## Run

Quick smoke verification:

```bash
python3 -m paper_exp.train --smoke
```

Prepare real paper datasets:

```bash
python3 -m paper_exp.prepare_data --raw-dir data --output-dir data/processed
```

Prepare the Kaggle dataset from the provided link:

```bash
python3 -m paper_exp.prepare_data --datasets KaggleSDG7 --raw-dir data --output-dir data/processed
```

Full paper-aligned run on prepared NASA/Oxford/CALCE data:

```bash
python3 -m paper_exp.train --require-real-data
```

Run on the prepared Kaggle dataset:

```bash
python3 -m paper_exp.train --datasets KaggleSDG7 --require-real-data
```

Save attention samples:

```bash
python3 -m paper_exp.train --save-attention
```

Outputs are written under `paper_exp/outputs/` and are ignored by git.

## Reported paper targets

The linked article reports:

| Metric | Paper target |
| --- | ---: |
| SOH RMSE | `0.021` |
| SOH R2 | `0.983` |
| Parameters | `~0.35M` |
| Latency | `6.1 ms/sample` |
| Energy | `0.63 mJ/sample` |

Local values depend on whether real datasets or the synthetic fallback are used, the selected hardware, and training duration.

