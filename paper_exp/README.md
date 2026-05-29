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
├── train.py        # Cross-validation training, metrics, efficiency report
└── README.md
```

## Dataset input

The paper uses NASA PCoE, Oxford, and CALCE battery degradation datasets. This repo does not ship those raw public datasets.

If you already processed the raw datasets, place files here:

```text
data/processed/NASA_paper_exp.npz
data/processed/Oxford_paper_exp.npz
data/processed/CALCE_paper_exp.npz
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

When these files are missing, the experiment uses a calibrated synthetic fallback that follows the same preprocessing and feature alignment pipeline. Use the real datasets for paper-level results.

## Run

Quick smoke verification:

```bash
python -m paper_exp.train --smoke
```

Full paper-aligned run:

```bash
python -m paper_exp.train
```

Save attention samples:

```bash
python -m paper_exp.train --save-attention
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

