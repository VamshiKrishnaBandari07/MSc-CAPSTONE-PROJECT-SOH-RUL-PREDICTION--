# Paper methodology — code mapping

**Reference:** Rahman et al., *Scientific Reports* (2026) — [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

## Datasets

| Dataset | Loader | Format |
|:---|:---|:---|
| NASA PCoE | `experiments/nasa_loader.py` | `.mat` |
| Oxford BDD | `experiments/oxford_loader.py` | `.mat` |
| CALCE CS2 | `experiments/calce_loader.py` | `.xlsx` |

## Preprocessing

| Paper | Code |
|:---|:---|
| ICA dQ/dV, DV dV/dQ, DC dI/dV | `experiments/paper_preprocessing.py` |
| Grid 2.5–4.2 V, 300 points | `PAPER_SEQ_LEN = 300` |
| Savitzky–Golay (15, 3) | `smooth_curve()` |
| Orchestration | `preprocess_paper.py` |

## Model & training

| Paper | Code |
|:---|:---|
| CNN–TCN–LSTM–attention | `model_paper.py` |
| MSE, sigmoid SOH | `experiments/trainer.py` |
| Adam 1e-3, WD 1e-5 | `experiments/paper_config.py` |
| Max 200 epochs, patience 20 | same |
| Grad clip 5, LR plateau ×0.5 | same |
| ±10 mV + feature noise | `PAPER_VOLTAGE_JITTER_V`, `PAPER_FEATURE_NOISE` |

## Evaluation

| Protocol | Flag | Code |
|:---|:---|:---|
| **Stratified 5-fold CV** | default | `experiments/cv.py` |
| Chronological 80/20 | `--chrono` | supplementary only |

## Run

```powershell
python run_paper_experiment.py --require-real --cpu
```
