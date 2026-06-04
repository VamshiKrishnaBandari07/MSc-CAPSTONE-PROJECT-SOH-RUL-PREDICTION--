# Methodology — Rahman et al. (2026)

**Article:** [Deep learning-based battery health prediction for enhancing electric vehicle performance](https://www.nature.com/articles/s41598-026-39911-8)  
**DOI:** [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

## Datasets (Section 3)

| Repository | Loader |
|:---|:---|
| NASA PCoE | `experiments/nasa_loader.py` |
| Oxford Battery Degradation | `experiments/oxford_loader.py` |
| CALCE | `experiments/calce_loader.py` |

## SOH label (Equation 1 in paper)

SOH(k) = Q_k / Q_BoL — beginning-of-life capacity per cell.

## Preprocessing

1. Savitzky–Golay smoothing (window **15**, order **3**)  
2. Interpolate V, I, Q onto **300** points from **2.5 V** to **4.2 V**  
3. Compute **ICA** (dQ/dV), **DV** (dV/dQ), **DC** (dI/dV)  
4. Min–max scale channels (per cycle)  
5. Training augmentation: **±10 mV** voltage jitter (`preprocess_paper` / `paper_preprocessing`) plus scaled feature noise in `trainer.py`  

Code: `experiments/paper_preprocessing.py`, `preprocess_paper.py`

## Model (Fig. 1)

1D-CNN → TCN (dilated causal) → LSTM → attention → sigmoid SOH head.

Code: `model_paper.py` (~0.39M parameters; paper reports ~0.35M)

## Training (Section 3.4)

| Hyperparameter | Value |
|:---|:---|
| Loss | MSE |
| Optimizer | Adam, lr 1e-3 |
| Max epochs | ~200 (early stop patience 20) |
| Grad clip | 5 |
| LR scheduler | ReduceLROnPlateau ×0.5 |
| Seed | 42 |

## Evaluation

**Primary:** stratified **5-fold CV** on each dataset (`--cv`, default).  
**Supplementary:** chronological 80/20 (`--chrono`).

## Run

```powershell
python run_paper_experiment.py --require-real --cpu
```
