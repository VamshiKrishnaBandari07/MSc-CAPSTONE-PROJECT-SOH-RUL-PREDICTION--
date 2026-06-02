# Paper Experiment Metric Comparison

This report summarises the paper experiment results obtained on the branch and compares them against the metrics reported in the referenced Scientific Reports paper.

Referenced paper:

> *Deep learning-based battery health prediction for enhancing electric vehicle performance*  
> DOI: `10.1038/s41598-026-39911-8`

Dataset used for the local reproduction attempts:

```text
KaggleSDG7
https://www.kaggle.com/datasets/drtawfikrrahman/deep-learning-ev-battery-pack-diagnostics-sdg-7
```

The Kaggle dataset was converted into PyTorch-ready `.npz` tensors using `paper_exp.prepare_data`. The converted files are ignored by git because they are generated data artefacts.

---

## 1. Paper target metrics

| Metric | Paper reported value |
| --- | ---: |
| SOH RMSE | `0.021` |
| SOH R2 | `0.983` |
| Parameters | `~0.35M` |
| Latency | `6.1 ms/sample` |
| Energy | `0.63 mJ/sample` |

---

## 2. Local experiment progression

| Stage | What changed | Epochs | Fold strategy | Mean SOH RMSE | Mean SOH R2 | Comment |
| --- | --- | ---: | --- | ---: | ---: | --- |
| 1 | Initial Kaggle DV/DC/ICA voltage-grid reproduction | 5 | `stratified_random` | `0.1314` | `0.0194` | Code ran correctly, but accuracy was far from the paper. |
| 2 | Used Kaggle-provided ICA/DVA plus voltage with global scaling | 5 | `stratified_random` | `0.1274` | `0.0779` | Slight improvement, still far from target. |
| 3 | Changed representation to cycle-to-cycle sequences | 10 | `stratified_random` | `0.0487` | `0.8346` | Major improvement; confirms the LSTM benefits from cycle-level ageing sequences. |
| 4 | Cycle-to-cycle sequences, lower LR, gradient clipping | 20 | `stratified_random` | `0.0394` | `0.8862` | Best stable 5-fold mean so far. |
| 5 | Best individual fold from Stage 4 | 20 | `stratified_random` | `0.0176` | `0.9792` | One fold reached paper-level RMSE, but the mean did not. |

---

## 3. Final comparison against paper

| Metric | Paper target | Best stable 5-fold local result | Gap |
| --- | ---: | ---: | ---: |
| SOH RMSE | `0.021` | `0.0394` | `+0.0184` |
| SOH R2 | `0.983` | `0.8862` | `-0.0968` |
| Parameters | `~0.35M` | `0.356M` | close match |

The local implementation **does not yet reproduce the exact paper mean metric**. It does, however, show meaningful progress:

- initial RMSE improved from `0.1314` to `0.0394`;
- initial R2 improved from `0.0194` to `0.8862`;
- model size matches the paper closely (`0.356M` vs `~0.35M`);
- one fold achieved paper-level RMSE (`0.0176`), but this is not sufficient to claim full reproduction.

---

## 4. Commands used for the best stable local run

Prepare the cycle-to-cycle Kaggle representation:

```bash
python3 -m paper_exp.prepare_data \
  --datasets KaggleSDG7 \
  --raw-dir data \
  --output-dir data/processed \
  --seq-len 32 \
  --kaggle-cycle-sequences
```

Run the best stable paper experiment attempt:

```bash
python3 -m paper_exp.train \
  --datasets KaggleSDG7 \
  --raw-dir data \
  --output-dir paper_exp/outputs/kaggle_cycle_sequence/results_20epoch_lr5e4 \
  --require-real-data \
  --seq-len 32 \
  --n-folds 5 \
  --fold-strategy stratified_random \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --grad-clip 1.0
```

---

## 5. Interpretation

The paper experiment is running correctly and the architecture parameter count is aligned with the paper. However, the exact published paper accuracy has not been reproduced as a 5-fold mean in this repository.

The strongest evidence so far is that the cycle-to-cycle representation substantially improves performance, which supports the interpretation that the recurrent component should model degradation over cycles rather than voltage samples inside a single cycle.

For dissertation presentation, the recommended wording is:

> "The implemented paper-aligned CNN-TCN-LSTM-Attention model reproduced the paper's parameter scale and achieved improving SOH performance on the Kaggle SDG7 dataset. The best stable 5-fold result was RMSE = 0.0394 and R2 = 0.8862, while one fold reached RMSE = 0.0176. Therefore, the paper's reported mean RMSE = 0.021 was not fully reproduced under the available local CPU-based protocol. The gap is attributed to differences in split protocol, exact preprocessing, training duration, and hardware."

---

## 6. Next steps to target the exact paper metric

Priority improvements:

1. Run the cycle-to-cycle setup for the full 300-epoch schedule on GPU.
2. Confirm the exact paper train/validation split protocol.
3. Add group-aware evaluation by cell/pack to separate strict generalisation from easier random cycle splits.
4. Save checkpoints and prediction CSVs for each fold.
5. Add per-fold plots of predicted vs true SOH.
6. Implement the paper's reported baselines if claiming comparative superiority.

---

## 7. Current honest conclusion

Status:

- Code runs: **yes**
- Dataset conversion runs: **yes**
- Paper architecture parameter scale matches: **yes**
- Exact paper mean RMSE/R2 reproduced: **no**
- Best local stable mean: **RMSE `0.0394`, R2 `0.8862`**
- Best local individual fold: **RMSE `0.0176`, R2 `0.9792`**

This report should be presented honestly as a strong reproduction attempt rather than an exact replication of the published paper metric.

