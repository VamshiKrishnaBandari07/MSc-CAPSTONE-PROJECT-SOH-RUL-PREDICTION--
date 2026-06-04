# Paper reproduction module

Official entry point for reproducing:

**Rahman et al.,** *Scientific Reports* **16**, 9871 (2026).  
DOI: [10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)

## Run

From repository root:

```powershell
python paper_reproduction/run.py --require-real --cpu
```

Equivalent to:

```powershell
python run_paper_experiment.py --require-real --cpu
```

## Implementation map

| Paper component | Code |
|:---|:---|
| ICA / DV / DC features | `experiments/paper_preprocessing.py` |
| 300-pt voltage grid | `experiments/paper_config.py` |
| CNN–TCN–LSTM–Attention | `model_paper.py` |
| 5-fold CV | `experiments/cv.py` |
| Training | `experiments/trainer.py` → `train_paper_experiment` |
| Results | `results/paper_experiment_report.json` |
