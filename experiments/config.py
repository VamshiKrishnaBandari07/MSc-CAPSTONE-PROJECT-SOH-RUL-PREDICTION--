import os

RANDOM_SEED = 42
DATASETS = ("NASA", "Oxford", "CALCE")
TRAIN_RATIO = 0.8
EARLY_STOPPING_PATIENCE = 5

CHECKPOINT_DIR = os.path.join("checkpoints")
RESULTS_DIR = os.path.join("results")

PAPER_REFERENCE = {
    "transformer": {
        "label": "Transformer (paper baseline)",
        "params_m": 1.25,
        "latency_ms": 12.4,
        "energy_mj": 0.86,
        "soh_rmse": 0.038,
        "targets": "SOH",
    },
    "paper_hybrid": {
        "label": "Paper hybrid CNN-TCN-LSTM-Attn",
        "params_m": 0.35,
        "latency_ms": 6.1,
        "energy_mj": 0.63,
        "soh_rmse": 0.021,
        "targets": "SOH",
    },
}
