import os

RANDOM_SEED = 42
DATASETS = ("NASA", "Oxford", "CALCE")
SEQ_LEN = 100
NUM_CYCLES = 150
TRAIN_RATIO = 0.8
BATCH_SIZE = 8
BATCH_SIZE_CPU = 4  # smaller batches for stable CPU training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5

# Paper may report longer schedules (e.g. 300 epochs). This repo uses early stopping by default.
PAPER_REPORTED_EPOCHS = 300

# Edge BMS power model used for energy estimates: energy_mJ = latency_ms * EDGE_POWER_WATTS
EDGE_POWER_WATTS = 0.103

CHECKPOINT_DIR = os.path.join("checkpoints")
RESULTS_DIR = os.path.join("results")

# Published reference values from Scientific Reports (2026) paper Table benchmarks.
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

MSC_DEFAULTS = {
    "rul_weight": 0.5,
    "monotonicity_weight": 0.25,
}
