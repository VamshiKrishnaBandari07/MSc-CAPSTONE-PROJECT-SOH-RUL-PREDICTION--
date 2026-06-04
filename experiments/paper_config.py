"""Hyperparameters aligned with Rahman et al., Scientific Reports (2026)."""

# Feature engineering (Section 3 — Methodology)
PAPER_VOLTAGE_MIN = 2.5
PAPER_VOLTAGE_MAX = 4.2
PAPER_SEQ_LEN = 300  # fixed voltage grid points (paper: 300 between 2.5 V and 4.2 V)
SG_WINDOW = 15
SG_POLYORDER = 3

# Training (Table 2 narrative: ~180–220 epochs, early stopping, LR schedule 0.5)
PAPER_LEARNING_RATE = 1e-3
PAPER_WEIGHT_DECAY = 1e-5
PAPER_BATCH_SIZE = 16
PAPER_BATCH_SIZE_CPU = 4  # stable on CPU with 300-pt grid + ~0.39M params
PAPER_MAX_EPOCHS = 200
PAPER_EARLY_STOPPING_PATIENCE = 20
PAPER_LR_SCHEDULER_FACTOR = 0.5
PAPER_LR_SCHEDULER_PATIENCE = 5
PAPER_GRAD_CLIP_NORM = 5.0
PAPER_VOLTAGE_JITTER_V = 0.01  # ±10 mV augmentation on raw voltage before features
PAPER_FEATURE_NOISE = 0.003  # light train noise (paired with voltage jitter scale in trainer)
PAPER_DROPOUT = 0.15  # paper uses dropout regularization (default module 0.2)

# Stable long runs (avoid early exit on occasional NaN epochs)
PAPER_MIN_EPOCHS = 20
PAPER_MAX_NONFINITE_VAL_SKIP = 8
PAPER_USE_GLOBAL_SCALE = False  # fold-wise scaler in trainer prevents leakage + instability
PAPER_FOLD_WISE_SCALE = True

# Table 4: mean over five independent runs (paper Methods)
PAPER_INDEPENDENT_RUNS = 5
PAPER_RUN_SEEDS = [42, 43, 44, 45, 46]

# Evaluation (paper: stratified 5-fold CV; chronological 80/20 is supplementary)
PAPER_CV_FOLDS = 5
PAPER_DEFAULT_EVAL = "cv5"  # "cv5" | "chronological"

# Published targets (Table 4 — NASA PCoE pooled evaluation)
PAPER_TARGET_SOH_RMSE = 0.021
PAPER_TARGET_SOH_R2 = 0.983
PAPER_TARGET_PARAMS_M = 0.35

# Model width (scaled toward ~0.35M parameters vs lite demo variant)
PAPER_CNN_CHANNELS = 64
PAPER_TCN_CHANNELS = [64, 128]
PAPER_LSTM_HIDDEN = 128
PAPER_LSTM_LAYERS = 2
