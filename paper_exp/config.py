from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PaperExperimentConfig:
    """Hyperparameters reported for the CNN-TCN-LSTM-Attention paper setup."""

    datasets: Tuple[str, ...] = ("NASA", "Oxford", "CALCE")
    supported_datasets: Tuple[str, ...] = ("NASA", "Oxford", "CALCE", "KaggleSDG7")
    seq_len: int = 128
    cycles_per_dataset: int = 3600
    n_folds: int = 5
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 1e-5
    dropout: float = 0.2
    early_stopping_patience: int = 20
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    random_seed: int = 42
    edge_power_watts: float = 0.103


PAPER_TARGETS = {
    "soh_rmse": 0.021,
    "soh_r2": 0.983,
    "parameters_millions": 0.35,
    "latency_ms": 6.1,
    "energy_mj": 0.63,
}

