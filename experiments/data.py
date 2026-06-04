"""Synthetic SOH labels for fallback when raw datasets are unavailable."""

from __future__ import annotations

import numpy as np

from experiments.config import RANDOM_SEED

DATASET_SEED_OFFSET = {"NASA": 0, "Oxford": 100, "CALCE": 200}


def dataset_rng(dataset_name: str) -> np.random.Generator:
    """Deterministic RNG per dataset (seed 42 + offset)."""
    offset = DATASET_SEED_OFFSET.get(dataset_name, 0)
    return np.random.default_rng(RANDOM_SEED + offset)


def generate_shared_labels(dataset_name: str = "NASA", num_cycles: int = 150) -> tuple:
    """Synthetic SOH/RUL trajectories for ``require_real=False`` fallback only."""
    if dataset_name == "NASA":
        eol_threshold = 0.70
        capacity_fade_rate = 0.28
    elif dataset_name == "Oxford":
        eol_threshold = 0.80
        capacity_fade_rate = 0.20
    else:
        eol_threshold = 0.75
        capacity_fade_rate = 0.25

    rng = dataset_rng(dataset_name)
    soh = np.clip(1.0 - capacity_fade_rate * np.linspace(0, 1, num_cycles), eol_threshold, 1.0)
    soh += rng.normal(0, 0.01, size=num_cycles)
    soh = np.clip(soh, 0, 1)
    rul = np.arange(num_cycles - 1, -1, -1, dtype=np.float32)
    return soh.astype(np.float32), rul.astype(np.float32), eol_threshold
