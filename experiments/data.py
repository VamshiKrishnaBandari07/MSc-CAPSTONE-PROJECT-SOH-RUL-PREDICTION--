"""Shared SOH/RUL label generation for fair cross-experiment comparison."""

import numpy as np

from experiments.config import RANDOM_SEED

DATASET_SEED_OFFSET = {"NASA": 0, "Oxford": 100, "CALCE": 200}


def dataset_rng(dataset_name):
    offset = DATASET_SEED_OFFSET.get(dataset_name, 0)
    return np.random.default_rng(RANDOM_SEED + offset)


def generate_shared_labels(dataset_name="NASA", num_cycles=150):
    """
    Produces consistent SOH and RUL trajectories per dataset.
    Both Experiment A (paper) and Experiment B (MSc) use these labels.
    """
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
    soh_values = []

    for cycle in range(num_cycles):
        degrad = capacity_fade_rate * (cycle / num_cycles) ** 1.3 + 0.05 * (cycle / num_cycles)
        recovery = 0.012 * np.sin(cycle / 5.0) if cycle % 12 == 0 else 0.0
        soh = 1.0 - degrad + recovery
        soh_values.append(float(np.clip(soh, 0.4, 1.0)))

    soh_array = np.array(soh_values, dtype=np.float32)
    eol_idx = next((i for i, s in enumerate(soh_array) if s <= eol_threshold), len(soh_array))
    rul_array = np.array([max(0, eol_idx - i) for i in range(len(soh_array))], dtype=np.float32)

    return soh_array, rul_array, eol_threshold
