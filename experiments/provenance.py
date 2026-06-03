"""Detect whether real dataset files are available for each experiment path."""

import os

from experiments.config import DATASETS
from experiments.nasa_loader import count_nasa_discharge_cycles


def _has_nasa_mat_files():
    nasa_dir = os.path.join(os.getcwd(), "data", "NASA")
    return os.path.isdir(nasa_dir) and any(f.lower().endswith(".mat") for f in os.listdir(nasa_dir))


def _has_oxford_mat():
    path = os.path.join(os.getcwd(), "data", "Oxford", "Oxford_Battery_Degradation_Dataset_1.mat")
    return os.path.isfile(path) and os.path.getsize(path) > 1_000_000


def _has_calce_cells():
    calce_dir = os.path.join(os.getcwd(), "data", "CALCE")
    if not os.path.isdir(calce_dir):
        return False
    for entry in os.listdir(calce_dir):
        nested = os.path.join(calce_dir, entry, entry)
        if os.path.isdir(nested) and any(f.lower().endswith((".xls", ".xlsx")) for f in os.listdir(nested)):
            return True
    return False


def detect_data_sources():
    sources = {}
    for dataset in DATASETS:
        if dataset == "NASA" and _has_nasa_mat_files():
            cycle_count = count_nasa_discharge_cycles(os.path.join("data", "NASA"))
            label = f"real_nasa_mat ({cycle_count} discharge cycles)"
        elif dataset == "Oxford" and _has_oxford_mat():
            from experiments.oxford_loader import count_oxford_cycles

            cycle_count = count_oxford_cycles(os.path.join("data", "Oxford"))
            label = f"real_oxford_mat ({cycle_count} characterisation cycles)"
        elif dataset == "CALCE" and _has_calce_cells():
            from experiments.calce_loader import count_calce_cycles

            cycle_count = count_calce_cycles(os.path.join("data", "CALCE"))
            label = f"real_calce_xlsx ({cycle_count} discharge cycles)"
        else:
            label = "synthetic_fallback"

        sources[dataset] = {
            "experiment_a_paper": label,
            "experiment_b_msc": label,
        }
    return sources


def experiment_config_snapshot():
    from experiments.config import (
        BATCH_SIZE,
        EARLY_STOPPING_PATIENCE,
        EDGE_POWER_WATTS,
        LEARNING_RATE,
        MAX_EPOCHS,
        MSC_DEFAULTS,
        NUM_CYCLES,
        PAPER_REPORTED_EPOCHS,
        RANDOM_SEED,
        SEQ_LEN,
        TRAIN_RATIO,
    )

    return {
        "random_seed": RANDOM_SEED,
        "seq_len": SEQ_LEN,
        "num_cycles_synthetic_default": NUM_CYCLES,
        "train_ratio": TRAIN_RATIO,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs_local": MAX_EPOCHS,
        "paper_reported_epochs": PAPER_REPORTED_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "msc_loss_weights": MSC_DEFAULTS,
        "msc_early_stop": "soh_rmse + rul_weight * (rul_rmse / max_rul)",
        "edge_power_watts": EDGE_POWER_WATTS,
        "energy_formula": "energy_mJ = latency_ms * edge_power_watts",
    }
