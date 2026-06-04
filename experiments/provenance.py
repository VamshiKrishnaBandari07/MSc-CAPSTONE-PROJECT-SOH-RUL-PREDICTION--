"""Detect whether real dataset files are available."""

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
            label = "missing_or_synthetic"
        sources[dataset] = label
    return sources


def experiment_config_snapshot():
    from experiments.paper_config import (
        PAPER_BATCH_SIZE,
        PAPER_CV_FOLDS,
        PAPER_DEFAULT_EVAL,
        PAPER_MAX_EPOCHS,
        PAPER_SEQ_LEN,
        PAPER_TARGET_SOH_RMSE,
    )
    from experiments.config import RANDOM_SEED

    return {
        "random_seed": RANDOM_SEED,
        "paper_reproduction": {
            "features": ["ICA_dQdV", "DV_dVdQ", "DC_dIdV"],
            "voltage_grid_points": PAPER_SEQ_LEN,
            "voltage_range_v": [2.5, 4.2],
            "sg_filter": {"window": 15, "polyorder": 3},
            "max_epochs": PAPER_MAX_EPOCHS,
            "batch_size": PAPER_BATCH_SIZE,
            "loss": "MSE",
            "target_soh_rmse": PAPER_TARGET_SOH_RMSE,
            "eval_protocol_default": PAPER_DEFAULT_EVAL,
            "cv_folds": PAPER_CV_FOLDS,
            "reference_doi": "10.1038/s41598-026-39911-8",
        },
    }
