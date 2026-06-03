"""
Experiment A — Paper reproduction (Scientific Reports 2026)

Reference: https://doi.org/10.1038/s41598-026-39911-8

Replicates the published hybrid CNN–TCN–LSTM–Attention SOH estimator using:
  - ICA (dQ/dV), DV (dV/dQ), DC (dI/dV) on a 300-point voltage grid (2.5–4.2 V)
  - Savitzky–Golay denoising (window=15, order=3)
  - MSE loss, ~200 epochs with early stopping, gradient clipping, LR scheduling

Target metrics (paper Table 4): SOH RMSE ≈ 0.021, R² ≈ 0.983

Usage:
  python download_data.py --all
  python run_paper_experiment.py
"""

import os
import time

import torch

from experiments.config import CHECKPOINT_DIR, DATASETS, PAPER_REFERENCE
from experiments.io_utils import ensure_dirs, save_json
from experiments.paper_config import PAPER_SEQ_LEN, PAPER_TARGET_SOH_RMSE
from experiments.provenance import detect_data_sources, experiment_config_snapshot
from experiments.report import print_comparison_report
from experiments.trainer import set_seed, train_paper_experiment
from model_paper import build_paper_model
from preprocess_paper import PaperDatasetLoader


def run_paper_experiment(datasets=None):
    datasets = datasets or DATASETS
    set_seed()
    ensure_dirs()

    print("\n" + "=" * 96)
    print("EXPERIMENT A — PAPER REPRODUCTION")
    print("Scientific Reports (2026) | DOI: 10.1038/s41598-026-39911-8")
    print(f"Features: ICA + DV + DC | Voltage grid: {PAPER_SEQ_LEN} pts | Target SOH RMSE: {PAPER_TARGET_SOH_RMSE}")
    print("=" * 96)

    results = []
    for dataset in datasets:
        print(f"\n{'#' * 96}\n# PAPER DATASET: {dataset}\n{'#' * 96}")
        features, soh = PaperDatasetLoader.load_dataset(dataset, seq_len=PAPER_SEQ_LEN)
        model = build_paper_model(seq_len=PAPER_SEQ_LEN)
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"[Paper | {dataset}] Model parameters: {params_m:.4f} M")

        checkpoint = os.path.join(CHECKPOINT_DIR, f"paper_{dataset.lower()}.pt")
        result = train_paper_experiment(
            model, features, soh, dataset, checkpoint, use_paper_protocol=True
        )
        results.append(result)

    payload = {
        "experiment": "A_paper_reproduction",
        "paper_doi": "10.1038/s41598-026-39911-8",
        "paper_target_soh_rmse": PAPER_TARGET_SOH_RMSE,
        "methodology": {
            "features": ["ICA_dQdV", "DV_dVdQ", "DC_dIdV"],
            "voltage_grid": f"{PAPER_SEQ_LEN} points, 2.5-4.2 V",
            "denoising": "Savitzky-Golay window=15 order=3",
            "loss": "MSE",
            "outputs": "SOH only",
        },
        "results": results,
        "data_sources": detect_data_sources(),
        "experiment_config": experiment_config_snapshot(),
        "published_reference": PAPER_REFERENCE,
    }
    report_path = save_json(payload, "paper_experiment_report.json")

    print("\n--- Paper Experiment Summary vs published hybrid (RMSE 0.021) ---")
    for result in results:
        m = result["metrics"]
        gap = ((m["rmse"] - PAPER_TARGET_SOH_RMSE) / PAPER_TARGET_SOH_RMSE) * 100
        print(
            f"{result['dataset']:<8} | SOH RMSE: {m['rmse']:.4f} ({gap:+.1f}% vs paper) | "
            f"R2: {m['r2']:.4f} | MAE: {m['mae']:.4f} | epoch {result['best_epoch']}"
        )
    print(f"\nFull report: {report_path}")
    return payload


if __name__ == "__main__":
    started = time.time()
    run_paper_experiment()
    print(f"\nPaper experiment runtime: {(time.time() - started) / 60:.1f} minutes")
