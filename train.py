"""Experiment B — MSc extension (joint SOH + RUL + physics-informed loss)."""

import os

from experiments.config import CHECKPOINT_DIR, DATASETS, NUM_CYCLES, SEQ_LEN
from experiments.io_utils import ensure_dirs, save_json
from experiments.trainer import set_seed, train_msc_experiment
from model import BatteryHealthPredictor
from preprocess import BatteryDatasetLoader


def run_msc_experiments(use_physics_loss=True):
    set_seed()
    ensure_dirs()
    results = []

    label = "Physics-Informed Joint" if use_physics_loss else "Ablation (No Physics)"
    print("\n" + "=" * 80)
    print(f"EXPERIMENT B — MSc Extension ({label})")
    print("=" * 80)

    for dataset in DATASETS:
        features, soh, rul = BatteryDatasetLoader.load_dataset(dataset, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN)
        suffix = "" if use_physics_loss else "_ablation"
        checkpoint = os.path.join(CHECKPOINT_DIR, f"msc{suffix}_{dataset.lower()}.pt")
        model = BatteryHealthPredictor(input_features=3)
        result = train_msc_experiment(
            model,
            features,
            soh,
            rul,
            dataset,
            checkpoint,
            use_physics_loss=use_physics_loss,
        )
        results.append(result)

    filename = "msc_experiment_results.json" if use_physics_loss else "msc_ablation_results.json"
    save_json(results, filename)

    print("\n--- MSc Experiment Summary ---")
    for result in results:
        soh_m = result["metrics"]["soh"]
        rul_m = result["metrics"]["rul"]
        print(
            f"{result['dataset']:<8} | SOH RMSE: {soh_m['rmse']:.4f} | "
            f"RUL RMSE: {rul_m['rmse']:.2f} | Mono: {soh_m['mono_violation_rate']:.2%}"
        )

    return results


if __name__ == "__main__":
    run_msc_experiments(use_physics_loss=True)
