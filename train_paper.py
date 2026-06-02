"""Experiment A — Paper-exact reproduction (SOH only, MSE loss).

NASA: loads real B0005–B0018 .mat files when present in data/NASA/.
Oxford/CALCE: synthetic fallback only.

For thesis-quality real NASA paper reproduction, prefer: python run_nasa_real.py
See docs/PAPER_EXPERIMENT_METRIC_COMPARISON.md for metric history and limitations.
"""

import os

from experiments.config import CHECKPOINT_DIR, DATASETS, NUM_CYCLES, SEQ_LEN
from experiments.io_utils import ensure_dirs, save_json
from experiments.trainer import set_seed, train_paper_experiment
from model_paper import BatterySOHPredictorPaper
from preprocess_paper import PaperDatasetLoader


def run_paper_experiments():
    set_seed()
    ensure_dirs()
    results = []

    print("\n" + "=" * 80)
    print("EXPERIMENT A — Paper-Exact Reproduction")
    print("=" * 80)

    for dataset in DATASETS:
        features, soh = PaperDatasetLoader.load_dataset(dataset, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN)
        checkpoint = os.path.join(CHECKPOINT_DIR, f"paper_{dataset.lower()}.pt")
        model = BatterySOHPredictorPaper(input_features=3)
        result = train_paper_experiment(model, features, soh, dataset, checkpoint)
        results.append(result)

    save_json(results, "paper_experiment_results.json")

    print("\n--- Paper Experiment Summary ---")
    for result in results:
        m = result["metrics"]
        print(
            f"{result['dataset']:<8} | SOH RMSE: {m['rmse']:.4f} | "
            f"MAE: {m['mae']:.4f} | R2: {m['r2']:.4f} | Mono: {m['mono_violation_rate']:.2%}"
        )

    return results


if __name__ == "__main__":
    run_paper_experiments()
