"""Run experiments on real NASA .mat data when available."""

import os
import time

import torch

from benchmark import benchmark_model, estimate_model_macs
from experiments.config import CHECKPOINT_DIR, NUM_CYCLES, PAPER_REFERENCE, SEQ_LEN
from experiments.io_utils import ensure_dirs, save_json
from experiments.report import build_summary_payload, print_comparison_report
from experiments.trainer import set_seed, train_msc_experiment, train_paper_experiment
from model import BatteryHealthPredictor
from model_paper import BatterySOHPredictorPaper
from preprocess import BatteryDatasetLoader
from preprocess_paper import PaperDatasetLoader


def _has_nasa_mat_files():
    nasa_dir = os.path.join(os.getcwd(), "data", "NASA")
    return os.path.isdir(nasa_dir) and any(f.lower().endswith(".mat") for f in os.listdir(nasa_dir))


def run_nasa_real_experiments(run_ablation=True):
    if not _has_nasa_mat_files():
        raise FileNotFoundError(
            "No NASA .mat files found. Run: python download_data.py --nasa"
        )

    set_seed()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "NASA"

    print("\n" + "=" * 96)
    print("NASA REAL-DATA EXPERIMENT (B0005/B0006/B0007/B0018 .mat files)")
    print(f"Device: {device.type.upper()} | Seed: 42")
    print("=" * 96)

    paper_features, paper_soh = PaperDatasetLoader.load_dataset(dataset, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN)
    msc_features, msc_soh, msc_rul = BatteryDatasetLoader.load_dataset(dataset, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN)

    print(f"\nReal NASA cycles loaded: {len(paper_soh)} (paper features), {len(msc_soh)} (MSc features)")

    paper_ckpt = os.path.join(CHECKPOINT_DIR, "paper_nasa_real.pt")
    paper_result = train_paper_experiment(
        BatterySOHPredictorPaper(input_features=3),
        paper_features,
        paper_soh,
        dataset,
        paper_ckpt,
    )

    msc_ckpt = os.path.join(CHECKPOINT_DIR, "msc_nasa_real.pt")
    msc_result = train_msc_experiment(
        BatteryHealthPredictor(input_features=3),
        msc_features,
        msc_soh,
        msc_rul,
        dataset,
        msc_ckpt,
        use_physics_loss=True,
    )

    ablation_results = []
    if run_ablation:
        ablation_ckpt = os.path.join(CHECKPOINT_DIR, "msc_ablation_nasa_real.pt")
        ablation_results.append(
            train_msc_experiment(
                BatteryHealthPredictor(input_features=3),
                msc_features,
                msc_soh,
                msc_rul,
                dataset,
                ablation_ckpt,
                use_physics_loss=False,
            )
        )

    model_paper = BatterySOHPredictorPaper().to(device)
    model_msc = BatteryHealthPredictor().to(device)
    power_bms_w = 0.103
    benchmark_stats = {
        "paper": {
            "params_m": sum(p.numel() for p in model_paper.parameters() if p.requires_grad) / 1e6,
            "latency_ms": benchmark_model(model_paper, device),
            "energy_mj": 0,
            "macs": estimate_model_macs("paper", SEQ_LEN),
        },
        "msc": {
            "params_m": sum(p.numel() for p in model_msc.parameters() if p.requires_grad) / 1e6,
            "latency_ms": benchmark_model(model_msc, device),
            "energy_mj": 0,
            "macs": estimate_model_macs("advanced", SEQ_LEN),
        },
        "published_transformer": PAPER_REFERENCE["transformer"],
        "published_paper_hybrid": PAPER_REFERENCE["paper_hybrid"],
    }
    benchmark_stats["paper"]["energy_mj"] = 0.103 * benchmark_stats["paper"]["latency_ms"]
    benchmark_stats["msc"]["energy_mj"] = 0.103 * benchmark_stats["msc"]["latency_ms"]

    summary = build_summary_payload([paper_result], [msc_result], benchmark_stats, ablation_results)
    summary["data_source"] = "NASA_real_mat"
    report_path = save_json(summary, "nasa_real_experiment_report.json")
    print_comparison_report([paper_result], [msc_result], benchmark_stats, ablation_results)
    print(f"\nReal NASA results saved to: {report_path}")
    return summary


if __name__ == "__main__":
    started = time.time()
    run_nasa_real_experiments(run_ablation=True)
    print(f"Runtime: {(time.time() - started) / 60:.1f} minutes")
