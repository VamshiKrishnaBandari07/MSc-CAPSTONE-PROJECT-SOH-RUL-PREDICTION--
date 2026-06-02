"""
MSc Capstone — Unified Experiment Runner

Experiment A: Paper-exact reproduction (SOH, MSE) on NASA / Oxford / CALCE
Experiment B: MSc extension (joint SOH + RUL + physics-informed loss)
Experiment C: Ablation (MSc without monotonicity penalty)
Computational benchmark: latency, parameters, energy
"""

import os
import time

import torch

from benchmark import benchmark_model, estimate_model_macs
from experiments.config import CHECKPOINT_DIR, DATASETS, EDGE_POWER_WATTS, NUM_CYCLES, PAPER_REFERENCE, SEQ_LEN
from experiments.io_utils import ensure_dirs, save_json
from experiments.report import build_summary_payload, print_comparison_report
from experiments.trainer import set_seed, train_msc_experiment, train_paper_experiment
from model import BatteryHealthPredictor
from model_paper import BatterySOHPredictorPaper
from preprocess import BatteryDatasetLoader
from preprocess_paper import PaperDatasetLoader
from experiments.provenance import detect_data_sources, experiment_config_snapshot


def _run_benchmarks(device):
    model_paper = BatterySOHPredictorPaper().to(device)
    model_msc = BatteryHealthPredictor().to(device)

    params_paper = sum(p.numel() for p in model_paper.parameters() if p.requires_grad)
    params_msc = sum(p.numel() for p in model_msc.parameters() if p.requires_grad)

    latency_paper = benchmark_model(model_paper, device)
    latency_msc = benchmark_model(model_msc, device)

    return {
        "paper": {
            "params_m": params_paper / 1e6,
            "latency_ms": latency_paper,
            "energy_mj": EDGE_POWER_WATTS * latency_paper,
            "macs": estimate_model_macs("paper", SEQ_LEN),
        },
        "msc": {
            "params_m": params_msc / 1e6,
            "latency_ms": latency_msc,
            "energy_mj": EDGE_POWER_WATTS * latency_msc,
            "macs": estimate_model_macs("advanced", SEQ_LEN),
        },
        "published_transformer": PAPER_REFERENCE["transformer"],
        "published_paper_hybrid": PAPER_REFERENCE["paper_hybrid"],
    }


def run_all_experiments(run_ablation=True):
    set_seed()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 96)
    print("MSc CAPSTONE - FULL EXPERIMENTAL SUITE")
    print(f"Device: {device.type.upper()} | Datasets: {', '.join(DATASETS)} | Seed: 42")
    print("=" * 96)

    paper_results = []
    msc_results = []
    ablation_results = []

    for dataset in DATASETS:
        print(f"\n{'#' * 96}\n# DATASET: {dataset}\n{'#' * 96}")

        paper_features, paper_soh = PaperDatasetLoader.load_dataset(
            dataset, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN
        )
        msc_features, msc_soh, msc_rul = BatteryDatasetLoader.load_dataset(
            dataset, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN
        )

        paper_ckpt = os.path.join(CHECKPOINT_DIR, f"paper_{dataset.lower()}.pt")
        paper_model = BatterySOHPredictorPaper(input_features=3)
        paper_result = train_paper_experiment(
            paper_model,
            paper_features,
            paper_soh,
            dataset,
            paper_ckpt,
        )
        paper_results.append(paper_result)

        msc_ckpt = os.path.join(CHECKPOINT_DIR, f"msc_{dataset.lower()}.pt")
        msc_model = BatteryHealthPredictor(input_features=3)
        msc_result = train_msc_experiment(
            msc_model,
            msc_features,
            msc_soh,
            msc_rul,
            dataset,
            msc_ckpt,
            use_physics_loss=True,
        )
        msc_results.append(msc_result)

        if run_ablation:
            ablation_ckpt = os.path.join(CHECKPOINT_DIR, f"msc_ablation_{dataset.lower()}.pt")
            ablation_model = BatteryHealthPredictor(input_features=3)
            ablation_result = train_msc_experiment(
                ablation_model,
                msc_features,
                msc_soh,
                msc_rul,
                dataset,
                ablation_ckpt,
                use_physics_loss=False,
            )
            ablation_results.append(ablation_result)

    print("\nRunning computational benchmark...")
    benchmark_stats = _run_benchmarks(device)

    summary = build_summary_payload(paper_results, msc_results, benchmark_stats, ablation_results)
    summary["data_sources"] = detect_data_sources()
    summary["experiment_config"] = experiment_config_snapshot()
    report_path = save_json(summary, "experiment_comparison_report.json")
    print_comparison_report(paper_results, msc_results, benchmark_stats, ablation_results)
    print(f"Full results saved to: {report_path}")
    print(f"Model checkpoints saved in: {CHECKPOINT_DIR}/")

    return summary


if __name__ == "__main__":
    started = time.time()
    run_all_experiments(run_ablation=True)
    elapsed = time.time() - started
    print(f"Total experiment runtime: {elapsed / 60:.1f} minutes")
