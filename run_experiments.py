"""
MSc Capstone — Unified Experiment Runner

Experiment A: Paper-exact reproduction (SOH, MSE) on NASA / Oxford / CALCE
Experiment B: MSc extension (joint SOH + RUL + physics-informed loss)
Experiment C: Ablation (MSc without monotonicity penalty)
Computational benchmark: latency, parameters, energy
"""

import os
import time

from benchmark import benchmark_model, estimate_model_macs
from experiments.cli import parse_runtime_args
from experiments.runtime import configure_runtime, get_device
from experiments.config import CHECKPOINT_DIR, DATASETS, EDGE_POWER_WATTS, NUM_CYCLES, PAPER_REFERENCE, SEQ_LEN
from experiments.paper_config import PAPER_SEQ_LEN
from experiments.io_utils import ensure_dirs, save_json
from experiments.report import build_summary_payload, print_comparison_report
from experiments.trainer import set_seed, train_msc_experiment, train_paper_experiment
from model import BatteryHealthPredictor
from model_paper import build_paper_model
from preprocess import BatteryDatasetLoader
from preprocess_paper import PaperDatasetLoader
from experiments.provenance import detect_data_sources, experiment_config_snapshot


def _run_benchmarks(device):
    model_paper = build_paper_model(seq_len=PAPER_SEQ_LEN).to(device)
    model_msc = BatteryHealthPredictor().to(device)

    params_paper = sum(p.numel() for p in model_paper.parameters() if p.requires_grad)
    params_msc = sum(p.numel() for p in model_msc.parameters() if p.requires_grad)

    latency_paper = benchmark_model(model_paper, device, seq_len=PAPER_SEQ_LEN)
    latency_msc = benchmark_model(model_msc, device, seq_len=SEQ_LEN)

    return {
        "paper": {
            "params_m": params_paper / 1e6,
            "latency_ms": latency_paper,
            "energy_mj": EDGE_POWER_WATTS * latency_paper,
            "macs": estimate_model_macs("paper", PAPER_SEQ_LEN),
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


def run_all_experiments(run_ablation=True, force_cpu=False, batch_size=None, paper_max_epochs=None, msc_max_epochs=None, datasets=None):
    datasets = datasets or DATASETS
    set_seed()
    ensure_dirs()
    device = configure_runtime(force_cpu=force_cpu)

    print("\n" + "=" * 96)
    print("MSc CAPSTONE - FULL EXPERIMENTAL SUITE")
    print("Experiment A: Paper reproduction (ICA+DV+DC, grid=%d) | Experiment B: MSc extension (ICA+DV+DCA)" % PAPER_SEQ_LEN)
    print(f"Device: {device.type.upper()} | Datasets: {', '.join(datasets)} | Seed: 42")
    if device.type == "cpu":
        print("CPU mode — full suite may take several hours. Use --dataset NASA for a faster run.")
    print("=" * 96)

    paper_results = []
    msc_results = []
    ablation_results = []

    for dataset in datasets:
        print(f"\n{'#' * 96}\n# DATASET: {dataset}\n{'#' * 96}")

        paper_features, paper_soh = PaperDatasetLoader.load_dataset(
            dataset, num_cycles=NUM_CYCLES, seq_len=PAPER_SEQ_LEN
        )
        msc_features, msc_soh, msc_rul = BatteryDatasetLoader.load_dataset(
            dataset, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN
        )

        paper_ckpt = os.path.join(CHECKPOINT_DIR, f"paper_{dataset.lower()}.pt")
        paper_model = build_paper_model(seq_len=PAPER_SEQ_LEN)
        paper_result = train_paper_experiment(
            paper_model,
            paper_features,
            paper_soh,
            dataset,
            paper_ckpt,
            epochs=paper_max_epochs,
            batch_size=batch_size,
            use_paper_protocol=True,
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
            epochs=msc_max_epochs,
            batch_size=batch_size,
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
                epochs=msc_max_epochs,
                batch_size=batch_size,
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
    args = parse_runtime_args("Full capstone suite — Experiments A + B + C (CPU/GPU)")
    started = time.time()
    datasets = (args.dataset,) if args.dataset else DATASETS
    run_all_experiments(
        run_ablation=True,
        force_cpu=args.cpu,
        batch_size=args.batch_size,
        paper_max_epochs=args.max_epochs,
        msc_max_epochs=args.max_epochs,
        datasets=datasets,
    )
    elapsed = time.time() - started
    print(f"Total experiment runtime: {elapsed / 60:.1f} minutes")
