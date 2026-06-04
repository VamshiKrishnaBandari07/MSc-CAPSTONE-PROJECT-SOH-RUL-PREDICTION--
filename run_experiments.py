"""
MSc Capstone — Unified Experiment Runner

PHASE 1 — Experiment A: Paper-exact reproduction (run first via run_paper_experiment.py)
PHASE 2 — Experiment B: MSc extension (joint SOH + RUL + physics-informed loss)
PHASE 2 — Experiment C: Ablation (MSc without monotonicity penalty)

Recommended order:
  1. python run_paper_experiment.py --require-real --cpu
  2. python run_experiments.py --msc-only --require-real --cpu
Or full suite: python run_experiments.py --require-real --cpu
"""

import os
import sys
import time
import json

from benchmark import benchmark_model, estimate_model_macs
from experiments.cli import parse_runtime_args, paper_eval_protocol
from experiments.runtime import configure_runtime, get_device
from experiments.config import CHECKPOINT_DIR, DATASETS, EDGE_POWER_WATTS, NUM_CYCLES, PAPER_REFERENCE, RESULTS_DIR, SEQ_LEN
from experiments.paper_config import PAPER_DEFAULT_EVAL, PAPER_SEQ_LEN
from experiments.io_utils import ensure_dirs, save_json
from experiments.report import build_summary_payload, print_comparison_report
from experiments.trainer import set_seed, train_msc_experiment, train_paper_experiment
from model import BatteryHealthPredictor
from model_paper import build_paper_model
from preprocess import BatteryDatasetLoader
from preprocess_paper import PaperDatasetLoader
from experiments.provenance import detect_data_sources, experiment_config_snapshot


def _load_paper_results_from_phase1():
    """When running --msc-only, merge Experiment A results from Phase 1 report."""
    path = os.path.join(RESULTS_DIR, "paper_experiment_report.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", [])
    if results:
        print(f"Loaded {len(results)} Phase 1 paper result(s) from {path}")
    return results


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


def run_all_experiments(
    run_ablation=True,
    force_cpu=False,
    batch_size=None,
    paper_max_epochs=None,
    msc_max_epochs=None,
    datasets=None,
    require_real=False,
    run_paper=True,
    run_msc=True,
    paper_eval_protocol=None,
):
    if not run_paper and not run_msc:
        raise ValueError("At least one of run_paper or run_msc must be True")

    datasets = datasets or DATASETS
    paper_eval_protocol = paper_eval_protocol or PAPER_DEFAULT_EVAL
    set_seed()
    ensure_dirs()
    device = configure_runtime(force_cpu=force_cpu)

    print("\n" + "=" * 96)
    print("MSc CAPSTONE — EXPERIMENTAL SUITE")
    if run_paper and run_msc:
        print("PHASE 1 (A): Paper reproduction → PHASE 2 (B+C): MSc extension")
    elif run_paper:
        print("PHASE 1 ONLY — Experiment A (paper reproduction)")
    else:
        print("PHASE 2 ONLY — Experiments B + C (MSc extension; assumes A completed)")
    print(f"Paper eval: {paper_eval_protocol} | Device: {device.type.upper()} | Datasets: {', '.join(datasets)}")
    print("=" * 96)

    paper_results = []
    msc_results = []
    ablation_results = []

    for dataset in datasets:
        print(f"\n{'#' * 96}\n# DATASET: {dataset}\n{'#' * 96}")

        if run_paper:
            paper_features, paper_soh = PaperDatasetLoader.load_dataset(
                dataset, num_cycles=NUM_CYCLES, seq_len=PAPER_SEQ_LEN, require_real=require_real
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
                eval_protocol=paper_eval_protocol,
            )
            paper_results.append(paper_result)

        if run_msc:
            msc_features, msc_soh, msc_rul = BatteryDatasetLoader.load_dataset(
                dataset, num_cycles=NUM_CYCLES, seq_len=SEQ_LEN
            )
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

    benchmark_stats = None
    if run_paper or run_msc:
        print("\nRunning computational benchmark...")
        benchmark_stats = _run_benchmarks(device)

    if run_msc and not run_paper and not paper_results:
        paper_results = _load_paper_results_from_phase1()

    phase1_eval_protocol = None
    paper_report_path = os.path.join(RESULTS_DIR, "paper_experiment_report.json")
    if not run_paper and os.path.isfile(paper_report_path):
        with open(paper_report_path, encoding="utf-8") as handle:
            phase1_eval_protocol = json.load(handle).get("eval_protocol")

    summary = build_summary_payload(paper_results, msc_results, benchmark_stats or {}, ablation_results)
    summary["data_sources"] = detect_data_sources()
    summary["experiment_config"] = experiment_config_snapshot()
    summary["eval_protocol"] = paper_eval_protocol if run_paper else phase1_eval_protocol
    if not run_paper and paper_results:
        summary["phase1_report"] = paper_report_path
    report_path = save_json(summary, "experiment_comparison_report.json")
    if paper_results and msc_results and benchmark_stats:
        print_comparison_report(paper_results, msc_results, benchmark_stats, ablation_results)
    print(f"Full results saved to: {report_path}")
    print(f"Model checkpoints saved in: {CHECKPOINT_DIR}/")
    return summary


if __name__ == "__main__":
    args = parse_runtime_args("MSc capstone suite — Experiments A + B + C")
    if args.paper_only and args.msc_only:
        print("Error: use only one of --paper-only or --msc-only", file=sys.stderr)
        sys.exit(1)

    datasets = (args.dataset,) if args.dataset else DATASETS
    protocol = paper_eval_protocol(args) or PAPER_DEFAULT_EVAL
    started = time.time()
    run_all_experiments(
        run_ablation=True,
        force_cpu=args.cpu,
        batch_size=args.batch_size,
        paper_max_epochs=args.max_epochs,
        msc_max_epochs=args.max_epochs,
        datasets=datasets,
        require_real=args.require_real,
        run_paper=not args.msc_only,
        run_msc=not args.paper_only,
        paper_eval_protocol=protocol,
    )
    elapsed = time.time() - started
    print(f"Total experiment runtime: {elapsed / 60:.1f} minutes")
