import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

from benchmark import benchmark_model
from model import BatteryHealthPredictor
from train import train_and_evaluate


def _mean_std(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def run_modified_experiment(args: argparse.Namespace) -> Dict[str, object]:
    """
    Run the repository's modified MSc/thesis experiment and save structured metrics.

    This intentionally reuses the root-level `train.py` implementation so the
    comparison is against the existing modified experiment, not a duplicate.
    """

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_metrics = {}
    trained_model = None
    for dataset_name in args.datasets:
        soh_rmse, rul_rmse, trained_model = train_and_evaluate(
            dataset_name=dataset_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
        )
        dataset_metrics[dataset_name] = {
            "soh_rmse": float(soh_rmse),
            "rul_rmse_cycles": float(rul_rmse),
        }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_for_benchmark = BatteryHealthPredictor(input_features=3).to(device)
    latency_ms = float(benchmark_model(model_for_benchmark, device, num_runs=args.benchmark_runs))
    energy_mj = float(latency_ms * args.edge_power_watts)
    parameter_count = sum(parameter.numel() for parameter in model_for_benchmark.parameters() if parameter.requires_grad)

    summary = {
        "experiment": "modified_msc_physics_informed_joint_soh_rul",
        "description": "Existing repository experiment from root train.py: joint SOH/RUL with physics-informed monotonicity loss.",
        "config": vars(args),
        "datasets": dataset_metrics,
        "summary": {
            "soh_rmse": _mean_std(metric["soh_rmse"] for metric in dataset_metrics.values()),
            "rul_rmse_cycles": _mean_std(metric["rul_rmse_cycles"] for metric in dataset_metrics.values()),
        },
        "parameter_count": int(parameter_count),
        "parameter_count_millions": float(parameter_count / 1e6),
        "efficiency": {
            "latency_ms_per_sample": latency_ms,
            "estimated_energy_mj_per_sample": energy_mj,
        },
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n================ modified experiment summary ================")
    print(
        f"SOH RMSE: {summary['summary']['soh_rmse']['mean']:.4f} "
        f"+/- {summary['summary']['soh_rmse']['std']:.4f}"
    )
    print(
        f"RUL RMSE: {summary['summary']['rul_rmse_cycles']['mean']:.2f} "
        f"+/- {summary['summary']['rul_rmse_cycles']['std']:.2f} cycles"
    )
    print(f"Params:   {summary['parameter_count_millions']:.3f}M")
    print(f"Latency:  {latency_ms:.3f} ms/sample | Energy: {energy_mj:.3f} mJ/sample")
    print(f"Saved metrics to {metrics_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the repository's modified MSc SOH/RUL experiment.")
    parser.add_argument("--datasets", nargs="+", default=["NASA", "Oxford", "CALCE"], choices=["NASA", "Oxford", "CALCE"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--benchmark-runs", type=int, default=100)
    parser.add_argument("--edge-power-watts", type=float, default=0.103)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--output-dir", default="paper_exp/outputs/modified_experiment")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny verification job.")
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.datasets = args.datasets[:1]
    args.epochs = 1
    args.batch_size = min(args.batch_size, 8)
    args.benchmark_runs = min(args.benchmark_runs, 10)
    args.output_dir = str(Path(args.output_dir) / "smoke")
    return args


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    run_modified_experiment(apply_smoke_overrides(parsed_args))

