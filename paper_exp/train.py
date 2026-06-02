import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from .config import PAPER_TARGETS, PaperExperimentConfig
    from .model import PaperCNNTCNLSTMAttention, count_parameters
    from .preprocess import PaperDatasetBundle, load_paper_experiment_data
except ImportError:  # Allows `python paper_exp/train.py` from the repo root.
    from config import PAPER_TARGETS, PaperExperimentConfig
    from model import PaperCNNTCNLSTMAttention, count_parameters
    from preprocess import PaperDatasetBundle, load_paper_experiment_data


class BatterySOHDataset(Dataset):
    """PyTorch dataset for paper feature tensors and SOH labels."""

    def __init__(self, features: np.ndarray, soh: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.soh = torch.tensor(soh, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.soh[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_stratified_cycle_folds(dataset_names: np.ndarray, n_folds: int) -> List[np.ndarray]:
    """
    Build 5-fold stratified cycle segmentation without an sklearn dependency.

    Each fold receives a contiguous cycle segment from every dataset so NASA, Oxford,
    and CALCE remain represented in all validation splits.
    """

    folds = [[] for _ in range(n_folds)]
    for dataset_name in np.unique(dataset_names):
        dataset_indices = np.where(dataset_names == dataset_name)[0]
        for fold_id, fold_indices in enumerate(np.array_split(dataset_indices, n_folds)):
            folds[fold_id].extend(fold_indices.tolist())
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    residual = predictions - targets
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / denominator if denominator > 0.0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    attention_sample = None
    with torch.no_grad():
        for features, soh in loader:
            features = features.to(device)
            pred, attention = model(features)
            predictions.append(pred.cpu().numpy())
            targets.append(soh.numpy())
            if attention_sample is None:
                attention_sample = attention.detach().cpu().numpy()

    pred_array = np.concatenate(predictions, axis=0).reshape(-1)
    target_array = np.concatenate(targets, axis=0).reshape(-1)
    return regression_metrics(pred_array, target_array), attention_sample


def train_one_fold(
    bundle: PaperDatasetBundle,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold_id: int,
) -> Tuple[Dict[str, float], np.ndarray, int]:
    dataset = BatterySOHDataset(bundle.features, bundle.soh)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = PaperCNNTCNLSTMAttention(dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    use_amp = args.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    best_state = deepcopy(model.state_dict())
    best_rmse = float("inf")
    best_metrics: Dict[str, float] = {"rmse": float("inf"), "mae": float("inf"), "r2": 0.0}
    best_attention = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for features, soh in train_loader:
            features = features.to(device)
            soh = soh.to(device)

            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with amp_context:
                predictions, _ = model(features)
                loss = criterion(predictions, soh)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_indices)
        metrics, attention_sample = evaluate_model(model, val_loader, device)
        scheduler.step(metrics["rmse"])

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_metrics = metrics
            best_attention = attention_sample
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Fold {fold_id:02d} | Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train MSE: {train_loss:.6f} | Val RMSE: {metrics['rmse']:.4f} | "
            f"Val R2: {metrics['r2']:.4f}"
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Fold {fold_id:02d} early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return best_metrics, best_attention, count_parameters(model)


def benchmark_efficiency(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    edge_power_watts: float,
    repeats: int = 50,
) -> Dict[str, float]:
    model.eval()
    sample = torch.randn(batch_size, 3, seq_len, device=device)

    with torch.no_grad():
        for _ in range(5):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / (repeats * batch_size)
    energy_mj = latency_ms * edge_power_watts
    return {
        "latency_ms_per_sample": float(latency_ms),
        "estimated_energy_mj_per_sample": float(energy_mj),
    }


def summarize_fold_metrics(fold_metrics: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    metrics = list(fold_metrics)
    summary = {}
    for key in ("rmse", "mae", "r2"):
        values = np.asarray([metric[key] for metric in metrics], dtype=np.float64)
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_paper_experiment_data(
        dataset_names=args.datasets,
        cycles_per_dataset=args.cycles_per_dataset,
        seq_len=args.seq_len,
        raw_dir=args.raw_dir,
        seed=args.seed,
        require_real_data=args.require_real_data,
    )
    folds = make_stratified_cycle_folds(bundle.dataset_names, args.n_folds)
    actual_seq_len = int(bundle.features.shape[-1])
    if actual_seq_len != args.seq_len:
        print(
            f"Warning: requested seq_len={args.seq_len}, but loaded processed features "
            f"have seq_len={actual_seq_len}. Using loaded length for benchmarking."
        )
    print(f"Loaded {len(bundle.soh)} cycles: features={bundle.features.shape}, device={device}")

    fold_metrics = []
    parameter_count = None
    for fold_number, val_indices in enumerate(folds, start=1):
        train_indices = np.setdiff1d(np.arange(len(bundle.soh)), val_indices)
        metrics, attention_sample, fold_params = train_one_fold(
            bundle=bundle,
            train_indices=train_indices,
            val_indices=val_indices,
            args=args,
            device=device,
            fold_id=fold_number,
        )
        fold_metrics.append(metrics)
        parameter_count = fold_params
        if args.save_attention and attention_sample is not None:
            np.savez_compressed(output_dir / f"attention_fold_{fold_number}.npz", attention=attention_sample)

    fresh_model = PaperCNNTCNLSTMAttention(dropout=args.dropout).to(device)
    efficiency = benchmark_efficiency(
        fresh_model,
        seq_len=actual_seq_len,
        batch_size=min(args.batch_size, 64),
        device=device,
        edge_power_watts=args.edge_power_watts,
        repeats=args.benchmark_repeats,
    )

    summary = {
        "paper": {
            "title": "Deep learning-based battery health prediction for enhancing electric vehicle performance",
            "doi": "10.1038/s41598-026-39911-8",
            "targets": PAPER_TARGETS,
        },
        "config": vars(args),
        "fold_metrics": fold_metrics,
        "summary": summarize_fold_metrics(fold_metrics),
        "parameter_count": int(parameter_count or count_parameters(fresh_model)),
        "parameter_count_millions": float((parameter_count or count_parameters(fresh_model)) / 1e6),
        "efficiency": efficiency,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n================ paper_exp summary ================")
    print(f"SOH RMSE: {summary['summary']['rmse']['mean']:.4f} +/- {summary['summary']['rmse']['std']:.4f}")
    print(f"SOH R2:   {summary['summary']['r2']['mean']:.4f} +/- {summary['summary']['r2']['std']:.4f}")
    print(f"Params:   {summary['parameter_count_millions']:.3f}M (paper target ~{PAPER_TARGETS['parameters_millions']}M)")
    print(
        "Latency:  "
        f"{efficiency['latency_ms_per_sample']:.3f} ms/sample | "
        f"Energy: {efficiency['estimated_energy_mj_per_sample']:.3f} mJ/sample"
    )
    print(f"Saved metrics to {metrics_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = PaperExperimentConfig()
    parser = argparse.ArgumentParser(description="Run the s41598-026-39911-8 paper_exp reproduction.")
    parser.add_argument("--datasets", nargs="+", default=list(defaults.datasets), choices=sorted(defaults.supported_datasets))
    parser.add_argument("--raw-dir", default="data", help="Directory containing processed <DATASET>_paper_exp.npz files.")
    parser.add_argument(
        "--require-real-data",
        action="store_true",
        help="Fail instead of using synthetic fallback when processed paper dataset files are missing.",
    )
    parser.add_argument("--output-dir", default=os.path.join("paper_exp", "outputs"))
    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument("--cycles-per-dataset", type=int, default=defaults.cycles_per_dataset)
    parser.add_argument("--n-folds", type=int, default=defaults.n_folds)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--adam-beta1", type=float, default=defaults.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=defaults.adam_beta2)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--scheduler-factor", type=float, default=defaults.scheduler_factor)
    parser.add_argument("--scheduler-patience", type=int, default=defaults.scheduler_patience)
    parser.add_argument("--edge-power-watts", type=float, default=defaults.edge_power_watts)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--mixed-precision", action="store_true", help="Use CUDA AMP when running on GPU.")
    parser.add_argument("--save-attention", action="store_true", help="Save one validation attention tensor per fold.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny verification job instead of the full paper setup.")
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.cycles_per_dataset = 12
    args.seq_len = 64
    args.n_folds = 2
    args.epochs = 1
    args.batch_size = 8
    args.early_stopping_patience = 1
    args.benchmark_repeats = 2
    args.output_dir = os.path.join("paper_exp", "outputs", "smoke")
    return args


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    run_experiment(apply_smoke_overrides(parsed_args))

