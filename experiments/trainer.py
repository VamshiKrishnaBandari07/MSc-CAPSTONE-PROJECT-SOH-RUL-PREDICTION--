"""Training loop for paper SOH reproduction (Scientific Reports 2026)."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from experiments.config import EARLY_STOPPING_PATIENCE, RANDOM_SEED, TRAIN_RATIO
from experiments.cv import chronological_split, stratified_kfold_splits
from experiments.io_utils import load_checkpoint, save_checkpoint
from experiments.metrics import monotonicity_violation_rate, regression_metrics
from experiments.paper_preprocessing import sanitize_feature_tensor
from experiments.runtime import get_device, paper_batch_size


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Stop training when validation RMSE plateaus."""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class PaperDataset(Dataset):
    """PyTorch dataset: (ICA, DV, DC) tensor → SOH target."""

    def __init__(self, features: np.ndarray, soh: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.soh = torch.tensor(soh, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.soh[idx]


def split_indices(n_samples: int, train_ratio: float = TRAIN_RATIO) -> int:
    """Chronological split index for supplementary 80/20 evaluation."""
    split_idx = max(1, int(n_samples * train_ratio))
    if split_idx >= n_samples:
        split_idx = n_samples - 1
    return split_idx


def _evaluate_paper_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for features, soh in loader:
            features = features.to(device)
            pred, _ = model(features)
            preds.append(pred.cpu().numpy())
            targets.append(soh.numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics = regression_metrics(y_true, y_pred)
    metrics["mono_violation_rate"] = monotonicity_violation_rate(y_pred)
    return metrics, y_true.flatten(), y_pred.flatten()


def _train_paper_on_indices(
    model: nn.Module,
    features: np.ndarray,
    soh: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset_name: str,
    checkpoint_path: str,
    epochs: Optional[int],
    batch_size: Optional[int],
    fold_label: Optional[int] = None,
) -> Dict[str, Any]:
    """Train one fold or one chronological split using paper hyperparameters."""
    from experiments.paper_config import (
        PAPER_EARLY_STOPPING_PATIENCE,
        PAPER_FEATURE_NOISE,
        PAPER_GRAD_CLIP_NORM,
        PAPER_LEARNING_RATE,
        PAPER_LR_SCHEDULER_FACTOR,
        PAPER_LR_SCHEDULER_PATIENCE,
        PAPER_MAX_EPOCHS,
        PAPER_VOLTAGE_JITTER_V,
        PAPER_VOLTAGE_MAX,
        PAPER_VOLTAGE_MIN,
        PAPER_WEIGHT_DECAY,
    )

    epochs = epochs or PAPER_MAX_EPOCHS
    batch_size = batch_size or paper_batch_size()
    lr = PAPER_LEARNING_RATE
    weight_decay = PAPER_WEIGHT_DECAY
    grad_clip = PAPER_GRAD_CLIP_NORM
    early_stop = EarlyStopping(patience=PAPER_EARLY_STOPPING_PATIENCE)
    sched_factor = PAPER_LR_SCHEDULER_FACTOR
    sched_patience = PAPER_LR_SCHEDULER_PATIENCE
    feature_noise = PAPER_FEATURE_NOISE
    # ±10 mV on 1.7 V grid ≈ relative scale on normalized feature channels (paper Section 3)
    voltage_jitter_scale = PAPER_VOLTAGE_JITTER_V / max(PAPER_VOLTAGE_MAX - PAPER_VOLTAGE_MIN, 1e-6)

    train_features = np.array(
        [sanitize_feature_tensor(features[i]) for i in train_idx], dtype=np.float32
    )
    train_soh = soh[train_idx]
    val_features = np.array(
        [sanitize_feature_tensor(features[i]) for i in val_idx], dtype=np.float32
    )
    val_soh = soh[val_idx]

    train_ds = PaperDataset(train_features, train_soh)
    val_ds = PaperDataset(val_features, val_soh)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = model.to(device)
    tag = f"{dataset_name}" + (f" fold {fold_label}" if fold_label is not None else "")
    if device.type == "cpu":
        print(
            f"[Paper | {tag}] Training on CPU "
            f"(batch_size={batch_size}, train={len(train_idx)}, val={len(val_idx)})"
        )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_factor, patience=sched_patience
    )

    history: List[Dict[str, Any]] = []
    best_metrics: Optional[Dict[str, float]] = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for features_batch, targets in train_loader:
            features_batch, targets = features_batch.to(device), targets.to(device)
            if voltage_jitter_scale > 0:
                features_batch = features_batch + torch.randn_like(features_batch) * voltage_jitter_scale
            if feature_noise > 0:
                features_batch = features_batch + torch.randn_like(features_batch) * feature_noise
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(features_batch)
            loss = criterion(pred, targets)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * features_batch.size(0)
            n_batches += 1

        if n_batches == 0:
            print(f"[Paper | {tag}] All batches non-finite at epoch {epoch}. Stopping.")
            break

        val_metrics, _, _ = _evaluate_paper_model(model, val_loader, device)
        if not np.isfinite(val_metrics["rmse"]):
            print(f"[Paper | {tag}] Non-finite validation at epoch {epoch}. Stopping.")
            break

        avg_train_loss = train_loss / len(train_ds)
        scheduler.step(val_metrics["rmse"])
        history.append(
            {
                "epoch": epoch,
                "train_mse": avg_train_loss,
                "val_soh_rmse": val_metrics["rmse"],
                "val_soh_mae": val_metrics["mae"],
                "val_soh_r2": val_metrics["r2"],
                "mono_violation_rate": val_metrics["mono_violation_rate"],
            }
        )

        if early_stop.step(val_metrics["rmse"]):
            best_metrics = val_metrics
            best_epoch = epoch
            save_checkpoint(
                model,
                checkpoint_path,
                metadata={"dataset": dataset_name, "epoch": epoch, "metrics": val_metrics, "fold": fold_label},
            )

        print(
            f"[Paper | {tag}] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train MSE: {avg_train_loss:.5f} | Val SOH RMSE: {val_metrics['rmse']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | R2: {val_metrics['r2']:.4f}"
        )

        if early_stop.should_stop:
            print(f"[Paper | {tag}] Early stopping at epoch {epoch}.")
            break

    val_y_true = np.array([], dtype=np.float64)
    val_y_pred = np.array([], dtype=np.float64)
    if best_metrics and os.path.isfile(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device)
        best_metrics, val_y_true, val_y_pred = _evaluate_paper_model(model, val_loader, device)

    return {
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "history": history,
        "train_cycles": int(len(train_idx)),
        "val_cycles": int(len(val_idx)),
        "val_y_true": val_y_true,
        "val_y_pred": val_y_pred,
    }


def train_paper_experiment(
    model: nn.Module,
    features: np.ndarray,
    soh: np.ndarray,
    dataset_name: str,
    checkpoint_path: str,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_protocol: str = "cv5",
) -> Dict[str, Any]:
    """
    Train paper reproduction model (MSE, SOH-only).

    eval_protocol:
      - ``cv5``: stratified 5-fold CV (primary paper metric)
      - ``chronological``: 80/20 chronological split (supplementary)
    """
    from experiments.paper_config import PAPER_DEFAULT_EVAL
    from model_paper import build_paper_model

    eval_protocol = eval_protocol or PAPER_DEFAULT_EVAL
    features = np.asarray(features, dtype=np.float32)
    soh = np.asarray(soh, dtype=np.float32)

    if eval_protocol == "cv5":
        seq_len = features.shape[2]
        fold_metrics: List[float] = []
        fold_histories: List[Dict[str, Any]] = []
        best_fold = 1
        best_fold_result: Optional[Dict[str, Any]] = None

        for fold_i, (train_idx, val_idx) in enumerate(stratified_kfold_splits(soh), start=1):
            print(f"\n[Paper | {dataset_name}] === Stratified CV fold {fold_i}/5 ===")
            fold_model = build_paper_model(seq_len=seq_len)
            fold_ckpt = checkpoint_path.replace(".pt", f"_fold{fold_i}.pt")
            fold_result = _train_paper_on_indices(
                fold_model,
                features,
                soh,
                train_idx,
                val_idx,
                dataset_name,
                fold_ckpt,
                epochs=epochs,
                batch_size=batch_size,
                fold_label=fold_i,
            )
            if fold_result["metrics"]:
                fold_metrics.append(fold_result["metrics"]["rmse"])
            fold_histories.append({"fold": fold_i, **fold_result})

        if fold_metrics:
            std_rmse = float(np.std(fold_metrics))
            best_fold = int(np.argmin(fold_metrics)) + 1
            best_fold_result = fold_histories[best_fold - 1]

            oof_true = [
                fr["val_y_true"] for fr in fold_histories if len(fr.get("val_y_true", [])) > 0
            ]
            oof_pred = [
                fr["val_y_pred"] for fr in fold_histories if len(fr.get("val_y_pred", [])) > 0
            ]
            if oof_true:
                pooled = regression_metrics(np.concatenate(oof_true), np.concatenate(oof_pred))
                mean_fold_rmse = float(np.mean(fold_metrics))
                aggregated = {
                    **pooled,
                    "rmse_std": std_rmse,
                    "rmse_folds": fold_metrics,
                    "rmse_mean_folds": mean_fold_rmse,
                    "mono_violation_rate": monotonicity_violation_rate(np.concatenate(oof_pred)),
                }
                print(
                    f"[Paper | {dataset_name}] CV summary (pooled OOF): SOH RMSE = {pooled['rmse']:.4f} "
                    f"| mean fold RMSE = {mean_fold_rmse:.4f} ± {std_rmse:.4f} "
                    f"(folds: {[round(x, 4) for x in fold_metrics]})"
                )
            else:
                mean_rmse = float(np.mean(fold_metrics))
                aggregated = {
                    "rmse": mean_rmse,
                    "rmse_std": std_rmse,
                    "rmse_folds": fold_metrics,
                    "mae": best_fold_result["metrics"]["mae"],
                    "r2": best_fold_result["metrics"]["r2"],
                    "mse": mean_rmse**2,
                    "mono_violation_rate": best_fold_result["metrics"]["mono_violation_rate"],
                }
                print(
                    f"[Paper | {dataset_name}] CV summary: SOH RMSE = {mean_rmse:.4f} ± {std_rmse:.4f} "
                    f"(folds: {[round(x, 4) for x in fold_metrics]})"
                )
        else:
            aggregated = None

        return {
            "dataset": dataset_name,
            "experiment": "paper_reproduction",
            "methodology": "scientific_reports_2026",
            "eval_protocol": "stratified_5fold_cv",
            "best_epoch": best_fold_result["best_epoch"] if fold_metrics else 0,
            "metrics": aggregated,
            "fold_results": fold_histories,
            "history": best_fold_result["history"] if fold_metrics else [],
            "checkpoint": checkpoint_path.replace(".pt", f"_fold{best_fold}.pt")
            if fold_metrics
            else checkpoint_path,
            "train_cycles": None,
            "val_cycles": None,
        }

    train_idx, val_idx = chronological_split(len(features))
    split_result = _train_paper_on_indices(
        model,
        features,
        soh,
        train_idx,
        val_idx,
        dataset_name,
        checkpoint_path,
        epochs=epochs,
        batch_size=batch_size,
    )
    return {
        "dataset": dataset_name,
        "experiment": "paper_reproduction",
        "methodology": "scientific_reports_2026",
        "eval_protocol": "chronological_80_20",
        "best_epoch": split_result["best_epoch"],
        "metrics": split_result["metrics"],
        "history": split_result["history"],
        "checkpoint": checkpoint_path,
        "train_cycles": split_result["train_cycles"],
        "val_cycles": split_result["val_cycles"],
    }
