import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from experiments.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MSC_DEFAULTS,
    RANDOM_SEED,
    TRAIN_RATIO,
    WEIGHT_DECAY,
)
from experiments.io_utils import save_checkpoint
from experiments.runtime import get_device, msc_batch_size, paper_batch_size
from experiments.metrics import monotonicity_violation_rate, regression_metrics


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class PaperDataset(Dataset):
    def __init__(self, features, soh):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.soh = torch.tensor(soh, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.soh[idx]


class MScDataset(Dataset):
    def __init__(self, features, soh, rul, max_rul):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.soh = torch.tensor(soh, dtype=torch.float32).unsqueeze(1)
        self.rul = torch.tensor(rul, dtype=torch.float32).unsqueeze(1) / max(max_rul, 1.0)
        self.max_rul = float(max_rul)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.soh[idx], self.rul[idx]


class JointPhysicsInformedLoss(nn.Module):
    def __init__(self, rul_weight=MSC_DEFAULTS["rul_weight"], monotonicity_weight=MSC_DEFAULTS["monotonicity_weight"]):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = rul_weight
        self.lambda_mono = monotonicity_weight

    def forward(self, pred_soh, pred_rul, target_soh, target_rul):
        loss_soh = self.mse_loss(pred_soh, target_soh)
        loss_rul = self.mse_loss(pred_rul, target_rul)

        if pred_soh.size(0) > 1 and self.lambda_mono > 0:
            diff = pred_soh[1:] - pred_soh[:-1]
            mono_penalty = torch.mean(torch.relu(diff))
        else:
            mono_penalty = torch.tensor(0.0, device=pred_soh.device)

        total = loss_soh + self.alpha * loss_rul + self.lambda_mono * mono_penalty
        return total, loss_soh, loss_rul, mono_penalty


def split_indices(n_samples, train_ratio=TRAIN_RATIO):
    split_idx = max(1, int(n_samples * train_ratio))
    if split_idx >= n_samples:
        split_idx = n_samples - 1
    return split_idx


def _msc_validation_score(soh_metrics, rul_metrics, max_rul):
    """Combined early-stopping score for joint SOH + RUL training."""
    normalized_rul_rmse = rul_metrics["rmse"] / max(max_rul, 1.0)
    return soh_metrics["rmse"] + MSC_DEFAULTS["rul_weight"] * normalized_rul_rmse


def _evaluate_paper_model(model, loader, device):
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


def _evaluate_msc_model(model, loader, device, max_rul):
    model.eval()
    soh_preds, soh_targets = [], []
    rul_preds, rul_targets = [], []
    with torch.no_grad():
        for features, soh, rul in loader:
            features = features.to(device)
            pred_soh, pred_rul, _ = model(features)
            soh_preds.append(pred_soh.cpu().numpy())
            soh_targets.append(soh.numpy())
            rul_preds.append((pred_rul * max_rul).cpu().numpy())
            rul_targets.append((rul * max_rul).numpy())

    y_soh_pred = np.concatenate(soh_preds, axis=0).flatten()
    y_soh_true = np.concatenate(soh_targets, axis=0).flatten()
    y_rul_pred = np.concatenate(rul_preds, axis=0).flatten()
    y_rul_true = np.concatenate(rul_targets, axis=0).flatten()

    soh_metrics = regression_metrics(y_soh_true, y_soh_pred)
    rul_metrics = regression_metrics(y_rul_true, y_rul_pred)
    soh_metrics["mono_violation_rate"] = monotonicity_violation_rate(y_soh_pred)

    return {
        "soh": soh_metrics,
        "rul": rul_metrics,
        "predictions": {
            "soh_true": y_soh_true.tolist(),
            "soh_pred": y_soh_pred.tolist(),
            "rul_true": y_rul_true.tolist(),
            "rul_pred": y_rul_pred.tolist(),
        },
    }


def train_paper_experiment(
    model,
    features,
    soh,
    dataset_name,
    checkpoint_path,
    epochs=None,
    batch_size=None,
    use_paper_protocol=True,
):
    """Train Experiment A — paper reproduction (MSE, SOH-only)."""
    if use_paper_protocol:
        from experiments.paper_config import (
            PAPER_BATCH_SIZE,
            PAPER_EARLY_STOPPING_PATIENCE,
            PAPER_GRAD_CLIP_NORM,
            PAPER_LEARNING_RATE,
            PAPER_LR_SCHEDULER_FACTOR,
            PAPER_LR_SCHEDULER_PATIENCE,
            PAPER_MAX_EPOCHS,
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
        feature_noise = 0.005  # training-time jitter on normalized ICA/DV/DC channels
    else:
        epochs = epochs or MAX_EPOCHS
        batch_size = batch_size or msc_batch_size()
        lr = LEARNING_RATE
        weight_decay = WEIGHT_DECAY
        grad_clip = None
        early_stop = EarlyStopping()
        sched_factor = 0.5
        sched_patience = 2
        feature_noise = 0.0

    split_idx = split_indices(len(features))
    train_ds = PaperDataset(features[:split_idx], soh[:split_idx])
    val_ds = PaperDataset(features[split_idx:], soh[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = model.to(device)
    if device.type == "cpu":
        print(f"[Paper | {dataset_name}] Training on CPU (batch_size={batch_size})")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_factor, patience=sched_patience
    )

    history = []
    best_metrics = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for features_batch, targets in train_loader:
            features_batch, targets = features_batch.to(device), targets.to(device)
            if feature_noise > 0:
                features_batch = features_batch + torch.randn_like(features_batch) * feature_noise
            optimizer.zero_grad()
            pred, _ = model(features_batch)
            loss = criterion(pred, targets)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * features_batch.size(0)

        val_metrics, _, _ = _evaluate_paper_model(model, val_loader, device)
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

        improved = early_stop.step(val_metrics["rmse"])
        if improved:
            best_metrics = val_metrics
            best_epoch = epoch
            save_checkpoint(
                model,
                checkpoint_path,
                metadata={"dataset": dataset_name, "epoch": epoch, "metrics": val_metrics},
            )

        print(
            f"[Paper | {dataset_name}] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train MSE: {avg_train_loss:.5f} | Val SOH RMSE: {val_metrics['rmse']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | R2: {val_metrics['r2']:.4f}"
        )

        if early_stop.should_stop:
            print(f"[Paper | {dataset_name}] Early stopping at epoch {epoch}.")
            break

    return {
        "dataset": dataset_name,
        "experiment": "paper_reproduction",
        "methodology": "scientific_reports_2026" if use_paper_protocol else "lite",
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "history": history,
        "checkpoint": checkpoint_path,
        "train_cycles": split_idx,
        "val_cycles": len(features) - split_idx,
    }


def train_msc_experiment(
    model,
    features,
    soh,
    rul,
    dataset_name,
    checkpoint_path,
    use_physics_loss=True,
    epochs=None,
    batch_size=None,
):
    if epochs is None:
        epochs = MAX_EPOCHS
    if batch_size is None:
        batch_size = msc_batch_size()
    max_rul = float(np.max(rul)) if len(rul) else 1.0
    split_idx = split_indices(len(features))

    train_ds = MScDataset(features[:split_idx], soh[:split_idx], rul[:split_idx], max_rul)
    val_ds = MScDataset(features[split_idx:], soh[split_idx:], rul[split_idx:], max_rul)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = model.to(device)
    if device.type == "cpu":
        print(f"[MSc | {dataset_name}] Training on CPU (batch_size={batch_size})")

    mono_weight = MSC_DEFAULTS["monotonicity_weight"] if use_physics_loss else 0.0
    criterion = JointPhysicsInformedLoss(
        rul_weight=MSC_DEFAULTS["rul_weight"],
        monotonicity_weight=mono_weight,
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stop = EarlyStopping()

    history = []
    best_result = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features_batch, targets_soh, targets_rul in train_loader:
            features_batch = features_batch.to(device)
            targets_soh = targets_soh.to(device)
            targets_rul = targets_rul.to(device)

            optimizer.zero_grad()
            pred_soh, pred_rul, _ = model(features_batch)
            loss, l_soh, l_rul, l_mono = criterion(pred_soh, pred_rul, targets_soh, targets_rul)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * features_batch.size(0)

        eval_result = _evaluate_msc_model(model, val_loader, device, max_rul)
        soh_m = eval_result["soh"]
        rul_m = eval_result["rul"]
        val_score = _msc_validation_score(soh_m, rul_m, max_rul)
        scheduler.step(soh_m["rmse"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / len(train_ds),
                "val_soh_rmse": soh_m["rmse"],
                "val_soh_mae": soh_m["mae"],
                "val_soh_r2": soh_m["r2"],
                "val_rul_rmse": rul_m["rmse"],
                "val_rul_mae": rul_m["mae"],
                "val_rul_r2": rul_m["r2"],
                "val_combined_score": val_score,
                "mono_violation_rate": soh_m["mono_violation_rate"],
            }
        )

        improved = early_stop.step(val_score)
        if improved:
            best_result = eval_result
            best_epoch = epoch
            save_checkpoint(
                model,
                checkpoint_path,
                metadata={
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "use_physics_loss": use_physics_loss,
                    "metrics": {"soh": soh_m, "rul": rul_m},
                },
            )

        print(
            f"[MSc | {dataset_name}] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_loss / len(train_ds):.5f} | "
            f"Val SOH RMSE: {soh_m['rmse']:.4f} | Val RUL RMSE: {rul_m['rmse']:.2f} | "
            f"Mono Violations: {soh_m['mono_violation_rate']:.2%}"
        )

        if early_stop.should_stop:
            print(f"[MSc | {dataset_name}] Early stopping at epoch {epoch}.")
            break

    return {
        "dataset": dataset_name,
        "experiment": "msc_physics_joint" if use_physics_loss else "msc_ablation_no_physics",
        "use_physics_loss": use_physics_loss,
        "best_epoch": best_epoch,
        "metrics": {
            "soh": best_result["soh"] if best_result else {},
            "rul": best_result["rul"] if best_result else {},
        },
        "history": history,
        "checkpoint": checkpoint_path,
        "max_rul": max_rul,
        "train_cycles": split_idx,
        "val_cycles": len(features) - split_idx,
    }
