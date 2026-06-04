"""Numerical stability helpers for paper SOH training."""

from __future__ import annotations

import numpy as np
import torch


def fit_fold_scaler(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel min–max from training fold only (no validation leakage)."""
    train = np.asarray(features[train_idx], dtype=np.float32)
    mins = train.min(axis=(0, 2), keepdims=True)
    maxs = train.max(axis=(0, 2), keepdims=True)
    return mins, maxs


def apply_fold_scaler(
    features: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    span = np.maximum(maxs - mins, 1e-6)
    scaled = (np.asarray(features, dtype=np.float32) - mins) / span
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def augment_features(
    features_batch: torch.Tensor,
    voltage_jitter_scale: float,
    feature_noise: float,
) -> torch.Tensor:
    """Paper ±10 mV jitter as light feature noise; clip to valid range."""
    out = features_batch
    if voltage_jitter_scale > 0:
        out = out + torch.randn_like(out) * voltage_jitter_scale
    if feature_noise > 0:
        out = out + torch.randn_like(out) * feature_noise
    return torch.clamp(out, 0.0, 1.0)


def sanitize_predictions(y_pred: np.ndarray) -> np.ndarray:
    """Finite SOH predictions in [0, 1] for metrics and OOF assembly."""
    y = np.asarray(y_pred, dtype=np.float64).flatten()
    y = np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(y, 0.0, 1.0)
