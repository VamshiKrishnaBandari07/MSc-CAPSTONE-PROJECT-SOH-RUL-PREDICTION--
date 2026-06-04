"""Post-load filtering and scaling aligned with Rahman et al. (2026) Section 3."""

from __future__ import annotations

import numpy as np


def iqr_inlier_mask(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """IQR outlier mask (paper: statistical outlier detection before training)."""
    values = np.asarray(values, dtype=np.float64).flatten()
    if len(values) < 8:
        return np.ones(len(values), dtype=bool)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    if iqr < 1e-12:
        return np.ones(len(values), dtype=bool)
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (values >= lo) & (values <= hi)


def filter_cycles_iqr(
    features: np.ndarray,
    soh: np.ndarray,
    k: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Remove cycles with anomalous SOH or feature energy (paper IQR step).
    Returns (features, soh, n_removed).
    """
    features = np.asarray(features, dtype=np.float32)
    soh = np.asarray(soh, dtype=np.float32).flatten()
    energy = np.linalg.norm(features.reshape(len(features), -1), axis=1)
    mask = iqr_inlier_mask(soh, k=k) & iqr_inlier_mask(energy, k=k)
    removed = int(len(soh) - np.sum(mask))
    return features[mask], soh[mask], removed


def global_minmax_channels(features: np.ndarray) -> np.ndarray:
    """Per-channel min–max over all cycles (paper: normalized across cells)."""
    out = np.empty_like(features, dtype=np.float32)
    for ch in range(features.shape[1]):
        plane = features[:, ch, :]
        lo = float(np.min(plane))
        hi = float(np.max(plane))
        if hi - lo < 1e-12:
            out[:, ch, :] = 0.0
        else:
            out[:, ch, :] = (plane - lo) / (hi - lo)
    return out


def prepare_paper_tensors(features: np.ndarray, soh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IQR filter then global channel scaling."""
    features, soh, removed = filter_cycles_iqr(features, soh)
    if removed:
        print(f"[Paper] IQR filter removed {removed} outlier cycles ({len(soh)} remaining).")
    features = global_minmax_channels(features)
    return features, soh
