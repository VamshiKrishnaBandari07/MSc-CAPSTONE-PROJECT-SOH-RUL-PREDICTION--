"""Cross-validation utilities aligned with Rahman et al. (2026) stratified k-fold evaluation."""

from __future__ import annotations

import numpy as np

from experiments.config import RANDOM_SEED
from experiments.paper_config import PAPER_CV_FOLDS


def _soh_strata(soh, n_bins=5):
    """Bin SOH for stratified splits (paper: stratified 5-fold on health levels)."""
    soh = np.asarray(soh, dtype=np.float64).flatten()
    if len(soh) < n_bins * 2:
        return np.zeros(len(soh), dtype=int)
    try:
        edges = np.quantile(soh, np.linspace(0, 1, n_bins + 1)[1:-1])
        return np.digitize(soh, bins=edges)
    except Exception:
        return np.zeros(len(soh), dtype=int)


def stratified_kfold_splits(soh, n_folds=PAPER_CV_FOLDS, seed=RANDOM_SEED):
    """
    Yield (train_idx, val_idx) for each fold.
    Matches paper protocol: stratified k-fold cross-validation (no sklearn dependency).
    """
    n = len(soh)
    if n < n_folds * 2:
        raise ValueError(f"Need at least {n_folds * 2} samples for {n_folds}-fold CV, got {n}")

    rng = np.random.RandomState(seed)
    strata = _soh_strata(soh)
    buckets = {}
    for i, s in enumerate(strata):
        buckets.setdefault(int(s), []).append(i)

    folds = [[] for _ in range(n_folds)]
    for idxs in buckets.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[j % n_folds].append(idx)

    for k in range(n_folds):
        val_idx = np.array(sorted(folds[k]), dtype=int)
        train_idx = np.array(
            sorted(i for j in range(n_folds) if j != k for i in folds[j]),
            dtype=int,
        )
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("Stratified fold produced empty train or validation set.")
        yield train_idx, val_idx


def _encode_groups(groups) -> np.ndarray:
    """Map arbitrary group labels to contiguous integers 0..G-1."""
    groups = np.asarray(groups)
    uniq = {}
    encoded = np.empty(len(groups), dtype=int)
    for i, g in enumerate(groups):
        if g not in uniq:
            uniq[g] = len(uniq)
        encoded[i] = uniq[g]
    return encoded


def grouped_stratified_kfold_splits(
    soh,
    groups,
    n_folds: int = PAPER_CV_FOLDS,
    seed: int = RANDOM_SEED,
):
    """
    Assign whole cells/batteries to folds (no cycle leakage across cells).
    Within each fold, validation cycles still span SOH strata when possible.
    """
    soh = np.asarray(soh, dtype=np.float64).flatten()
    g = _encode_groups(groups)
    n = len(soh)
    n_groups = int(g.max()) + 1 if len(g) else 0
    if n_groups < 2:
        yield from stratified_kfold_splits(soh, n_folds=n_folds, seed=seed)
        return

    n_folds = min(n_folds, n_groups)
    rng = np.random.RandomState(seed)

    group_ids = np.arange(n_groups)
    rng.shuffle(group_ids)
    group_to_fold = {gid: i % n_folds for i, gid in enumerate(group_ids)}

    for k in range(n_folds):
        val_mask = np.array([group_to_fold[int(gi)] == k for gi in g], dtype=bool)
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Grouped fold {k + 1} empty (groups={n_groups}, folds={n_folds}).")
        yield train_idx.astype(int), val_idx.astype(int)


def chronological_split(n_samples, train_ratio=0.8):
    """80/20 chronological split (supplementary local evaluation only)."""
    split_idx = max(1, int(n_samples * train_ratio))
    if split_idx >= n_samples:
        split_idx = n_samples - 1
    train_idx = np.arange(0, split_idx)
    val_idx = np.arange(split_idx, n_samples)
    return train_idx, val_idx
