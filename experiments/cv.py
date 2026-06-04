"""Cross-validation utilities aligned with Rahman et al. (2026) stratified k-fold evaluation."""

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


def chronological_split(n_samples, train_ratio=0.8):
    """80/20 chronological split (supplementary local evaluation only)."""
    split_idx = max(1, int(n_samples * train_ratio))
    if split_idx >= n_samples:
        split_idx = n_samples - 1
    train_idx = np.arange(0, split_idx)
    val_idx = np.arange(split_idx, n_samples)
    return train_idx, val_idx
