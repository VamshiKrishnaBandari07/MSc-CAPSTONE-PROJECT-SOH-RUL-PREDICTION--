import numpy as np

from experiments.cv import grouped_stratified_kfold_splits


def test_grouped_cv_no_cell_leakage():
    soh = np.linspace(0.9, 0.6, 40)
    groups = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10)
    for train_idx, val_idx in grouped_stratified_kfold_splits(soh, groups, n_folds=4, seed=42):
        val_cells = set(groups[val_idx])
        train_cells = set(groups[train_idx])
        assert val_cells.isdisjoint(train_cells)


def test_grouped_cv_covers_all_samples():
    soh = np.random.default_rng(0).uniform(0.6, 1.0, 50)
    groups = np.array([f"cell_{i // 10}" for i in range(50)])
    seen = np.zeros(50, dtype=bool)
    for _, val_idx in grouped_stratified_kfold_splits(soh, groups, n_folds=5, seed=1):
        seen[val_idx] = True
    assert seen.all()
