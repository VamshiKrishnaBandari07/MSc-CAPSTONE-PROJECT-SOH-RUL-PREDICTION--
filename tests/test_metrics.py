import numpy as np

from experiments.metrics import monotonicity_violation_rate, regression_metrics
from experiments.trainer import _msc_validation_score, split_indices


def test_regression_metrics_perfect():
    y = np.array([0.9, 0.8, 0.7])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0


def test_monotonicity_violation_rate():
    assert monotonicity_violation_rate([1.0, 0.9, 0.8]) == 0.0
    assert monotonicity_violation_rate([0.8, 0.9, 0.7]) == 0.5


def test_split_indices_reserves_validation():
    assert split_indices(10) == 8
    assert split_indices(2) == 1
    assert split_indices(1) == 0  # single sample: no train split possible


def test_msc_validation_score_balances_soh_and_rul():
    soh = {"rmse": 0.1}
    rul = {"rmse": 20.0}
    score = _msc_validation_score(soh, rul, max_rul=100.0)
    assert score == 0.1 + 0.5 * 0.2


def test_nasa_per_cell_rul_labels():
    import os

    from preprocess import BatteryDatasetLoader

    nasa_dir = os.path.join(os.getcwd(), "data", "NASA")
    if not os.path.isdir(nasa_dir) or not any(f.endswith(".mat") for f in os.listdir(nasa_dir)):
        return

    _, soh, rul = BatteryDatasetLoader.load_dataset("NASA")
    assert len(rul) == len(soh)
    assert np.all(rul >= 0)
    assert float(np.max(rul)) > 0
