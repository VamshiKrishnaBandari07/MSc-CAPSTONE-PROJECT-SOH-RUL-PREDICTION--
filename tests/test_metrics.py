import numpy as np

from experiments.metrics import monotonicity_violation_rate, regression_metrics
from experiments.trainer import split_indices


def test_regression_metrics_perfect():
    y = np.array([0.9, 0.8, 0.7])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_monotonicity_violation_rate():
    assert monotonicity_violation_rate([1.0, 0.9, 0.8]) == 0.0


def test_split_indices_reserves_validation():
    assert split_indices(10) == 8
