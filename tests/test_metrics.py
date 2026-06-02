import numpy as np

from experiments.metrics import monotonicity_violation_rate, regression_metrics


def test_regression_metrics_perfect():
    y = np.array([0.9, 0.8, 0.7])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0


def test_monotonicity_violation_rate():
    assert monotonicity_violation_rate([1.0, 0.9, 0.8]) == 0.0
    assert monotonicity_violation_rate([0.8, 0.9, 0.7]) == 0.5
