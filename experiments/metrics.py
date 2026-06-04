import numpy as np


def regression_metrics(y_true, y_pred):
    """Compute RMSE, MAE, and R² for regression targets."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "mse": mse}


def monotonicity_violation_rate(soh_sequence):
    """
    Fraction of consecutive cycle pairs where predicted SOH increases.
    Lower is better; 0.0 means fully monotonic degradation trajectory.
    """
    soh = np.asarray(soh_sequence, dtype=np.float64).flatten()
    if len(soh) < 2:
        return 0.0
    increases = np.sum(soh[1:] > soh[:-1])
    return float(increases / (len(soh) - 1))
