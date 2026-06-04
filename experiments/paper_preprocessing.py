"""Paper-exact feature pipeline: ICA (dQ/dV), DV (dV/dQ), DC (dI/dV) on a fixed voltage grid."""

import numpy as np
import scipy.signal as signal

from experiments.paper_config import (
    PAPER_SEQ_LEN,
    PAPER_VOLTAGE_MAX,
    PAPER_VOLTAGE_MIN,
    SG_POLYORDER,
    SG_WINDOW,
)


def smooth_curve(y, window_length=SG_WINDOW, polyorder=SG_POLYORDER):
    if len(y) <= window_length:
        window_length = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
    if window_length < 3:
        return y
    try:
        return signal.savgol_filter(y, window_length=window_length, polyorder=polyorder)
    except Exception:
        kernel = np.ones(max(3, window_length // 2)) / max(3, window_length // 2)
        return np.convolve(y, kernel, mode="same")


def _min_max_scale(x):
    xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _interp_on_voltage_grid(voltage, values, n_points=PAPER_SEQ_LEN, v_min=PAPER_VOLTAGE_MIN, v_max=PAPER_VOLTAGE_MAX):
    """Interpolate capacity/current onto uniform voltage grid (paper Section 3)."""
    v_smooth = smooth_curve(np.asarray(voltage, dtype=np.float64))
    val_smooth = smooth_curve(np.asarray(values, dtype=np.float64))
    order = np.argsort(v_smooth)
    v_sorted = v_smooth[order]
    val_sorted = val_smooth[order]

    # Collapse duplicate voltage samples
    v_unique, idx = np.unique(v_sorted, return_index=True)
    val_unique = val_sorted[idx]
    if len(v_unique) < 4:
        return None, None

    grid_v = np.linspace(v_min, v_max, n_points)
    val_grid = np.interp(grid_v, v_unique, val_unique, left=val_unique[0], right=val_unique[-1])
    return grid_v, val_grid


def apply_voltage_jitter(voltage, jitter_v, rng):
    if jitter_v <= 0:
        return voltage
    return voltage + rng.uniform(-jitter_v, jitter_v, size=len(voltage))


def extract_paper_cycle_tensor(voltage, current, capacity, seq_len=PAPER_SEQ_LEN, rng=None, jitter_v=0.0):
    """
    Returns (3, seq_len) tensor: [ICA, DV, DC] per Scientific Reports (2026) Fig. 2–3.
    """
    voltage = np.asarray(voltage, dtype=np.float64).flatten()
    current = np.asarray(current, dtype=np.float64).flatten()
    capacity = np.asarray(capacity, dtype=np.float64).flatten()
    n = min(len(voltage), len(current), len(capacity))
    if n < 20:
        return None

    voltage, current, capacity = voltage[:n], current[:n], capacity[:n]
    if rng is not None and jitter_v > 0:
        voltage = apply_voltage_jitter(voltage, jitter_v, rng)

    grid_v, q_grid = _interp_on_voltage_grid(voltage, capacity, n_points=seq_len)
    _, i_grid = _interp_on_voltage_grid(voltage, current, n_points=seq_len)
    if grid_v is None or q_grid is None or i_grid is None:
        return None

    v_norm = _min_max_scale(grid_v)
    q_norm = _min_max_scale(q_grid)
    i_norm = _min_max_scale(i_grid)

    dv = np.diff(v_norm)
    dq = np.diff(q_norm)
    di = np.diff(i_norm)

    dv_safe = np.where(np.abs(dv) < 1e-8, 1e-8, dv)
    dq_safe = np.where(np.abs(dq) < 1e-8, 1e-8, dq)

    ica = dq_safe / dv_safe  # dQ/dV
    dv_curve = dv_safe / dq_safe  # dV/dQ
    dc = di / dv_safe  # dI/dV

    ica = smooth_curve(ica)
    dv_curve = smooth_curve(dv_curve)
    dc = smooth_curve(dc)

    # Align derivative length to seq_len
    x_diff = np.linspace(0, 1, len(ica))
    x_full = np.linspace(0, 1, seq_len)
    ica_a = np.interp(x_full, x_diff, _min_max_scale(ica))
    dv_a = np.interp(x_full, x_diff, _min_max_scale(dv_curve))
    dc_a = np.interp(x_full, x_diff, _min_max_scale(dc))

    tensor = np.stack([ica_a, dv_a, dc_a], axis=0).astype(np.float32)
    if not np.all(np.isfinite(tensor)):
        return None
    return np.clip(tensor, -10.0, 10.0)


def sanitize_feature_tensor(tensor):
    """Replace non-finite values before training (stability on real CALCE/NASA logs)."""
    arr = np.asarray(tensor, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(arr, -10.0, 10.0)
