import glob
import os

import numpy as np
import scipy.io
import scipy.signal as signal

from experiments.data import generate_shared_labels

RANDOM_SEED = 42


def smooth_curve(y, window_length=15, polyorder=3):
    if len(y) <= window_length:
        window_length = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
    if window_length < 3:
        return y
    return signal.savgol_filter(y, window_length=window_length, polyorder=polyorder)


def calculate_ic_dv_curves_paper(voltage, capacity):
    v_smooth = smooth_curve(voltage)
    q_smooth = smooth_curve(capacity)

    dv = np.diff(v_smooth)
    dq = np.diff(q_smooth)

    dq_safe = np.where(np.abs(dq) < 1e-6, 1e-6, dq)
    dv_safe = np.where(np.abs(dv) < 1e-6, 1e-6, dv)

    dq_dv = dq_safe / dv_safe
    dv_dq = dv_safe / dq_safe

    dq_dv_smooth = smooth_curve(dq_dv)
    dv_dq_smooth = smooth_curve(dv_dq)

    length = len(voltage)
    x_orig = np.linspace(0, 1, length)
    x_diff = np.linspace(0, 1, length - 1)

    dq_dv_aligned = np.interp(x_orig, x_diff, dq_dv_smooth)
    dv_dq_aligned = np.interp(x_orig, x_diff, dv_dq_smooth)

    return dq_dv_aligned, dv_dq_aligned


def _min_max_scale(x):
    xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin == 0:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _align_sequence(values, seq_len):
    if len(values) > seq_len:
        idx = np.linspace(0, len(values) - 1, seq_len, dtype=int)
        return values[idx]
    if len(values) < seq_len:
        return np.interp(np.linspace(0, 1, seq_len), np.linspace(0, 1, len(values)), values)
    return values


from experiments.nasa_loader import iter_nasa_discharge_cycles


def _load_nasa_paper_features(data_dir, seq_len=100):
    features, soh_values = [], []

    for voltage, current, capacity_profile, soh in iter_nasa_discharge_cycles(data_dir):
        ica, dva = calculate_ic_dv_curves_paper(voltage, capacity_profile)
        voltage_norm = _min_max_scale(smooth_curve(voltage))
        features.append(
            np.stack(
                [_align_sequence(ica, seq_len), _align_sequence(dva, seq_len), _align_sequence(voltage_norm, seq_len)],
                axis=0,
            )
        )
        soh_values.append(soh)

    if not features:
        return None
    return np.array(features, dtype=np.float32), np.array(soh_values, dtype=np.float32)


def generate_paper_synthetic_data(dataset_name="NASA", num_cycles=150, seq_len=100):
    if dataset_name == "NASA":
        noise = 0.003
    elif dataset_name == "Oxford":
        noise = 0.002
    else:
        noise = 0.004

    soh_array, _, _ = generate_shared_labels(dataset_name, num_cycles)
    rng = np.random.default_rng(RANDOM_SEED + hash(dataset_name) % 1000)
    base_v = np.linspace(3.0, 4.2, seq_len)
    data = []

    for cycle in range(num_cycles):
        soh = float(soh_array[cycle])
        current_cap = 2.0 * soh
        peak_shift = 0.1 * (1.0 - soh)
        base_q = current_cap * (1.0 / (1.0 + np.exp(-10 * (base_v - 3.6 + peak_shift))))

        raw_v = base_v + rng.normal(0, 0.005, seq_len)
        raw_q = base_q + rng.normal(0, noise, seq_len)
        ica, dva = calculate_ic_dv_curves_paper(raw_v, raw_q)
        voltage_norm = _min_max_scale(raw_v)
        data.append(np.stack([ica, dva, voltage_norm], axis=0))

    return np.array(data, dtype=np.float32), soh_array


class PaperDatasetLoader:
    @staticmethod
    def load_dataset(dataset_name="NASA", raw_path=None, num_cycles=150, seq_len=100):
        if raw_path is None:
            raw_path = os.path.join(os.getcwd(), "data", dataset_name)

        print(f"[Paper | {dataset_name}] Loading from: {raw_path}")

        if dataset_name == "NASA" and os.path.isdir(raw_path):
            nasa = _load_nasa_paper_features(raw_path, seq_len)
            if nasa is not None:
                print(f"[Paper | {dataset_name}] Loaded {len(nasa[0])} cycles from .mat files.")
                return nasa

        print(f"[Paper | {dataset_name}] Using paper-aligned synthetic simulator.")
        return generate_paper_synthetic_data(dataset_name, num_cycles, seq_len)


if __name__ == "__main__":
    print("--- Paper Preprocessing Verification ---")
    for ds in ["NASA", "Oxford", "CALCE"]:
        features, soh = PaperDatasetLoader.load_dataset(ds, num_cycles=10)
        print(f"[{ds}] Features: {features.shape} | SOH: {soh.shape}")
    print("Verification Successful!")
