import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import scipy.signal as signal


DATASET_PROFILES = {
    "NASA": {
        "nominal_capacity": 2.0,
        "eol_soh": 0.70,
        "fade": 0.30,
        "noise": 0.006,
        "current": 1.5,
        "voltage_min": 3.20,
        "voltage_max": 4.20,
    },
    "Oxford": {
        "nominal_capacity": 0.74,
        "eol_soh": 0.80,
        "fade": 0.22,
        "noise": 0.004,
        "current": 0.74,
        "voltage_min": 3.00,
        "voltage_max": 4.20,
    },
    "CALCE": {
        "nominal_capacity": 1.10,
        "eol_soh": 0.75,
        "fade": 0.26,
        "noise": 0.007,
        "current": 1.1,
        "voltage_min": 3.10,
        "voltage_max": 4.20,
    },
}


@dataclass
class PaperDatasetBundle:
    """Aligned feature tensor and labels for paper-style cross-dataset training."""

    features: np.ndarray
    soh: np.ndarray
    dataset_names: np.ndarray
    cycle_indices: np.ndarray


def _odd_window(length: int, requested: int) -> int:
    window = min(requested, length - 1 if length % 2 == 0 else length)
    if window % 2 == 0:
        window -= 1
    return max(window, 3)


def smooth_curve(values: np.ndarray, window_length: int = 15, polyorder: int = 3) -> np.ndarray:
    """Denoise charge-cycle signals before finite-difference feature extraction."""

    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values

    window = _odd_window(values.size, window_length)
    if window <= polyorder:
        return values
    return signal.savgol_filter(values, window_length=window, polyorder=polyorder)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    safe = np.where(np.abs(denominator) < 1e-8, np.sign(denominator) * 1e-8 + 1e-8, denominator)
    return numerator / safe


def _min_max_scale(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def _align_to_voltage_grid(values: np.ndarray, voltage_axis: np.ndarray, voltage_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(voltage_axis)
    sorted_axis = voltage_axis[order]
    sorted_values = values[order]

    unique_axis, unique_idx = np.unique(sorted_axis, return_index=True)
    unique_values = sorted_values[unique_idx]
    if unique_axis.size < 2:
        return np.full_like(voltage_grid, float(unique_values[0]) if unique_values.size else 0.0)
    return np.interp(voltage_grid, unique_axis, unique_values)


def extract_dv_dc_ica_features(
    voltage: Sequence[float],
    capacity: Sequence[float],
    current: Sequence[float],
    seq_len: int = 128,
    smoothing_window: int = 15,
) -> np.ndarray:
    """
    Extract the paper's synchronized multi-domain tensor:
    ICA=dQ/dV, DV=dV/dQ, and DC=dI/dV on a common voltage grid.
    """

    voltage = np.asarray(voltage, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if not (voltage.size == capacity.size == current.size):
        raise ValueError("voltage, capacity, and current must have the same length")
    if voltage.size < 4:
        raise ValueError("at least four samples are required to compute differential features")

    v_smooth = smooth_curve(voltage, window_length=smoothing_window)
    q_smooth = smooth_curve(capacity, window_length=smoothing_window)
    i_smooth = smooth_curve(current, window_length=smoothing_window)

    dv = np.diff(v_smooth)
    dq = np.diff(q_smooth)
    di = np.diff(i_smooth)

    ica = smooth_curve(_safe_divide(dq, dv), window_length=smoothing_window)
    differential_voltage = smooth_curve(_safe_divide(dv, dq), window_length=smoothing_window)
    differential_current = smooth_curve(_safe_divide(di, dv), window_length=smoothing_window)

    voltage_midpoints = 0.5 * (v_smooth[:-1] + v_smooth[1:])
    voltage_grid = np.linspace(float(np.min(v_smooth)), float(np.max(v_smooth)), seq_len)

    aligned = [
        _min_max_scale(_align_to_voltage_grid(ica, voltage_midpoints, voltage_grid)),
        _min_max_scale(_align_to_voltage_grid(differential_voltage, voltage_midpoints, voltage_grid)),
        _min_max_scale(_align_to_voltage_grid(differential_current, voltage_midpoints, voltage_grid)),
    ]
    return np.stack(aligned, axis=0).astype(np.float32)


def _simulate_cycle(profile: dict, soh: float, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    voltage = np.linspace(profile["voltage_min"], profile["voltage_max"], seq_len)
    progress_loss = 1.0 - soh

    peak_shift = 0.18 * progress_loss
    plateau_width = 10.0 - 2.0 * progress_loss
    capacity_curve = profile["nominal_capacity"] * soh
    capacity = capacity_curve / (1.0 + np.exp(-plateau_width * (voltage - 3.65 + peak_shift)))

    cv_taper = 1.0 - 0.35 / (1.0 + np.exp(-18.0 * (voltage - 4.02)))
    kinetic_ripple = 0.015 * np.sin(np.linspace(0.0, 4.0 * np.pi, seq_len) + 5.0 * progress_loss)
    current = profile["current"] * cv_taper + kinetic_ripple

    noisy_voltage = voltage + rng.normal(0.0, profile["noise"], seq_len)
    noisy_capacity = capacity + rng.normal(0.0, profile["noise"] * profile["nominal_capacity"], seq_len)
    noisy_current = current + rng.normal(0.0, profile["noise"], seq_len)
    return extract_dv_dc_ica_features(noisy_voltage, noisy_capacity, noisy_current, seq_len=seq_len)


def generate_paper_synthetic_dataset(
    dataset_name: str,
    num_cycles: int,
    seq_len: int,
    seed: Optional[int] = None,
) -> PaperDatasetBundle:
    """
    Generate a calibrated fallback when public raw files are unavailable locally.

    The synthetic curves follow the paper methodology (DV/DC/ICA after denoising and
    voltage-grid alignment), but they are not a substitute for the real NASA/Oxford/CALCE files.
    """

    if dataset_name not in DATASET_PROFILES:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of {sorted(DATASET_PROFILES)}")

    rng = np.random.default_rng(seed)
    profile = DATASET_PROFILES[dataset_name]
    features = []
    soh_values = []
    cycle_indices = np.arange(num_cycles, dtype=np.int32)

    for cycle in cycle_indices:
        progress = cycle / max(num_cycles - 1, 1)
        base_fade = profile["fade"] * (progress ** 1.28)
        slow_fade = 0.025 * progress
        local_recovery = 0.006 * np.sin(cycle / 19.0)
        soh = np.clip(1.0 - base_fade - slow_fade + local_recovery, profile["eol_soh"] - 0.08, 1.0)

        features.append(_simulate_cycle(profile, float(soh), seq_len, rng))
        soh_values.append(float(soh))

    return PaperDatasetBundle(
        features=np.asarray(features, dtype=np.float32),
        soh=np.asarray(soh_values, dtype=np.float32),
        dataset_names=np.asarray([dataset_name] * num_cycles),
        cycle_indices=cycle_indices,
    )


def _load_npz_dataset(path: str, dataset_name: str) -> Optional[PaperDatasetBundle]:
    if not os.path.exists(path):
        return None

    with np.load(path, allow_pickle=False) as loaded:
        if "features" not in loaded or "soh" not in loaded:
            raise ValueError(f"{path} must contain 'features' and 'soh' arrays")
        features = loaded["features"].astype(np.float32)
        soh = loaded["soh"].astype(np.float32)
        dataset_names = loaded["dataset_names"] if "dataset_names" in loaded else np.asarray([dataset_name] * len(soh))
        cycle_indices = loaded["cycle_indices"] if "cycle_indices" in loaded else np.arange(len(soh), dtype=np.int32)

    if features.ndim != 3 or features.shape[1] != 3:
        raise ValueError(f"{path} features must have shape [cycles, 3, seq_len]")
    if len(features) != len(soh):
        raise ValueError(f"{path} features and soh arrays must have matching first dimension")

    return PaperDatasetBundle(features, soh, dataset_names.astype(str), cycle_indices.astype(np.int32))


def load_paper_experiment_data(
    dataset_names: Iterable[str],
    cycles_per_dataset: int,
    seq_len: int,
    raw_dir: Optional[str] = None,
    seed: int = 42,
) -> PaperDatasetBundle:
    """
    Load prepared paper experiment arrays if present, otherwise synthesize calibrated data.

    Expected real-data format per dataset:
    `data/processed/<DATASET>_paper_exp.npz` with arrays `features` [N,3,L] and `soh` [N].
    """

    bundles = []
    for offset, dataset_name in enumerate(dataset_names):
        npz_path = None
        if raw_dir:
            npz_path = os.path.join(raw_dir, "processed", f"{dataset_name}_paper_exp.npz")
        bundle = _load_npz_dataset(npz_path, dataset_name) if npz_path else None
        if bundle is None:
            print(f"[{dataset_name}] Prepared NPZ not found; using paper-style synthetic fallback.")
            bundle = generate_paper_synthetic_dataset(
                dataset_name=dataset_name,
                num_cycles=cycles_per_dataset,
                seq_len=seq_len,
                seed=seed + offset,
            )
        bundles.append(bundle)

    return PaperDatasetBundle(
        features=np.concatenate([bundle.features for bundle in bundles], axis=0),
        soh=np.concatenate([bundle.soh for bundle in bundles], axis=0),
        dataset_names=np.concatenate([bundle.dataset_names for bundle in bundles], axis=0),
        cycle_indices=np.concatenate([bundle.cycle_indices for bundle in bundles], axis=0),
    )


if __name__ == "__main__":
    bundle = load_paper_experiment_data(("NASA", "Oxford", "CALCE"), cycles_per_dataset=8, seq_len=64)
    print("--- paper_exp preprocessing verification ---")
    print(f"Features: {bundle.features.shape} | SOH: {bundle.soh.shape}")
    print(f"Datasets: {sorted(set(bundle.dataset_names.tolist()))}")

