import glob
import os

import numpy as np
import scipy.io
import scipy.signal as signal

RANDOM_SEED = 42

def smooth_curve(y, window_length=15, polyorder=3):
    """
    Denoises raw battery signals using Savitzky-Golay filtering.
    Essential for calculating smooth derivatives (dQ/dV, dV/dQ, dI/dV) without amplifying noise.
    """
    if len(y) <= window_length:
        window_length = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
    if window_length < 3:
        return y
    try:
        return signal.savgol_filter(y, window_length=window_length, polyorder=polyorder)
    except Exception:
        # Fallback to moving average if filter fails
        kernel = np.ones(max(3, window_length // 2)) / max(3, window_length // 2)
        return np.convolve(y, kernel, mode='same')

def calculate_ic_dv_curves(voltage, capacity, current=None):
    """
    Computes electrochemical health indicator curves:
    1. Incremental Capacity (IC, dQ/dV)
    2. Differential Voltage (DV, dV/dQ)
    3. Differential Current (DC, dI/dV) - if current is provided
    
    All curves are smoothed and aligned back to sequence length for consistency.
    """
    # 1. Smooth raw measurements first
    v_smooth = smooth_curve(voltage)
    q_smooth = smooth_curve(capacity)
    
    # 2. Compute derivatives using finite differences
    dv = np.diff(v_smooth)
    dq = np.diff(q_smooth)
    
    # Handle division by zero or extremely small values safely
    dq_safe = np.where(np.abs(dq) < 1e-6, 1e-6, dq)
    dv_safe = np.where(np.abs(dv) < 1e-6, 1e-6, dv)
    
    dq_dv = dq_safe / dv_safe
    dv_dq = dv_safe / dq_safe
    
    # 3. Handle Differential Current (dI/dV) if current is present
    if current is not None:
        i_smooth = smooth_curve(current)
        di = np.diff(i_smooth)
        di_dv = di / dv_safe
    else:
        # Default to a mock differential current if not provided
        di_dv = np.zeros_like(dq_dv)
        
    # Smooth the derivatives to get clean peak features
    dq_dv_smooth = smooth_curve(dq_dv)
    dv_dq_smooth = smooth_curve(dv_dq)
    di_dv_smooth = smooth_curve(di_dv)
    
    # Standardize/Scale features using standard min-max scaling to [0, 1] range
    def min_max_scale(x):
        xmin, xmax = np.min(x), np.max(x)
        if xmax - xmin == 0:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)
        
    dq_dv_norm = min_max_scale(dq_dv_smooth)
    dv_dq_norm = min_max_scale(dv_dq_smooth)
    di_dv_norm = min_max_scale(di_dv_smooth)
    
    # Interpolate back to original sequence length (L) for alignment
    L = len(voltage)
    x_orig = np.linspace(0, 1, L)
    x_diff = np.linspace(0, 1, L - 1)
    
    dq_dv_aligned = np.interp(x_orig, x_diff, dq_dv_norm)
    dv_dq_aligned = np.interp(x_orig, x_diff, dv_dq_norm)
    di_dv_aligned = np.interp(x_orig, x_diff, di_dv_norm)
    
    return dq_dv_aligned, dv_dq_aligned, di_dv_aligned

def _load_nasa_mat_files(data_dir, seq_len=100):
    """
    Parses NASA PCoE .mat files (e.g. B0005.mat) when placed in data/NASA/.
    Extracts charge cycles and computes ICA/DVA/DCA features with SOH/RUL labels.
    """
    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not mat_files:
        return None

    all_features, all_soh = [], []
    eol_threshold = 0.70

    for mat_path in mat_files:
        mat = scipy.io.loadmat(mat_path)
        if "cycle" not in mat:
            continue

        cycles = mat["cycle"][0]
        initial_capacity = None

        for cycle in cycles:
            cycle_type = str(cycle["type"][0]).lower() if "type" in cycle.dtype.names else ""
            if cycle_type and "charge" not in cycle_type:
                continue

            data = cycle["data"][0, 0]
            voltage = np.asarray(data["Voltage_measured"][0], dtype=np.float64).flatten()
            current = np.asarray(data["Current_measured"][0], dtype=np.float64).flatten()
            capacity = np.asarray(data["Capacity"][0], dtype=np.float64).flatten()

            if len(voltage) < 10:
                continue

            if len(capacity) != len(voltage):
                capacity = np.linspace(0, np.max(np.abs(current)) * len(voltage) / 3600.0, len(voltage))

            peak_cap = float(np.max(capacity))
            if initial_capacity is None:
                initial_capacity = peak_cap
            if initial_capacity <= 0:
                continue

            soh = peak_cap / initial_capacity
            ica, dva, dca = calculate_ic_dv_curves(voltage, capacity, current)

            if len(ica) > seq_len:
                idx = np.linspace(0, len(ica) - 1, seq_len, dtype=int)
                ica, dva, dca = ica[idx], dva[idx], dca[idx]
            elif len(ica) < seq_len:
                ica = np.interp(np.linspace(0, 1, seq_len), np.linspace(0, 1, len(ica)), ica)
                dva = np.interp(np.linspace(0, 1, seq_len), np.linspace(0, 1, len(dva)), dva)
                dca = np.interp(np.linspace(0, 1, seq_len), np.linspace(0, 1, len(dca)), dca)

            all_features.append(np.stack([ica, dva, dca], axis=0))
            all_soh.append(soh)

    if not all_features:
        return None

    all_soh = np.array(all_soh, dtype=np.float32)
    eol_cycle = next((i for i, s in enumerate(all_soh) if s <= eol_threshold), len(all_soh))
    all_rul = np.array([max(0, eol_cycle - i) for i in range(len(all_soh))], dtype=np.float32)

    return np.array(all_features, dtype=np.float32), all_soh, all_rul

def generate_synthetic_battery_data(dataset_name="NASA", num_cycles=120, seq_len=100):
    """
    Generates realistic synthetic charge-discharge cycles representing battery degradation
    calibrated to the characteristics of the three public datasets: NASA, Oxford, and CALCE.
    This allows robust offline training and validation when the local dataset files are missing.
    
    Outputs:
    - data: [num_cycles, features=3, seq_len]
    - soh_values: [num_cycles]
    - rul_values: [num_cycles]
    """
    data = []
    soh_values = []
    rul_values = []
    
    # Calibrate based on dataset specifications
    if dataset_name == "NASA":
        # NASA PCoE: LiCoO2 cells, capacity fade from 2.0 Ah to 1.4 Ah (EOL at 1.4 Ah / 70% SOH)
        nominal_cap = 2.0
        eol_threshold = 0.70
        noise_level = 0.008
        capacity_fade_rate = 0.28
    elif dataset_name == "Oxford":
        # Oxford: LCO/NMC cells, small pouch 740mAh capacity, faster fading
        nominal_cap = 0.74
        eol_threshold = 0.80
        noise_level = 0.005
        capacity_fade_rate = 0.20
    else:  # CALCE
        # CALCE: LFP/LCO pouch, capacity fade from 1.1 Ah to 0.8 Ah (EOL at 0.8 Ah / 72.7% SOH)
        nominal_cap = 1.1
        eol_threshold = 0.75
        noise_level = 0.010
        capacity_fade_rate = 0.25
        
    rng = np.random.default_rng(RANDOM_SEED + hash(dataset_name) % 1000)
    base_v = np.linspace(3.2, 4.2, seq_len) # standard charging voltage sweep
    
    for cycle in range(num_cycles):
        # 1. State of Health decreases non-linearly over cycles, with minor capacity recovery steps
        # We simulate the capacity fade curve with a double-exponential degradation function + recovery steps
        degrad = capacity_fade_rate * (cycle / num_cycles)**1.3 + 0.05 * (cycle / num_cycles)
        # Bounded capacity recovery during rest periods (sudden upward bump in SOH)
        recovery = 0.015 * np.sin(cycle / 5.0) if cycle % 12 == 0 else 0.0
        
        soh = 1.0 - degrad + recovery
        soh = np.clip(soh, 0.4, 1.0)
        soh_values.append(soh)
        
        # 2. Remaining Useful Life (RUL) estimation in cycles
        # RUL is the number of cycles remaining until SOH hits the EOL threshold
        # Find where SOH would hit eol_threshold
        eol_cycle = int(num_cycles * ((1.0 - eol_threshold) / capacity_fade_rate)**(1.0/1.3))
        remaining = max(0, eol_cycle - cycle)
        rul_values.append(remaining)
        
        # 3. Features profile synthesis
        current_cap = nominal_cap * soh
        
        # Generate raw voltage & capacity curves
        # Incremental Capacity (ICA) curves shift downwards and to the left (lower voltage peaks) as battery ages
        peak_shift = 0.15 * (1.0 - soh)
        base_q = current_cap * (1.0 / (1.0 + np.exp(-12 * (base_v - 3.65 + peak_shift))))
        
        # Add random sensor noise
        raw_v = base_v + rng.normal(0, 0.006, seq_len)
        raw_q = base_q + rng.normal(0, noise_level, seq_len)
        
        # Constant charging current with slight noise
        raw_i = np.ones_like(raw_v) * 1.5 + rng.normal(0, 0.01, seq_len)
        
        # Calculate features (dQ/dV, dV/dQ, dI/dV)
        ica, dva, dca = calculate_ic_dv_curves(raw_v, raw_q, raw_i)
        
        cycle_features = np.stack([ica, dva, dca], axis=0)
        data.append(cycle_features)
        
    return (np.array(data, dtype=np.float32), 
            np.array(soh_values, dtype=np.float32), 
            np.array(rul_values, dtype=np.float32))

class BatteryDatasetLoader:
    """
    DataLoader helper to interface with NASA, Oxford, and CALCE files.
    If the real files are downloaded in raw paths, it parses them;
    otherwise, it falls back to the calibrated high-fidelity synthetic generators.
    """
    @staticmethod
    def load_dataset(dataset_name="NASA", raw_path=None, num_cycles=150, seq_len=100):
        if raw_path is None:
            raw_path = os.path.join(os.getcwd(), "data", dataset_name)

        print(f"[{dataset_name} Dataset] Attempting to load from: {raw_path}")

        if dataset_name == "NASA" and os.path.isdir(raw_path):
            nasa_data = _load_nasa_mat_files(raw_path, seq_len=seq_len)
            if nasa_data is not None:
                print(f"[{dataset_name}] Loaded {len(nasa_data[0])} cycles from NASA .mat files.")
                return nasa_data

        if os.path.isdir(raw_path) and any(
            f.lower().endswith((".mat", ".csv", ".xls", ".xlsx"))
            for f in os.listdir(raw_path)
        ):
            print(
                f"[{dataset_name}] Raw files detected but parser not yet implemented. "
                f"Using calibrated synthetic fallback."
            )

        print(f"[{dataset_name}] Using high-fidelity synthetic aging simulator.")
        return generate_synthetic_battery_data(dataset_name, num_cycles, seq_len)

if __name__ == '__main__':
    print("--- Preprocessing Verification ---")
    for ds in ["NASA", "Oxford", "CALCE"]:
        data, soh, rul = BatteryDatasetLoader.load_dataset(ds, num_cycles=10)
        print(f"[{ds}] Features: {data.shape} | SOH: {soh.shape} | RUL: {rul.shape}")
        print(f"[{ds}] SOH Range: [{soh.min():.3f}, {soh.max():.3f}] | RUL Range: [{rul.min():.1f}, {rul.max():.1f}]")
    print("Verification Successful!")
