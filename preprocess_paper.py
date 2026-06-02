import numpy as np
import scipy.signal as signal

RANDOM_SEED = 42

def smooth_curve(y, window_length=15, polyorder=3):
    """
    Savitzky-Golay filtering for denoising raw measurements exactly as described in the paper.
    """
    if len(y) <= window_length:
        window_length = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
    if window_length < 3:
        return y
    return signal.savgol_filter(y, window_length=window_length, polyorder=polyorder)

def calculate_ic_dv_curves_paper(voltage, capacity):
    """
    Extracts the core physical indicators described in the paper:
    1. Incremental Capacity (ICA, dQ/dV)
    2. Differential Voltage (DVA, dV/dQ)
    Both are smoothed and aligned back to the original sequence length.
    """
    v_smooth = smooth_curve(voltage)
    q_smooth = smooth_curve(capacity)
    
    dv = np.diff(v_smooth)
    dq = np.diff(q_smooth)
    
    # Safe finite difference division
    dq_safe = np.where(np.abs(dq) < 1e-6, 1e-6, dq)
    dv_safe = np.where(np.abs(dv) < 1e-6, 1e-6, dv)
    
    dq_dv = dq_safe / dv_safe
    dv_dq = dv_safe / dq_safe
    
    dq_dv_smooth = smooth_curve(dq_dv)
    dv_dq_smooth = smooth_curve(dv_dq)
    
    L = len(voltage)
    x_orig = np.linspace(0, 1, L)
    x_diff = np.linspace(0, 1, L - 1)
    
    dq_dv_aligned = np.interp(x_orig, x_diff, dq_dv_smooth)
    dv_dq_aligned = np.interp(x_orig, x_diff, dv_dq_smooth)
    
    return dq_dv_aligned, dv_dq_aligned

def generate_paper_synthetic_data(num_cycles=100, seq_len=100):
    """
    Generates standard synthetic battery cycle features exactly reflecting paper profiles:
    - Input features: (ICA, DVA, Voltage sweep)
    - Labels: SOH values
    """
    data = []
    soh_values = []
    
    rng = np.random.default_rng(RANDOM_SEED)
    base_v = np.linspace(3.0, 4.2, seq_len)
    
    for cycle in range(num_cycles):
        soh = 1.0 - 0.25 * (cycle / num_cycles)**1.2 + rng.normal(0, 0.003)
        soh = np.clip(soh, 0.5, 1.0)
        soh_values.append(soh)
        
        current_cap = 2.0 * soh
        peak_shift = 0.1 * (1.0 - soh)
        base_q = current_cap * (1.0 / (1.0 + np.exp(-10 * (base_v - 3.6 + peak_shift))))
        
        raw_v = base_v + rng.normal(0, 0.005, seq_len)
        raw_q = base_q + rng.normal(0, 0.002, seq_len)
        
        ica, dva = calculate_ic_dv_curves_paper(raw_v, raw_q)
        
        # 3 features: ICA (dQ/dV), DVA (dV/dQ), Voltage Sweep (V)
        cycle_features = np.stack([ica, dva, raw_v], axis=0)
        data.append(cycle_features)
        
    return np.array(data, dtype=np.float32), np.array(soh_values, dtype=np.float32)

if __name__ == '__main__':
    data, soh = generate_paper_synthetic_data(10, 100)
    print("--- Exact Preprocessing Scaffolding Verified ---")
    print(f"Data shape: {data.shape} | SOH shape: {soh.shape}")
