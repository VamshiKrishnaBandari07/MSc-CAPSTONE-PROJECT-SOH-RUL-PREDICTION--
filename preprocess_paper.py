"""Experiment A — Paper-aligned preprocessing (Scientific Reports 2026).



Multi-domain input channels per the paper:

  - ICA: dQ/dV (Incremental Capacity Analysis)

  - DV:  dV/dQ (Differential Voltage)

  - DC:  dI/dV (Differential Current)



Voltage grid: 300 points from 2.5 V to 4.2 V, Savitzky–Golay (window=15, order=3).

"""



import os



import numpy as np



from experiments.calce_loader import iter_calce_all_cells

from experiments.data import dataset_rng, generate_shared_labels

from experiments.nasa_loader import iter_nasa_discharge_cycles

from experiments.oxford_loader import iter_oxford_characterisation_cycles

from experiments.paper_config import PAPER_SEQ_LEN, PAPER_VOLTAGE_JITTER_V, PAPER_VOLTAGE_MAX, PAPER_VOLTAGE_MIN

from experiments.paper_preprocessing import extract_paper_cycle_tensor, sanitize_feature_tensor





def _cycles_to_tensors(cycle_iter, seq_len=PAPER_SEQ_LEN, rng=None, training_jitter=False):

    features, soh_values = [], []

    jitter = PAPER_VOLTAGE_JITTER_V if training_jitter else 0.0



    for voltage, current, capacity_profile, soh in cycle_iter:

        tensor = extract_paper_cycle_tensor(

            voltage, current, capacity_profile, seq_len=seq_len, rng=rng, jitter_v=jitter

        )

        if tensor is None:

            continue

        features.append(sanitize_feature_tensor(tensor))

        soh_values.append(soh)



    if not features:

        return None

    return np.array(features, dtype=np.float32), np.array(soh_values, dtype=np.float32)





def _load_nasa_paper_features(data_dir, seq_len=PAPER_SEQ_LEN):

    return _cycles_to_tensors(iter_nasa_discharge_cycles(data_dir), seq_len=seq_len)





def _load_oxford_paper_features(data_dir, seq_len=PAPER_SEQ_LEN):

    mat_path = os.path.join(data_dir, "Oxford_Battery_Degradation_Dataset_1.mat")

    if not os.path.isfile(mat_path):

        return None

    return _cycles_to_tensors(iter_oxford_characterisation_cycles(mat_path), seq_len=seq_len)





def _load_calce_paper_features(data_dir, seq_len=PAPER_SEQ_LEN):

    return _cycles_to_tensors(iter_calce_all_cells(data_dir), seq_len=seq_len)





def generate_paper_synthetic_data(dataset_name="NASA", num_cycles=150, seq_len=PAPER_SEQ_LEN):

    """Paper-aligned synthetic fallback when raw files are not downloaded."""

    soh_array, _ = generate_shared_labels(dataset_name, num_cycles)

    rng = dataset_rng(dataset_name)

    base_v = np.linspace(PAPER_VOLTAGE_MIN, PAPER_VOLTAGE_MAX, seq_len)

    data = []



    for cycle in range(num_cycles):

        soh = float(soh_array[cycle])

        current_cap = 2.0 * soh

        peak_shift = 0.1 * (1.0 - soh)

        base_q = current_cap * (1.0 / (1.0 + np.exp(-10 * (base_v - 3.6 + peak_shift))))

        raw_v = base_v + rng.normal(0, 0.005, seq_len)

        raw_i = np.ones(seq_len) * 1.5 + rng.normal(0, 0.02, seq_len)

        raw_q = base_q + rng.normal(0, 0.003, seq_len)

        tensor = extract_paper_cycle_tensor(raw_v, raw_i, raw_q, seq_len=seq_len)

        if tensor is not None:

            data.append(tensor)



    return np.array(data, dtype=np.float32), soh_array





class PaperDatasetLoader:

    @staticmethod

    def load_dataset(dataset_name="NASA", raw_path=None, num_cycles=150, seq_len=PAPER_SEQ_LEN, require_real=False):

        if raw_path is None:
            raw_path = os.path.join(os.getcwd(), "data", dataset_name)

        print(f"[Paper | {dataset_name}] Loading from: {raw_path}")

        loaded = None
        if dataset_name == "NASA" and os.path.isdir(raw_path):
            loaded = _load_nasa_paper_features(raw_path, seq_len)
        elif dataset_name == "Oxford" and os.path.isdir(raw_path):
            loaded = _load_oxford_paper_features(raw_path, seq_len)
        elif dataset_name == "CALCE" and os.path.isdir(raw_path):
            loaded = _load_calce_paper_features(raw_path, seq_len)

        if loaded is not None:
            print(f"[Paper | {dataset_name}] Loaded {len(loaded[0])} cycles (ICA+DV+DC, grid={seq_len}).")
            return loaded

        if require_real:
            raise FileNotFoundError(
                f"[Paper | {dataset_name}] Real data required but not found under {raw_path}. "
                "Run: python download_data.py --all"
            )

        print(f"[Paper | {dataset_name}] WARNING: Using synthetic fallback (not comparable to paper 0.021).")
        return generate_paper_synthetic_data(dataset_name, num_cycles, seq_len)





if __name__ == "__main__":

    print("--- Paper preprocessing (ICA + DV + DC) ---")

    for ds in ["NASA", "Oxford", "CALCE"]:

        features, soh = PaperDatasetLoader.load_dataset(ds, num_cycles=10)

        print(f"[{ds}] Features: {features.shape} | SOH: {soh.shape}")


