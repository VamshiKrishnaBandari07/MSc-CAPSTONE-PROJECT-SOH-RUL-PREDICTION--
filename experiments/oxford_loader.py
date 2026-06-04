"""Parse Oxford Battery Degradation Dataset 1 (.mat)."""

import os
import re

import numpy as np
import scipy.io

NOMINAL_CAPACITY_MAH = 740.0


def _unwrap_struct(obj):
    while isinstance(obj, np.ndarray) and obj.size == 1:
        obj = obj[0, 0]
    return obj


def _segment_arrays(segment):
    segment = _unwrap_struct(segment)
    voltage = np.asarray(segment.v, dtype=np.float64).flatten()
    capacity_mah = np.asarray(segment.q, dtype=np.float64).flatten()
    if hasattr(segment, "t"):
        time = np.asarray(segment.t, dtype=np.float64).flatten()
    else:
        time = np.arange(len(voltage), dtype=np.float64)
    return voltage, capacity_mah, time


def _cycle_sort_key(name):
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else 0


def iter_oxford_characterisation_cycles(mat_path):
    """
    Yields (voltage, current, capacity_profile_ah, soh) for each 1-C charge
    characterisation segment (C1ch) across Cell1–Cell8.
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    cell_keys = sorted(k for k in mat if k.startswith("Cell"))

    for cell_key in cell_keys:
        cell = _unwrap_struct(mat[cell_key])
        cycle_names = sorted(
            (f for f in cell._fieldnames if str(f).startswith("cyc")),
            key=_cycle_sort_key,
        )
        initial_capacity = None

        for cycle_name in cycle_names:
            cycle = _unwrap_struct(getattr(cell, cycle_name))
            if not hasattr(cycle, "C1ch"):
                continue

            voltage, capacity_mah, time = _segment_arrays(cycle.C1ch)
            if len(voltage) < 10:
                continue

            cap_peak = float(np.nanmax(capacity_mah))
            if cap_peak <= 1e-6:
                continue

            if initial_capacity is None:
                initial_capacity = cap_peak

            soh = float(np.clip(cap_peak / initial_capacity, 0.0, 1.2))
            capacity_ah = capacity_mah / 1000.0

            if len(time) > 1:
                dt = np.diff(time, prepend=time[0])
                dt = np.clip(dt, 1e-6, None)
                dq = np.diff(capacity_mah, prepend=capacity_mah[0])
                current = np.abs(dq / dt) * 3600.0 / 1000.0  # mAh/s -> A
                current = np.clip(current, 0.0, 2.0)
                if np.median(current[current > 0.01]) < 0.05:
                    current = np.ones_like(voltage) * (NOMINAL_CAPACITY_MAH / 1000.0)
            else:
                current = np.ones_like(voltage) * (NOMINAL_CAPACITY_MAH / 1000.0)

            yield voltage, current, capacity_ah, soh, cell_key


def count_oxford_cycles(data_dir):
    mat_path = os.path.join(data_dir, "Oxford_Battery_Degradation_Dataset_1.mat")
    if not os.path.isfile(mat_path):
        return 0
    return sum(1 for _ in iter_oxford_characterisation_cycles(mat_path))
