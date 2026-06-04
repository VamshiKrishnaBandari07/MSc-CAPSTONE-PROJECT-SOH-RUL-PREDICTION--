"""Shared NASA PCoE .mat file parsing for discharge cycles."""

import glob
import os

import numpy as np
import scipy.io


def _normalize_cycle_type(cycle):
    if "type" not in cycle.dtype.names:
        return ""
    return str(cycle["type"][0]).lower().strip()


def _extract_cycles_from_mat(mat_path):
    mat = scipy.io.loadmat(mat_path)
    if "cycle" in mat:
        return mat["cycle"][0]

    stem = os.path.splitext(os.path.basename(mat_path))[0]
    if stem in mat:
        cell = mat[stem][0, 0]
        if hasattr(cell, "dtype") and "cycle" in cell.dtype.names:
            return cell["cycle"][0]
    return None


def _build_capacity_profile(data, voltage, current, cap_scalar):
    if "Time" in data.dtype.names:
        time = np.asarray(data["Time"][0], dtype=np.float64).flatten()
        if len(time) == len(current) and len(time) > 1:
            dt = np.diff(time, prepend=time[0])
            dt = np.clip(dt, 0, None)
        else:
            dt = np.ones(len(current))
    else:
        dt = np.ones(len(current))

    q = np.cumsum(np.abs(current) * dt / 3600.0)
    peak = float(np.max(q)) if len(q) else 0.0
    if peak > 1e-9:
        q = q / peak * cap_scalar
    else:
        q = np.linspace(0, cap_scalar, len(voltage))
    return q


def iter_nasa_discharge_cycles_from_file(mat_path):
    """
    Yields (voltage, current, capacity_profile, soh_scalar) for discharge cycles
    in a single NASA .mat file. SOH is normalised to the first discharge capacity
    of that cell.
    """
    cycles = _extract_cycles_from_mat(mat_path)
    if cycles is None:
        return

    initial_capacity = None
    for cycle in cycles:
        if _normalize_cycle_type(cycle) not in ("discharge", "d"):
            continue

        data = cycle["data"][0, 0]
        if "Capacity" not in data.dtype.names:
            continue

        cap_scalar = float(np.asarray(data["Capacity"][0], dtype=np.float64).flatten()[0])
        voltage = np.asarray(data["Voltage_measured"][0], dtype=np.float64).flatten()
        current = np.asarray(data["Current_measured"][0], dtype=np.float64).flatten()

        if len(voltage) < 10 or cap_scalar <= 0:
            continue

        if initial_capacity is None:
            initial_capacity = cap_scalar

        soh = float(np.clip(cap_scalar / initial_capacity, 0.0, 1.0))
        capacity_profile = _build_capacity_profile(data, voltage, current, cap_scalar)

        yield voltage, current, capacity_profile, soh


def iter_nasa_discharge_cycles(data_dir):
    """
    Yields (voltage, current, capacity_profile, soh_scalar) for each discharge cycle
    across all B*.mat files in data_dir.
    """
    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    for mat_path in mat_files:
        yield from iter_nasa_discharge_cycles_from_file(mat_path)


def count_nasa_discharge_cycles(data_dir):
    """Return total discharge cycles across all .mat files in data_dir."""
    return sum(1 for _ in iter_nasa_discharge_cycles(data_dir))
