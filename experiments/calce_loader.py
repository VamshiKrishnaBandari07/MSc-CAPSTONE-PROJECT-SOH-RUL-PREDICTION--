"""Parse CALCE CS2 battery Excel logs (.xlsx)."""

import glob
import os

import numpy as np
import pandas as pd

EOL_SOH_THRESHOLD = 0.70
CHANNEL_PREFIX = "Channel_"


def _list_cell_dirs(data_dir):
    """Return folders like data/CALCE/CS2_35/CS2_35 containing xlsx logs."""
    dirs = []
    if not os.path.isdir(data_dir):
        return dirs

    for entry in sorted(os.listdir(data_dir)):
        cell_root = os.path.join(data_dir, entry)
        if not os.path.isdir(cell_root):
            continue
        nested = os.path.join(cell_root, entry)
        search_dir = nested if os.path.isdir(nested) else cell_root
        if any(f.lower().endswith((".xls", ".xlsx")) for f in os.listdir(search_dir)):
            dirs.append(search_dir)
    return dirs


def _read_cell_dataframe(cell_dir):
    files = sorted(
        glob.glob(os.path.join(cell_dir, "*.xlsx")) + glob.glob(os.path.join(cell_dir, "*.xls")),
        key=os.path.getmtime,
    )
    if not files:
        return None

    frames = []
    for path in files:
        xl = pd.ExcelFile(path)
        channel_sheets = [s for s in xl.sheet_names if CHANNEL_PREFIX in s]
        if not channel_sheets:
            continue
        for sheet in channel_sheets:
            df = pd.read_excel(path, sheet_name=sheet)
            frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    if "Date_Time" in df.columns:
        df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
        df = df.sort_values("Date_Time")
    df = df.reset_index(drop=True)

    diff = df["Cycle_Index"].diff()
    reset_points = diff < 0
    new_index = df["Cycle_Index"].copy().astype(float)
    prev = 0
    for idx in np.where(reset_points)[0]:
        segment_max = float(df["Cycle_Index"].iloc[prev:idx].max())
        new_index.iloc[idx:] += segment_max
        prev = idx
    df["Cycle_Index"] = new_index.astype(int)
    return df


def _per_cycle_discharge_capacity(df):
    caps = []
    for _, group in df.groupby("Cycle_Index", sort=True):
        discharge = group[group["Current(A)"] < -0.01]
        if len(discharge) < 5:
            continue
        cap = float(discharge["Discharge_Capacity(Ah)"].max() - discharge["Discharge_Capacity(Ah)"].min())
        if cap <= 0:
            cap = float(discharge["Discharge_Capacity(Ah)"].max())
        caps.append(max(cap, 0.0))

    if not caps:
        return []

    cum = np.array(caps, dtype=np.float64)
    if len(cum) > 1 and cum[-1] > cum[0] * 1.5:
        per = np.diff(cum, prepend=0.0)
        per[0] = cum[0]
        return per.tolist()
    return caps


def iter_calce_discharge_cycles(cell_dir):
    """
    Yields (voltage, current, capacity_profile, soh) for each discharge in one CS2 cell folder.
    """
    df = _read_cell_dataframe(cell_dir)
    if df is None or df.empty:
        return

    per_cycle_caps = _per_cycle_discharge_capacity(df)
    if not per_cycle_caps:
        return

    cycle_ids = sorted(df["Cycle_Index"].unique())
    if len(cycle_ids) != len(per_cycle_caps):
        cycle_ids = cycle_ids[: len(per_cycle_caps)]

    initial_capacity = float(per_cycle_caps[0])
    eol_idx = next(
        (i for i, cap in enumerate(per_cycle_caps) if cap / initial_capacity <= EOL_SOH_THRESHOLD),
        len(per_cycle_caps),
    )

    for i, cycle_id in enumerate(cycle_ids):
        group = df[df["Cycle_Index"] == cycle_id]
        discharge = group[group["Current(A)"] < -0.01]
        if len(discharge) < 10:
            continue

        voltage = discharge["Voltage(V)"].to_numpy(dtype=np.float64)
        current = np.abs(discharge["Current(A)"].to_numpy(dtype=np.float64))
        cap_ah = discharge["Discharge_Capacity(Ah)"].to_numpy(dtype=np.float64)
        cap_ah = cap_ah - cap_ah.min()

        cap_scalar = float(per_cycle_caps[i])
        soh = float(np.clip(cap_scalar / initial_capacity, 0.0, 1.2))
        _ = eol_idx  # reserved for RUL in preprocess wrapper

        yield voltage, current, cap_ah, soh


def iter_calce_all_cells(data_dir):
    for cell_dir in _list_cell_dirs(data_dir):
        yield from iter_calce_discharge_cycles(cell_dir)


def count_calce_cycles(data_dir):
    return sum(1 for _ in iter_calce_all_cells(data_dir))
