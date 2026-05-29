import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence
import urllib.error
import urllib.request
import zipfile

import numpy as np
import scipy.io

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional tabular readers are validated at runtime.
    pd = None

try:
    from .preprocess import extract_dv_dc_ica_features
except ImportError:  # Allows `python paper_exp/prepare_data.py` from the repo root.
    from preprocess import extract_dv_dc_ica_features


@dataclass
class RawBatteryCycle:
    dataset_name: str
    cell_id: str
    cycle_index: int
    voltage: np.ndarray
    current: np.ndarray
    capacity_trace: Optional[np.ndarray]
    soh: Optional[float]


VOLTAGE_KEYS = ("voltage_measured", "voltage", "voltage_v", "cell_voltage", "v")
CURRENT_KEYS = ("current_measured", "current", "current_a", "cell_current", "i")
CAPACITY_KEYS = (
    "charge_capacity",
    "discharge_capacity",
    "capacity",
    "capacity_ah",
    "q",
    "ah",
    "ampere_hour",
    "ampere_hour_throughput",
    "cumulative_ampere_hour_throughput",
    "amp_hour",
    "throughput",
)
TIME_KEYS = ("test_time", "relative_time", "relativeTime", "time", "time_s", "t")
CYCLE_KEYS = (
    "cycle",
    "cycle_id",
    "cycle_index",
    "cycle number",
    "cycle_number",
    "cycleindex",
    "session",
    "session_id",
    "charge_cycle",
)
ENTITY_KEYS = ("cell_id", "cellid", "battery_id", "batteryid", "vehicle_id", "vehicleid", "vehicle_name", "pack_id", "packid", "vin")
SOH_KEYS = ("soh", "state_of_health", "state of health", "health", "capacity_retention")
KAGGLE_DATASET_SLUG = "drtawfikrrahman/deep-learning-ev-battery-pack-diagnostics-sdg-7"
KAGGLE_DATASET_NAME = "KaggleSDG7"


def _normalize_key(value: object) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _as_numeric_array(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).squeeze()
    except (TypeError, ValueError):
        return None
    if array.ndim == 0:
        array = array.reshape(1)
    if array.size == 0 or not np.isfinite(array).any():
        return None
    return array.astype(np.float64)


def _iter_mapping_items(obj: object) -> Iterator[tuple]:
    if isinstance(obj, dict):
        yield from obj.items()


def _get_mapping_value(mapping: Dict, key_options: Sequence[str]) -> Optional[object]:
    normalized_options = {_normalize_key(key) for key in key_options}
    for key, value in mapping.items():
        normalized_key = _normalize_key(key)
        if normalized_key in normalized_options:
            return value
    for key, value in mapping.items():
        normalized_key = _normalize_key(key)
        if any(option in normalized_key or normalized_key in option for option in normalized_options):
            return value
    return None


def _find_value(obj: object, key_options: Sequence[str], max_depth: int = 5) -> Optional[object]:
    if max_depth < 0:
        return None
    if isinstance(obj, dict):
        direct = _get_mapping_value(obj, key_options)
        if direct is not None:
            return direct
        for _, value in obj.items():
            found = _find_value(value, key_options, max_depth - 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_value(value, key_options, max_depth - 1)
            if found is not None:
                return found
    return None


def _resample_like(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == target_len:
        return values
    if values.size < 2:
        return np.full(target_len, float(values[0]) if values.size else 0.0)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst, src, values)


def _integrate_capacity(current: np.ndarray, time: Optional[np.ndarray]) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    if time is not None and len(time) >= len(current):
        time = _resample_like(np.asarray(time, dtype=np.float64), len(current))
        delta_t = np.diff(time, prepend=time[0])
        positive_step = np.median(delta_t[delta_t > 0]) if np.any(delta_t > 0) else 1.0
        delta_t = np.where(delta_t <= 0, positive_step, delta_t)
    else:
        delta_t = np.ones_like(current)
    capacity = np.cumsum(np.abs(current) * delta_t) / 3600.0
    if np.isclose(capacity[-1], capacity[0]):
        return np.linspace(0.0, 1.0, len(current))
    return capacity - capacity[0]


def _cycle_from_record(
    record: Dict,
    dataset_name: str,
    cell_id: str,
    cycle_index: int,
    soh: Optional[float] = None,
) -> Optional[RawBatteryCycle]:
    voltage = _as_numeric_array(_find_value(record, VOLTAGE_KEYS))
    current = _as_numeric_array(_find_value(record, CURRENT_KEYS))
    if voltage is None or current is None or max(voltage.size, current.size) < 8:
        return None

    target_len = max(voltage.size, current.size)
    voltage = _resample_like(voltage, target_len)
    current = _resample_like(current, target_len)

    capacity_trace = _as_numeric_array(_find_value(record, CAPACITY_KEYS))
    if capacity_trace is not None and capacity_trace.size > 1:
        capacity_trace = _resample_like(capacity_trace, target_len)
    else:
        time = _as_numeric_array(_find_value(record, TIME_KEYS))
        capacity_trace = _integrate_capacity(current, time)

    return RawBatteryCycle(
        dataset_name=dataset_name,
        cell_id=cell_id,
        cycle_index=cycle_index,
        voltage=voltage,
        current=current,
        capacity_trace=capacity_trace,
        soh=soh,
    )


def _load_mat_file(path: Path) -> Dict:
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)


def _first_payload(loaded: Dict) -> object:
    for key, value in loaded.items():
        if not key.startswith("__"):
            return value
    return loaded


def _as_list(value: object) -> List:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.reshape(-1))
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def parse_nasa_mat(path: Path) -> List[RawBatteryCycle]:
    loaded = _load_mat_file(path)
    payload = _first_payload(loaded)
    cycles = _as_list(payload.get("cycle") if isinstance(payload, dict) else None)
    if not cycles:
        return parse_generic_mat(path, "NASA")

    charge_records = []
    discharge_capacities = []
    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue
        cycle_type = str(cycle.get("type", "")).lower()
        data = cycle.get("data", cycle)
        if cycle_type == "discharge":
            capacity = _as_numeric_array(_find_value(data, ("capacity",)))
            if capacity is not None:
                discharge_capacities.append(float(np.nanmax(capacity)))
        elif cycle_type == "charge":
            charge_records.append(data)

    if not charge_records:
        return parse_generic_mat(path, "NASA")

    nominal_capacity = discharge_capacities[0] if discharge_capacities else None
    parsed = []
    for cycle_index, record in enumerate(charge_records):
        soh = None
        if nominal_capacity and discharge_capacities:
            capacity = discharge_capacities[min(cycle_index, len(discharge_capacities) - 1)]
            soh = float(np.clip(capacity / nominal_capacity, 0.0, 1.2))
        cycle = _cycle_from_record(record, "NASA", path.stem, cycle_index, soh=soh)
        if cycle is not None:
            parsed.append(cycle)
    return parsed


def _walk_records(obj: object, max_depth: int = 8) -> Iterator[Dict]:
    if max_depth < 0:
        return
    if isinstance(obj, dict):
        if _find_value(obj, VOLTAGE_KEYS, max_depth=1) is not None and _find_value(obj, CURRENT_KEYS, max_depth=1) is not None:
            yield obj
        for _, value in _iter_mapping_items(obj):
            yield from _walk_records(value, max_depth - 1)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _walk_records(value, max_depth - 1)
    elif isinstance(obj, np.ndarray):
        for value in obj.reshape(-1):
            yield from _walk_records(value, max_depth - 1)


def parse_generic_mat(path: Path, dataset_name: str) -> List[RawBatteryCycle]:
    loaded = _load_mat_file(path)
    raw_cycles = []
    for cycle_index, record in enumerate(_walk_records(loaded)):
        cycle = _cycle_from_record(record, dataset_name, path.stem, cycle_index)
        if cycle is not None:
            raw_cycles.append(cycle)
    return _assign_soh_from_capacity(raw_cycles)


def _read_table(path: Path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        if pd is None:
            raise RuntimeError("Reading Excel CALCE files requires pandas plus openpyxl/xlrd.")
        return pd.read_excel(path)
    if pd is not None:
        return pd.read_csv(path)

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def _table_columns(table) -> List[str]:
    if pd is not None and hasattr(table, "columns"):
        return [str(column) for column in table.columns]
    return list(table[0].keys()) if table else []


def _column_values(table, column: str) -> np.ndarray:
    if pd is not None and hasattr(table, "columns"):
        return np.asarray(table[column], dtype=np.float64)
    return np.asarray([float(row[column]) for row in table if row.get(column, "") != ""], dtype=np.float64)


def _find_column(columns: Sequence[str], key_options: Sequence[str]) -> Optional[str]:
    normalized_options = {_normalize_key(key) for key in key_options}
    for column in columns:
        normalized_column = _normalize_key(column)
        if normalized_column in normalized_options:
            return column
    for column in columns:
        normalized_column = _normalize_key(column)
        if any(option in normalized_column or normalized_column in option for option in normalized_options):
            return column
    return None


def parse_tabular_file(path: Path, dataset_name: str) -> List[RawBatteryCycle]:
    table = _read_table(path)
    columns = _table_columns(table)
    voltage_col = _find_column(columns, VOLTAGE_KEYS)
    current_col = _find_column(columns, CURRENT_KEYS)
    capacity_col = _find_column(columns, CAPACITY_KEYS)
    time_col = _find_column(columns, TIME_KEYS)
    cycle_col = _find_column(columns, CYCLE_KEYS)
    entity_col = _find_column(columns, ENTITY_KEYS)
    soh_col = _find_column(columns, SOH_KEYS)

    if voltage_col is None or current_col is None:
        return []

    if pd is not None and hasattr(table, "columns"):
        group_columns = [column for column in (entity_col, cycle_col) if column is not None]
        groups = table.groupby(group_columns, sort=True) if group_columns else [(path.stem, table)]
        cycles = []
        for cycle_index, (group_key, group) in enumerate(groups):
            voltage = np.asarray(group[voltage_col], dtype=np.float64)
            current = np.asarray(group[current_col], dtype=np.float64)
            capacity = np.asarray(group[capacity_col], dtype=np.float64) if capacity_col else None
            if capacity is None or capacity.size < 2:
                time = np.asarray(group[time_col], dtype=np.float64) if time_col else None
                capacity = _integrate_capacity(current, time)
            soh = None
            if soh_col:
                soh_values = np.asarray(group[soh_col], dtype=np.float64)
                soh_values = soh_values[np.isfinite(soh_values)]
                if soh_values.size:
                    soh = float(np.nanmean(soh_values))
                    if soh > 1.5:
                        soh /= 100.0
            cell_id = str(group_key[0] if isinstance(group_key, tuple) else group_key) if entity_col else path.stem
            cycles.append(
                RawBatteryCycle(dataset_name, cell_id, cycle_index, voltage, current, capacity, soh=soh)
            )
        return _assign_soh_from_capacity(cycles)

    voltage = _column_values(table, voltage_col)
    current = _column_values(table, current_col)
    capacity = _column_values(table, capacity_col) if capacity_col else _integrate_capacity(current, None)
    soh = None
    if soh_col:
        soh_values = _column_values(table, soh_col)
        soh_values = soh_values[np.isfinite(soh_values)]
        if soh_values.size:
            soh = float(np.nanmean(soh_values))
            if soh > 1.5:
                soh /= 100.0
    return _assign_soh_from_capacity([RawBatteryCycle(dataset_name, path.stem, 0, voltage, current, capacity, soh=soh)])


def _assign_soh_from_capacity(cycles: List[RawBatteryCycle]) -> List[RawBatteryCycle]:
    capacities = []
    for cycle in cycles:
        if cycle.capacity_trace is not None and cycle.capacity_trace.size:
            capacities.append(float(np.nanmax(cycle.capacity_trace) - np.nanmin(cycle.capacity_trace)))
    nominal = next((capacity for capacity in capacities if capacity > 0), None)
    if nominal is None:
        return cycles
    assigned = []
    for cycle in cycles:
        if cycle.soh is None and cycle.capacity_trace is not None and cycle.capacity_trace.size:
            capacity = float(np.nanmax(cycle.capacity_trace) - np.nanmin(cycle.capacity_trace))
            cycle.soh = float(np.clip(capacity / nominal, 0.0, 1.2))
        assigned.append(cycle)
    return assigned


def _candidate_dataset_dirs(raw_dir: Path, dataset_name: str) -> List[Path]:
    if dataset_name == KAGGLE_DATASET_NAME:
        return [
            raw_dir / KAGGLE_DATASET_NAME,
            raw_dir / "Kaggle",
            raw_dir / "kaggle",
            raw_dir,
        ]
    return [raw_dir / dataset_name]


def _parse_raw_file(path: Path, dataset_name: str) -> List[RawBatteryCycle]:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        if dataset_name == "NASA" or path.stem.upper().startswith("B"):
            cycles = parse_nasa_mat(path)
            for cycle in cycles:
                if dataset_name == KAGGLE_DATASET_NAME:
                    cycle.dataset_name = KAGGLE_DATASET_NAME
            return cycles
        return parse_generic_mat(path, dataset_name)
    if suffix in {".csv", ".txt", ".xlsx", ".xls"}:
        return parse_tabular_file(path, dataset_name)
    return []


def parse_dataset_dir(raw_dir: Path, dataset_name: str) -> List[RawBatteryCycle]:
    dataset_dirs = [path for path in _candidate_dataset_dirs(raw_dir, dataset_name) if path.exists()]
    if not dataset_dirs:
        expected = _candidate_dataset_dirs(raw_dir, dataset_name)[0]
        raise FileNotFoundError(f"Missing dataset directory: {expected}")

    cycles: List[RawBatteryCycle] = []
    seen_files = set()
    for dataset_dir in dataset_dirs:
        for path in sorted(dataset_dir.rglob("*")):
            if not path.is_file() or path.name.startswith(".") or path in seen_files:
                continue
            seen_files.add(path)
            cycles.extend(_parse_raw_file(path, dataset_name))

    valid_cycles = [
        cycle for cycle in _assign_soh_from_capacity(cycles)
        if cycle.soh is not None and cycle.capacity_trace is not None and len(cycle.voltage) >= 8
    ]
    if not valid_cycles:
        raise RuntimeError(
            f"No usable {dataset_name} cycles found under {', '.join(str(path) for path in dataset_dirs)}. "
            "Expected raw voltage/current/capacity cycle files from the paper datasets or Kaggle export."
        )
    return valid_cycles


def convert_cycles(cycles: Sequence[RawBatteryCycle], seq_len: int) -> Dict[str, np.ndarray]:
    features = []
    soh = []
    dataset_names = []
    cell_ids = []
    cycle_indices = []

    for cycle in cycles:
        try:
            feature_tensor = extract_dv_dc_ica_features(
                voltage=cycle.voltage,
                capacity=cycle.capacity_trace,
                current=cycle.current,
                seq_len=seq_len,
            )
        except ValueError:
            continue
        features.append(feature_tensor)
        soh.append(cycle.soh)
        dataset_names.append(cycle.dataset_name)
        cell_ids.append(cycle.cell_id)
        cycle_indices.append(cycle.cycle_index)

    if not features:
        raise RuntimeError("No cycles could be converted into DV/DC/ICA features.")

    return {
        "features": np.asarray(features, dtype=np.float32),
        "soh": np.asarray(soh, dtype=np.float32),
        "dataset_names": np.asarray(dataset_names),
        "cell_ids": np.asarray(cell_ids),
        "cycle_indices": np.asarray(cycle_indices, dtype=np.int32),
    }


def prepare_dataset(raw_dir: Path, output_dir: Path, dataset_name: str, seq_len: int) -> Path:
    cycles = parse_dataset_dir(raw_dir, dataset_name)
    arrays = convert_cycles(cycles, seq_len=seq_len)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_paper_exp.npz"
    np.savez_compressed(output_path, **arrays)
    print(
        f"[{dataset_name}] wrote {output_path} | "
        f"features={arrays['features'].shape} | soh_range=({arrays['soh'].min():.3f}, {arrays['soh'].max():.3f})"
    )
    return output_path


def write_demo_raw_data(raw_dir: Path) -> None:
    """Create tiny CSV-form battery curves so the converter can be smoke-tested."""

    rng = np.random.default_rng(7)
    for dataset_name in ("NASA", "Oxford", "CALCE", KAGGLE_DATASET_NAME):
        dataset_dir = raw_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        path = dataset_dir / f"{dataset_name}_demo_cycles.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cycle_index", "time_s", "voltage_v", "current_a", "charge_capacity_ah"])
            for cycle in range(4):
                soh = 1.0 - 0.03 * cycle
                voltage = np.linspace(3.1, 4.2, 48)
                current = 1.0 - 0.2 / (1.0 + np.exp(-16 * (voltage - 4.0)))
                capacity = soh / (1.0 + np.exp(-10 * (voltage - 3.65 + 0.05 * (1.0 - soh))))
                for sample_idx, (v, i, q) in enumerate(zip(voltage, current, capacity)):
                    writer.writerow([
                        cycle,
                        sample_idx,
                        v + rng.normal(0.0, 0.002),
                        i + rng.normal(0.0, 0.002),
                        q + rng.normal(0.0, 0.001),
                    ])


def _load_kaggle_credentials() -> Optional[tuple]:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with kaggle_json.open("r", encoding="utf-8") as handle:
            credentials = json.load(handle)
        username = credentials.get("username")
        key = credentials.get("key")
        if username and key:
            return username, key
    return None


def download_kaggle_dataset(slug: str, destination_dir: Path) -> Path:
    """Download and extract a public Kaggle dataset using Kaggle API credentials."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    zip_path = destination_dir / "kaggle_dataset.zip"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    request = urllib.request.Request(url)

    credentials = _load_kaggle_credentials()
    if credentials:
        import base64

        token = base64.b64encode(f"{credentials[0]}:{credentials[1]}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")

    try:
        with urllib.request.urlopen(request, timeout=120) as response, zip_path.open("wb") as handle:
            handle.write(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403, 404}:
            raise RuntimeError(
                "Could not download the Kaggle dataset automatically. Configure Kaggle API credentials "
                "(`KAGGLE_USERNAME` and `KAGGLE_KEY`, or ~/.kaggle/kaggle.json), then retry; "
                f"or download {slug} manually and extract it into {destination_dir}."
            ) from exc
        raise

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination_dir)
    print(f"Downloaded and extracted Kaggle dataset {slug} to {destination_dir}")
    return destination_dir




def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare NASA/Oxford/CALCE/Kaggle data for paper_exp.")
    parser.add_argument("--raw-dir", default="data", help="Directory containing NASA, Oxford, CALCE, or KaggleSDG7 folders.")
    parser.add_argument("--output-dir", default="data/processed", help="Where *_paper_exp.npz files are written.")
    parser.add_argument("--datasets", nargs="+", default=["NASA", "Oxford", "CALCE"], choices=["NASA", "Oxford", "CALCE", KAGGLE_DATASET_NAME])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Download the user-provided Kaggle dataset into <raw-dir>/KaggleSDG7 before conversion.",
    )
    parser.add_argument("--kaggle-slug", default=KAGGLE_DATASET_SLUG)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create tiny demo raw CSV files before conversion; intended only for converter smoke tests.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    if args.demo:
        write_demo_raw_data(raw_dir)
    if args.download_kaggle:
        download_kaggle_dataset(args.kaggle_slug, raw_dir / KAGGLE_DATASET_NAME)
    for dataset_name in args.datasets:
        prepare_dataset(raw_dir, output_dir, dataset_name, seq_len=args.seq_len)


if __name__ == "__main__":
    main()

