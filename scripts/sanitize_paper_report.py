"""Strip non-JSON-safe fields and refresh summary after paper_experiment_report.json is written."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.provenance import detect_data_sources, experiment_config_snapshot  # noqa: E402

REPORT = ROOT / "results" / "paper_experiment_report.json"


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _strip_fold_arrays(payload: dict) -> dict:
    for r in payload.get("results", []):
        for fr in r.get("fold_results", []) or []:
            fr.pop("val_y_true", None)
            fr.pop("val_y_pred", None)
            m = fr.get("metrics")
            if isinstance(m, dict):
                fr["metrics"] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in m.items()}
    return payload


def sanitize(payload: dict) -> dict:
    payload = _to_jsonable(payload)
    payload = _strip_fold_arrays(payload)
    payload["experiment"] = "paper_reproduction"
    payload.pop("phase", None)
    payload.pop("next_step", None)
    payload["data_sources"] = detect_data_sources()
    payload["experiment_config"] = experiment_config_snapshot()
    return payload


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    data = sanitize(data)
    REPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Sanitized {REPORT}")
    import subprocess

    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_summary.py")], check=False)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_results_docs.py")], check=False)


if __name__ == "__main__":
    main()
