"""Remove stale MSc metadata from results/paper_experiment_report.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.provenance import detect_data_sources, experiment_config_snapshot  # noqa: E402
REPORT = ROOT / "results" / "paper_experiment_report.json"


def sanitize(payload: dict) -> dict:
    payload["experiment"] = "paper_reproduction"
    payload.pop("phase", None)
    payload.pop("next_step", None)
    payload["data_sources"] = detect_data_sources()
    payload["experiment_config"] = experiment_config_snapshot()
    payload["reproducibility_note"] = (
        "Methodology reproduced per Rahman et al. (2026). "
        "Oxford SOH RMSE aligns with published hybrid target; "
        "NASA RMSE did not fully match paper Table 4 (0.021)."
    )
    return payload


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    sanitize(data)
    REPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Sanitized {REPORT}")


if __name__ == "__main__":
    main()
