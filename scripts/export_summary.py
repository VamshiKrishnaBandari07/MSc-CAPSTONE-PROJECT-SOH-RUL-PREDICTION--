"""Write slim results/summary.json from paper_experiment_report.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "paper_experiment_report.json"
SUMMARY = ROOT / "results" / "summary.json"


def export_summary(report: dict) -> dict:
    rows = []
    for r in report.get("results", []):
        m = r.get("metrics") or {}
        rows.append(
            {
                "dataset": r["dataset"],
                "eval_protocol": r.get("eval_protocol"),
                "rmse": m.get("rmse"),
                "rmse_std": m.get("rmse_std"),
                "rmse_folds": m.get("rmse_folds"),
                "mae": m.get("mae"),
                "r2": m.get("r2"),
                "mono_violation_rate": m.get("mono_violation_rate"),
            }
        )
    return {
        "experiment": report.get("experiment"),
        "paper_doi": report.get("paper_doi"),
        "eval_protocol": report.get("eval_protocol"),
        "paper_target_soh_rmse_nasa": report.get("paper_target_soh_rmse"),
        "results": rows,
        "data_sources": report.get("data_sources"),
        "reproducibility_note": report.get("reproducibility_note"),
    }


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    SUMMARY.write_text(json.dumps(export_summary(report), indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY}")


if __name__ == "__main__":
    main()
