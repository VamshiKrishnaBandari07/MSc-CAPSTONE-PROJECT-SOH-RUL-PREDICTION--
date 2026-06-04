"""Finish pytest, figures, and docs after training completed but report save failed."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOG = ROOT / "results" / "full_pipeline.log"
PY = sys.executable


def _decode_log() -> str:
    raw = LOG.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _parse_cv_summaries(text: str) -> dict[str, list[float]]:
    pattern = r"\[Paper \| (\w+)\] CV summary \(pooled OOF\): SOH RMSE = ([0-9.]+)"
    out: dict[str, list[float]] = {}
    for dataset, rmse in re.findall(pattern, text):
        out.setdefault(dataset, []).append(float(rmse))
    return out


def _build_minimal_report(summaries: dict[str, list[float]]) -> dict:
    import numpy as np

    from experiments.config import DATASETS, PAPER_REFERENCE
    from experiments.provenance import detect_data_sources, experiment_config_snapshot

    results = []
    for ds in DATASETS:
        rmses = summaries.get(ds, [])
        if not rmses:
            continue
        results.append(
            {
                "dataset": ds,
                "experiment": "paper_reproduction",
                "eval_protocol": "stratified_5fold_cv",
                "independent_runs": len(rmses),
                "metrics": {
                    "rmse": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                    "rmse_runs": rmses,
                },
            }
        )

    nasa_rmse = next((r["metrics"]["rmse"] for r in results if r["dataset"] == "NASA"), None)
    note = (
        f"Recovered from training log after JSON save error. NASA RMSE {nasa_rmse:.4f} vs paper 0.021."
        if nasa_rmse
        else "Recovered from training log."
    )

    return {
        "experiment": "paper_reproduction",
        "device": "cpu",
        "paper_doi": "10.1038/s41598-026-39911-8",
        "paper_target_soh_rmse": 0.021,
        "eval_protocol": "cv5",
        "methodology": {
            "features": ["ICA_dQdV", "DV_dVdQ", "DC_dIdV"],
            "note": "Report rebuilt by scripts/finish_pipeline.py from full_pipeline.log",
        },
        "results": results,
        "data_sources": detect_data_sources(),
        "experiment_config": experiment_config_snapshot(),
        "published_reference": PAPER_REFERENCE,
        "reproducibility_note": note,
    }


def main() -> None:
    if not LOG.is_file():
        raise SystemExit(f"Missing log: {LOG}")

    text = _decode_log()
    summaries = _parse_cv_summaries(text)
    if len(summaries) < 3:
        raise SystemExit(f"Expected 3 datasets in log, got {list(summaries.keys())}")

    report_path = ROOT / "results" / "paper_experiment_report.json"
    import json

    report_path.write_text(
        json.dumps(_build_minimal_report(summaries), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    for ds, rmses in summaries.items():
        print(f"  {ds}: {len(rmses)} runs, mean RMSE = {sum(rmses)/len(rmses):.4f}")

    for script in (
        "scripts/sanitize_paper_report.py",
        "generate_figures.py",
        "scripts/export_summary.py",
        "scripts/sync_results_docs.py",
    ):
        subprocess.run([PY, str(ROOT / script)], cwd=ROOT, check=True)

    subprocess.run([PY, "-m", "pytest", str(ROOT / "tests"), "-q", "--tb=short"], cwd=ROOT, check=True)
    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
