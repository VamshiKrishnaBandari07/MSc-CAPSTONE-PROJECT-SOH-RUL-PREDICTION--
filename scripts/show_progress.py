"""Parse full_pipeline.log and print training progress percentage."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "full_pipeline.log"
ALT = ROOT / "results" / "experiment_run.log"

DATASETS = ("NASA", "Oxford", "CALCE")
RUNS_PER_DATASET = 5
FOLDS_PER_RUN = 5
AVG_EPOCHS_PER_FOLD = 35  # early stopping ~25–40
MAX_EPOCHS = 200
POST_TRAIN_SHARE = 0.03  # pytest + figures


def _decode_file(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-16-le", errors="replace")


def _read_log() -> tuple[str, Path | None]:
    """Prefer the most recently updated log (active run)."""
    candidates = [p for p in (LOG, ALT) if p.is_file() and p.stat().st_size >= 50]
    if not candidates:
        return "", None
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    return _decode_file(path), path


def parse_progress(text: str) -> dict:
    completed_runs = len(re.findall(r"CV summary \(pooled OOF\)", text))

    dataset = "NASA"
    for name in DATASETS:
        if f"# PAPER DATASET: {name}" in text:
            dataset = name
    ds_idx = DATASETS.index(dataset)

    run_m = list(re.finditer(r"Independent run (\d+)/5", text))
    run_i = int(run_m[-1].group(1)) if run_m else 1

    fold_m = list(re.finditer(r"=== Stratified CV fold (\d+)/5 ===", text))
    fold_i = int(fold_m[-1].group(1)) if fold_m else 1

    epoch_m = list(re.finditer(r"Epoch (\d+)/200", text))
    epoch_i = int(epoch_m[-1].group(1)) if epoch_m else 0

    if "Done in" in text or "PAPER REPRODUCTION FIGURES" in text:
        phase, train_pct, overall = "complete", 100.0, 100.0
    elif "pytest" in text.lower() and "passed" in text.lower():
        phase, train_pct, overall = "testing", 100.0, 98.0
    elif re.search(r">>>.*generate_figures", text):
        phase, train_pct, overall = "figures", 100.0, 99.0
    else:
        phase = "training"
        runs_before = ds_idx * RUNS_PER_DATASET + (run_i - 1)
        fold_frac = (fold_i - 1) + min(epoch_i / AVG_EPOCHS_PER_FOLD, 1.0)
        total_folds = len(DATASETS) * RUNS_PER_DATASET * FOLDS_PER_RUN
        completed_folds = runs_before * FOLDS_PER_RUN + fold_frac
        train_pct = 100.0 * completed_folds / total_folds
        overall = train_pct * (1.0 - POST_TRAIN_SHARE)

    total_runs = len(DATASETS) * RUNS_PER_DATASET
    return {
        "phase": phase,
        "dataset": dataset,
        "run": run_i,
        "fold": fold_i,
        "epoch": epoch_i,
        "completed_independent_runs": completed_runs,
        "total_independent_runs": total_runs,
        "training_pct": round(train_pct, 1),
        "overall_pct": round(overall, 1),
    }


def main() -> None:
    text, log_path = _read_log()
    if not text.strip():
        print("No log found. Start: python scripts/run_train_and_eval.py")
        return
    p = parse_progress(text)
    bar_len = 30
    filled = int(bar_len * p["overall_pct"] / 100)
    bar = "#" * filled + "-" * (bar_len - filled)
    print("=" * 56)
    print("PAPER PIPELINE PROGRESS")
    print("=" * 56)
    print(f"  Overall:  [{bar}] {p['overall_pct']}%")
    print(f"  Training: {p['training_pct']}%  (phase: {p['phase']})")
    print(f"  Dataset:  {p['dataset']}  |  Run {p['run']}/5  |  Fold {p['fold']}/5  |  Epoch {p['epoch']}/200")
    print(f"  Finished independent runs: {p['completed_independent_runs']}/{p['total_independent_runs']}")
    print("=" * 56)
    print(f"  Log: {log_path or LOG}")


if __name__ == "__main__":
    main()
