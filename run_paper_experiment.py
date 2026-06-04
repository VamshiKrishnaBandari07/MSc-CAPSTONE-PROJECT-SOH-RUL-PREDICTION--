"""
Paper reproduction — Rahman et al., Scientific Reports (2026)

Reference: https://doi.org/10.1038/s41598-026-39911-8

Evaluation (default): stratified 5-fold CV (paper protocol).
Use --chrono for fast 80/20 supplementary runs on CPU.

Usage:
  python download_data.py --all
  python run_paper_experiment.py --require-real --cpu
  python run_paper_experiment.py --require-real --cpu --dataset NASA
  python run_paper_experiment.py --require-real --cpu --chrono   # fast local split
"""

import json
import os
import time

import numpy as np

from experiments.cli import parse_runtime_args, paper_eval_protocol
from experiments.config import CHECKPOINT_DIR, DATASETS, PAPER_REFERENCE, RESULTS_DIR
from experiments.io_utils import ensure_dirs, save_json
from experiments.paper_config import (
    PAPER_DEFAULT_EVAL,
    PAPER_INDEPENDENT_RUNS,
    PAPER_RUN_SEEDS,
    PAPER_SEQ_LEN,
    PAPER_TARGET_SOH_RMSE,
)
from experiments.provenance import detect_data_sources, experiment_config_snapshot
from experiments.runtime import configure_runtime
from experiments.trainer import set_seed, train_paper_experiment
from model_paper import build_paper_model
from preprocess_paper import PaperDatasetLoader


def _aggregate_independent_runs(run_results: list) -> dict:
    """Mean metrics across independent runs (paper Table 4 protocol)."""
    metrics_list = [r["metrics"] for r in run_results if r.get("metrics")]
    if not metrics_list:
        return None
    rmses = [m["rmse"] for m in metrics_list]
    r2s = [m["r2"] for m in metrics_list]
    maes = [m["mae"] for m in metrics_list]
    best = run_results[int(np.argmin(rmses))]
    return {
        "rmse": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "rmse_runs": rmses,
        "r2": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "mae": float(np.mean(maes)),
        "mse": float(np.mean([m.get("mse", m["rmse"] ** 2) for m in metrics_list])),
        "mono_violation_rate": float(
            np.mean([m.get("mono_violation_rate", 0.0) for m in metrics_list])
        ),
        "rmse_mean_folds": float(np.mean([m.get("rmse_mean_folds", m["rmse"]) for m in metrics_list])),
        "history": best.get("history", []),
        "checkpoint": best.get("checkpoint"),
        "best_epoch": best.get("best_epoch"),
    }


def run_paper_experiment(
    datasets=None,
    force_cpu=False,
    batch_size=None,
    max_epochs=None,
    require_real=False,
    eval_protocol=None,
    n_runs=None,
):
    datasets = datasets or DATASETS
    eval_protocol = eval_protocol or PAPER_DEFAULT_EVAL
    n_runs = n_runs if n_runs is not None else (PAPER_INDEPENDENT_RUNS if eval_protocol == "cv5" else 1)
    run_seeds = PAPER_RUN_SEEDS[: max(1, n_runs)]
    ensure_dirs()
    device = configure_runtime(force_cpu=force_cpu)

    print("\n" + "=" * 96)
    print("PAPER REPRODUCTION — Hybrid CNN-TCN-LSTM-Attention (SOH)")
    print("Scientific Reports (2026) | DOI: 10.1038/s41598-026-39911-8")
    print(f"Device: {device.type.upper()} | Grid: {PAPER_SEQ_LEN} pts | Target SOH RMSE: {PAPER_TARGET_SOH_RMSE}")
    print(f"Evaluation: {eval_protocol} ({'paper Table 4 protocol' if eval_protocol == 'cv5' else 'supplementary 80/20'})")
    print(f"Independent runs: {len(run_seeds)} (seeds: {run_seeds})")
    if eval_protocol == "cv5" and device.type == "cpu":
        hours = 2 * len(run_seeds) * len(datasets)
        print(f"5-fold CV × {len(run_seeds)} runs on CPU: roughly {hours}–{hours * 4} hours total.")
    print("=" * 96)

    results = []
    for dataset in datasets:
        print(f"\n{'#' * 96}\n# PAPER DATASET: {dataset}\n{'#' * 96}")
        features, soh = PaperDatasetLoader.load_dataset(
            dataset, seq_len=PAPER_SEQ_LEN, require_real=require_real
        )
        params_m = sum(p.numel() for p in build_paper_model(seq_len=PAPER_SEQ_LEN).parameters() if p.requires_grad) / 1e6
        print(f"[Paper | {dataset}] Model parameters: {params_m:.4f} M | cycles: {len(soh)}")

        checkpoint = os.path.join(CHECKPOINT_DIR, f"paper_{dataset.lower()}.pt")
        run_results = []
        for run_i, seed in enumerate(run_seeds, start=1):
            print(f"\n[Paper | {dataset}] === Independent run {run_i}/{len(run_seeds)} (seed={seed}) ===")
            set_seed(seed)
            model = build_paper_model(seq_len=PAPER_SEQ_LEN)
            run_ckpt = checkpoint.replace(".pt", f"_run{run_i}.pt")
            run_results.append(
                train_paper_experiment(
                    model,
                    features,
                    soh,
                    dataset,
                    run_ckpt,
                    epochs=max_epochs,
                    batch_size=batch_size,
                    eval_protocol=eval_protocol,
                )
            )

        if len(run_results) == 1:
            result = run_results[0]
        else:
            aggregated = _aggregate_independent_runs(run_results)
            result = {
                **run_results[0],
                "metrics": aggregated,
                "independent_runs": len(run_results),
                "run_seeds": run_seeds,
                "per_run_metrics": [
                    {"seed": s, "rmse": r["metrics"]["rmse"], "r2": r["metrics"]["r2"]}
                    for s, r in zip(run_seeds, run_results)
                    if r.get("metrics")
                ],
            }
        results.append(result)

    # When re-running one dataset, merge into existing report (keep all three).
    report_file = os.path.join(RESULTS_DIR, "paper_experiment_report.json")
    if len(datasets) < len(DATASETS) and os.path.isfile(report_file):
        with open(report_file, encoding="utf-8") as handle:
            prior = json.load(handle)
        merged = {r["dataset"]: r for r in prior.get("results", [])}
        for r in results:
            merged[r["dataset"]] = r
        results = [merged[d] for d in DATASETS if d in merged]

    payload = {
        "experiment": "paper_reproduction",
        "device": device.type,
        "paper_doi": "10.1038/s41598-026-39911-8",
        "paper_target_soh_rmse": PAPER_TARGET_SOH_RMSE,
        "eval_protocol": eval_protocol,
        "methodology": {
            "features": ["ICA_dQdV", "DV_dVdQ", "DC_dIdV"],
            "voltage_grid": f"{PAPER_SEQ_LEN} points, 2.5-4.2 V",
            "denoising": "Savitzky-Golay window=15 order=3",
            "loss": "MSE",
            "outputs": "SOH only",
            "augmentation": "±10 mV voltage jitter + feature noise (train only)",
            "preprocessing": "IQR outlier removal + global per-channel min-max (paper Section 3)",
            "cross_validation": "stratified 5-fold" if eval_protocol == "cv5" else "chronological 80/20",
            "independent_runs": len(run_seeds),
            "run_seeds": run_seeds,
        },
        "results": results,
        "data_sources": detect_data_sources(),
        "experiment_config": experiment_config_snapshot(),
        "published_reference": PAPER_REFERENCE,
        "reproducibility_note": _reproducibility_note(results),
    }
    report_path = save_json(payload, "paper_experiment_report.json")

    import subprocess
    import sys as _sys

    subprocess.run([_sys.executable, "scripts/sanitize_paper_report.py"], check=False)

    print("\n--- Paper Experiment Summary vs published hybrid (RMSE 0.021) ---")
    for result in results:
        m = result["metrics"]
        if not m:
            print(f"{result['dataset']:<8} | No valid metrics")
            continue
        gap = ((m["rmse"] - PAPER_TARGET_SOH_RMSE) / PAPER_TARGET_SOH_RMSE) * 100
        std = m.get("rmse_std")
        rmse_txt = f"{m['rmse']:.4f} ± {std:.4f}" if std is not None else f"{m['rmse']:.4f}"
        print(
            f"{result['dataset']:<8} | SOH RMSE: {rmse_txt} ({gap:+.1f}% vs paper) | "
            f"R2: {m.get('r2', 0):.4f} | protocol: {result.get('eval_protocol')}"
        )
    print(f"\nFull report: {report_path}")
    return payload


def _reproducibility_note(results: list) -> str:
    nasa = next((r for r in results if r["dataset"] == "NASA"), None)
    if not nasa or not nasa.get("metrics"):
        return "Methodology reproduced per Rahman et al. (2026); metrics pending."
    rmse = nasa["metrics"]["rmse"]
    if rmse <= PAPER_TARGET_SOH_RMSE * 1.05:
        return (
            f"NASA pooled CV RMSE {rmse:.4f} meets paper Table 4 target ({PAPER_TARGET_SOH_RMSE}) "
            f"within 5% tolerance."
        )
    return (
        f"Methodology reproduced per Rahman et al. (2026). NASA RMSE {rmse:.4f} vs "
        f"paper Table 4 ({PAPER_TARGET_SOH_RMSE}). Gap may reflect dataset/protocol differences."
    )


if __name__ == "__main__":
    args = parse_runtime_args("Paper SOH reproduction (CPU/GPU)")
    datasets = (args.dataset,) if args.dataset else DATASETS
    protocol = paper_eval_protocol(args) or PAPER_DEFAULT_EVAL
    started = time.time()
    run_paper_experiment(
        datasets=datasets,
        force_cpu=args.cpu,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        require_real=args.require_real,
        eval_protocol=protocol,
        n_runs=args.paper_runs,
    )
    print(f"\nPaper experiment runtime: {(time.time() - started) / 60:.1f} minutes")
