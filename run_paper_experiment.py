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

import os
import time

from experiments.cli import parse_runtime_args, paper_eval_protocol
from experiments.config import CHECKPOINT_DIR, DATASETS, PAPER_REFERENCE
from experiments.io_utils import ensure_dirs, save_json
from experiments.paper_config import PAPER_DEFAULT_EVAL, PAPER_SEQ_LEN, PAPER_TARGET_SOH_RMSE
from experiments.provenance import detect_data_sources, experiment_config_snapshot
from experiments.runtime import configure_runtime
from experiments.trainer import set_seed, train_paper_experiment
from model_paper import build_paper_model
from preprocess_paper import PaperDatasetLoader


def run_paper_experiment(
    datasets=None,
    force_cpu=False,
    batch_size=None,
    max_epochs=None,
    require_real=False,
    eval_protocol=None,
):
    datasets = datasets or DATASETS
    eval_protocol = eval_protocol or PAPER_DEFAULT_EVAL
    set_seed()
    ensure_dirs()
    device = configure_runtime(force_cpu=force_cpu)

    print("\n" + "=" * 96)
    print("PAPER REPRODUCTION — Hybrid CNN-TCN-LSTM-Attention (SOH)")
    print("Scientific Reports (2026) | DOI: 10.1038/s41598-026-39911-8")
    print(f"Device: {device.type.upper()} | Grid: {PAPER_SEQ_LEN} pts | Target SOH RMSE: {PAPER_TARGET_SOH_RMSE}")
    print(f"Evaluation: {eval_protocol} ({'paper Table 4 protocol' if eval_protocol == 'cv5' else 'supplementary 80/20'})")
    if eval_protocol == "cv5" and device.type == "cpu":
        print("5-fold CV on CPU: ~2–8 hours for all datasets. Use --dataset NASA or --chrono for faster runs.")
    elif device.type == "cpu":
        print("CPU mode — expect ~30–90 min per dataset (early stopping may finish sooner).")
    print("=" * 96)

    results = []
    for dataset in datasets:
        print(f"\n{'#' * 96}\n# PAPER DATASET: {dataset}\n{'#' * 96}")
        features, soh = PaperDatasetLoader.load_dataset(
            dataset, seq_len=PAPER_SEQ_LEN, require_real=require_real
        )
        model = build_paper_model(seq_len=PAPER_SEQ_LEN)
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"[Paper | {dataset}] Model parameters: {params_m:.4f} M | cycles: {len(soh)}")

        checkpoint = os.path.join(CHECKPOINT_DIR, f"paper_{dataset.lower()}.pt")
        result = train_paper_experiment(
            model,
            features,
            soh,
            dataset,
            checkpoint,
            epochs=max_epochs,
            batch_size=batch_size,
            eval_protocol=eval_protocol,
        )
        results.append(result)

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
            "cross_validation": "stratified 5-fold" if eval_protocol == "cv5" else "chronological 80/20",
        },
        "results": results,
        "data_sources": detect_data_sources(),
        "experiment_config": experiment_config_snapshot(),
        "published_reference": PAPER_REFERENCE,
    }
    report_path = save_json(payload, "paper_experiment_report.json")

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
    )
    print(f"\nPaper experiment runtime: {(time.time() - started) / 60:.1f} minutes")
