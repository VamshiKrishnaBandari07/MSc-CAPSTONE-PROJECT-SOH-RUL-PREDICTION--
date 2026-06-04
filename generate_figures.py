"""
Generate thesis-ready figures from experiment checkpoints and results.

Usage:
    python generate_figures.py
    python generate_figures.py --report results/experiment_comparison_report.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.config import CHECKPOINT_DIR, DATASETS, PAPER_REFERENCE, RESULTS_DIR
from experiments.inference import collect_all_predictions, collect_dataset_predictions
from experiments.io_utils import ensure_dirs, save_json

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Consistent thesis styling
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.constrained_layout.use": True,
    }
)

COLORS = {
    "true": "#2c3e50",
    "paper": "#3498db",
    "msc": "#e74c3c",
    "ablation": "#95a5a6",
    "ref_transformer": "#bdc3c7",
    "ref_paper": "#27ae60",
}


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png = os.path.join(FIGURES_DIR, f"{name}.png")
    pdf = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_soh_trajectories(all_preds, datasets=None):
    datasets = datasets or DATASETS
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data = all_preds[dataset]
        paper_cycles = data["paper"]["cycles"]
        msc_cycles = data["msc"]["cycles"]
        ax.plot(paper_cycles, data["paper"]["soh_true"], color=COLORS["true"], lw=2, label="True SOH")
        ax.plot(paper_cycles, data["paper"]["soh_pred"], color=COLORS["paper"], ls="--", lw=1.8, label="Paper repro.")
        ax.plot(msc_cycles, data["msc"]["soh_pred"], color=COLORS["msc"], ls="-.", lw=1.8, label="MSc PI-MT")
        ax.set_title(f"{dataset} Dataset")
        ax.set_xlabel("Cycle index")
        ax.set_ylim(0.35, 1.02)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("State of Health (SOH)")
    axes[-1].legend(loc="lower left", frameon=True)
    fig.suptitle("Validation SOH Trajectories: Paper Reproduction vs MSc Extension", fontsize=13, y=1.02)
    return _save(fig, "fig01_soh_trajectories")


def plot_rul_trajectories(all_preds, datasets=None):
    datasets = datasets or DATASETS
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        data = all_preds[dataset]
        cycles = data["msc"]["cycles"]
        ax.plot(cycles, data["msc"]["rul_true"], color=COLORS["true"], lw=2, label="True RUL")
        ax.plot(cycles, data["msc"]["rul_pred"], color=COLORS["msc"], ls="--", lw=1.8, label="MSc PI-MT RUL")
        ax.set_title(f"{dataset} Dataset")
        ax.set_xlabel("Cycle index")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Remaining Useful Life (cycles)")
    axes[-1].legend(loc="upper right", frameon=True)
    fig.suptitle("Validation RUL Trajectories (MSc Extension)", fontsize=13, y=1.02)
    return _save(fig, "fig02_rul_trajectories")


def plot_soh_scatter(all_preds, datasets=None):
    datasets = datasets or DATASETS
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        true = np.array(all_preds[dataset]["msc"]["soh_true"])
        pred = np.array(all_preds[dataset]["msc"]["soh_pred"])
        ax.scatter(true, pred, alpha=0.75, color=COLORS["msc"], edgecolors="white", linewidths=0.5, s=45)
        lims = [min(true.min(), pred.min()) - 0.02, max(true.max(), pred.max()) + 0.02]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{dataset}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Predicted SOH")
    for ax in axes:
        ax.set_xlabel("True SOH")
    fig.suptitle("MSc Model: Predicted vs True SOH (Validation Set)", fontsize=13, y=1.02)
    return _save(fig, "fig03_soh_scatter")


def plot_rmse_comparison(report, datasets=None):
    paper_res = {r["dataset"]: r["metrics"]["rmse"] for r in report["experiment_a_paper_reproduction"]}
    msc_res = {r["dataset"]: r["metrics"]["soh"]["rmse"] for r in report["experiment_b_msc_extension"]}
    datasets = datasets or tuple(paper_res.keys())
    ref_t = PAPER_REFERENCE["transformer"]["soh_rmse"]
    ref_p = PAPER_REFERENCE["paper_hybrid"]["soh_rmse"]

    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(datasets)), 5))
    ax.bar(x - 1.5 * width, [ref_t] * len(datasets), width, label="Transformer (published)", color=COLORS["ref_transformer"])
    ax.bar(x - 0.5 * width, [ref_p] * len(datasets), width, label="Paper hybrid (published)", color=COLORS["ref_paper"])
    ax.bar(x + 0.5 * width, [paper_res[d] for d in datasets], width, label="Paper repro. (ours)", color=COLORS["paper"])
    ax.bar(x + 1.5 * width, [msc_res[d] for d in datasets], width, label="MSc PI-MT (ours)", color=COLORS["msc"])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("SOH RMSE (lower is better)")
    ax.set_title("SOH RMSE Comparison Across Datasets")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "fig04_soh_rmse_comparison")


def plot_training_curves(report, datasets=None):
    paper_by_ds = {r["dataset"]: r for r in report["experiment_a_paper_reproduction"]}
    msc_by_ds = {r["dataset"]: r for r in report["experiment_b_msc_extension"]}
    datasets = datasets or tuple(paper_by_ds.keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        paper_hist = paper_by_ds[dataset]["history"]
        msc_hist = msc_by_ds[dataset]["history"]
        ax.plot([h["epoch"] for h in paper_hist], [h["val_soh_rmse"] for h in paper_hist],
                marker="o", color=COLORS["paper"], label="Paper repro.")
        ax.plot([h["epoch"] for h in msc_hist], [h["val_soh_rmse"] for h in msc_hist],
                marker="s", color=COLORS["msc"], label="MSc PI-MT")
        ax.axhline(PAPER_REFERENCE["paper_hybrid"]["soh_rmse"], color=COLORS["ref_paper"], ls=":", lw=1.2,
                   label="Published paper RMSE")
        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Validation SOH RMSE")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Training Convergence on Validation Set", fontsize=13, y=1.02)
    return _save(fig, "fig05_training_convergence")


def plot_computational_profile(report):
    comp = report["computational_profile"]
    labels = ["Parameters (M)", "Latency (ms)", "Energy (mJ)"]
    transformer = [
        PAPER_REFERENCE["transformer"]["params_m"],
        PAPER_REFERENCE["transformer"]["latency_ms"],
        PAPER_REFERENCE["transformer"]["energy_mj"],
    ]
    paper = [comp["paper"]["params_m"], comp["paper"]["latency_ms"], comp["paper"]["energy_mj"]]
    msc = [comp["msc"]["params_m"], comp["msc"]["latency_ms"], comp["msc"]["energy_mj"]]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, transformer, width, label="Transformer (published)", color=COLORS["ref_transformer"])
    ax.bar(x, paper, width, label="Paper repro. (ours)", color=COLORS["paper"])
    ax.bar(x + width, msc, width, label="MSc PI-MT (ours)", color=COLORS["msc"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Computational Profile Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "fig06_computational_profile")


def plot_ablation(report):
    ablation = report.get("experiment_c_ablation_no_physics", [])
    msc = report.get("experiment_b_msc_extension", [])
    if not ablation:
        return None

    msc_by_ds = {r["dataset"]: r for r in msc}
    datasets = [r["dataset"] for r in ablation]
    mono_no = [r["metrics"]["soh"]["mono_violation_rate"] * 100 for r in ablation]
    mono_yes = [msc_by_ds[d]["metrics"]["soh"]["mono_violation_rate"] * 100 for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, mono_no, width, label="Without physics loss", color=COLORS["ablation"])
    ax.bar(x + width / 2, mono_yes, width, label="With physics loss", color=COLORS["msc"])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Monotonicity violation rate (%)")
    ax.set_title("Ablation: Impact of Physics-Informed Monotonicity Loss")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "fig07_ablation_monotonicity")


def generate_nasa_real_figures(report_path=None):
    """Generate figures specifically for real NASA B0005-B0018 experiments."""
    if report_path is None:
        report_path = os.path.join(RESULTS_DIR, "nasa_real_experiment_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError("Run `python run_nasa_real.py` first.")

    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)

    preds = collect_dataset_predictions(
        "NASA",
        paper_ckpt=os.path.join(CHECKPOINT_DIR, "paper_nasa_real.pt"),
        msc_ckpt=os.path.join(CHECKPOINT_DIR, "msc_nasa_real.pt"),
        ablation_ckpt=os.path.join(CHECKPOINT_DIR, "msc_ablation_nasa_real.pt"),
    )
    all_preds = {"NASA": preds}

    saved = []
    nasa_only = ("NASA",)
    saved.append(plot_soh_trajectories(all_preds, nasa_only))
    saved.append(plot_rul_trajectories(all_preds, nasa_only))
    saved.append(plot_soh_scatter(all_preds, nasa_only))
    saved.append(plot_rmse_comparison(report, datasets=nasa_only))
    saved.append(plot_training_curves(report, datasets=nasa_only))

    # Rename outputs with nasa_real prefix
    renamed = []
    for png, pdf in saved:
        base = os.path.basename(png).replace("fig0", "fig_nasa_real_0")
        new_png = os.path.join(FIGURES_DIR, base)
        new_pdf = os.path.join(FIGURES_DIR, base.replace(".png", ".pdf"))
        os.replace(png, new_png)
        os.replace(pdf, new_pdf)
        renamed.append((new_png, new_pdf))

    print("\nNASA REAL-DATA FIGURES:")
    for png, pdf in renamed:
        print(f"  PNG: {png}")
        print(f"  PDF: {pdf}")
    return renamed


def generate_all_figures(report_path=None, include_nasa_real=False):
    ensure_dirs()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if report_path is None:
        report_path = os.path.join(RESULTS_DIR, "experiment_comparison_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"Report not found at {report_path}. Run `python run_experiments.py` first."
        )

    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)

    print("Collecting predictions from saved checkpoints...")
    all_preds = collect_all_predictions()
    save_json(all_preds, "validation_predictions.json")

    saved = []
    saved.append(plot_soh_trajectories(all_preds))
    saved.append(plot_rul_trajectories(all_preds))
    saved.append(plot_soh_scatter(all_preds))
    saved.append(plot_rmse_comparison(report))
    saved.append(plot_training_curves(report))
    saved.append(plot_computational_profile(report))
    ablation = plot_ablation(report)
    if ablation:
        saved.append(ablation)

    print("\n" + "=" * 60)
    print("THESIS FIGURES GENERATED")
    print("=" * 60)
    for png, pdf in saved:
        print(f"  PNG: {png}")
        print(f"  PDF: {pdf}")
    print(f"\nPredictions JSON: {os.path.join(RESULTS_DIR, 'validation_predictions.json')}")
    print("=" * 60)

    if include_nasa_real and os.path.exists(os.path.join(CHECKPOINT_DIR, "paper_nasa_real.pt")):
        print("\nGenerating NASA real-data figures...")
        generate_nasa_real_figures()

    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate thesis figures for MSc capstone.")
    parser.add_argument("--report", default=None, help="Path to experiment_comparison_report.json")
    parser.add_argument("--nasa-real", action="store_true", help="Also generate NASA real-data figures")
    parser.add_argument("--nasa-real-only", action="store_true", help="Only NASA real-data figures")
    args = parser.parse_args()

    if args.nasa_real_only:
        generate_nasa_real_figures()
    else:
        generate_all_figures(args.report, include_nasa_real=args.nasa_real)
