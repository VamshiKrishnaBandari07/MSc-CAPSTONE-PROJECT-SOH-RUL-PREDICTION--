"""
Generate figures for paper reproduction (Scientific Reports 2026).

Usage:
    python generate_figures.py
    python generate_figures.py --report results/paper_experiment_report.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.config import DATASETS, PAPER_REFERENCE, RESULTS_DIR
from experiments.inference import collect_all_predictions
from experiments.io_utils import ensure_dirs, save_json

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

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
    "ours": "#3498db",
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


def _paper_results_list(report):
    if "results" in report:
        return report["results"]
    return report.get("experiment_a_paper_reproduction", report.get("results", []))


def plot_soh_trajectories(all_preds, datasets=None):
    datasets = datasets or DATASETS
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        data = all_preds[dataset]
        cycles = data["cycles"]
        ax.plot(cycles, data["soh_true"], color=COLORS["true"], lw=2, label="True SOH")
        ax.plot(cycles, data["soh_pred"], color=COLORS["ours"], ls="--", lw=1.8, label="Reproduction")
        ax.set_title(dataset)
        ax.set_xlabel("Cycle index")
        ax.set_ylim(0.35, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("State of Health (SOH)")
    axes[-1].legend(loc="lower left")
    fig.suptitle("Validation SOH Trajectories (Paper Reproduction)", fontsize=13, y=1.02)
    return _save(fig, "fig01_soh_trajectories")


def plot_soh_scatter(all_preds, datasets=None):
    datasets = datasets or DATASETS
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        true = np.array(all_preds[dataset]["soh_true"])
        pred = np.array(all_preds[dataset]["soh_pred"])
        ax.scatter(true, pred, alpha=0.75, color=COLORS["ours"], edgecolors="white", s=45)
        lims = [min(true.min(), pred.min()) - 0.02, max(true.max(), pred.max()) + 0.02]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(dataset)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Predicted SOH")
    for ax in axes:
        ax.set_xlabel("True SOH")
    fig.suptitle("Predicted vs True SOH (Validation Set)", fontsize=13, y=1.02)
    return _save(fig, "fig02_soh_scatter")


def plot_rmse_comparison(report, datasets=None):
    paper_res = {r["dataset"]: r["metrics"]["rmse"] for r in _paper_results_list(report)}
    datasets = datasets or tuple(paper_res.keys())
    ref_t = PAPER_REFERENCE["transformer"]["soh_rmse"]
    ref_p = PAPER_REFERENCE["paper_hybrid"]["soh_rmse"]
    x = np.arange(len(datasets))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(datasets)), 5))
    ax.bar(x - width, [ref_t] * len(datasets), width, label="Transformer (published)", color=COLORS["ref_transformer"])
    ax.bar(x, [ref_p] * len(datasets), width, label="Paper hybrid (published)", color=COLORS["ref_paper"])
    ax.bar(x + width, [paper_res[d] for d in datasets], width, label="This reproduction", color=COLORS["ours"])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("SOH RMSE (lower is better)")
    ax.set_title("SOH RMSE vs Published Baselines")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "fig03_soh_rmse_comparison")


def plot_training_curves(report, datasets=None):
    paper_by_ds = {r["dataset"]: r for r in _paper_results_list(report)}
    datasets = datasets or tuple(paper_by_ds.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        hist = paper_by_ds[dataset].get("history") or paper_by_ds[dataset].get("fold_results", [{}])[0].get("history", [])
        if not hist and "fold_results" in paper_by_ds[dataset]:
            hist = paper_by_ds[dataset]["fold_results"][0].get("history", [])
        epochs = [h["epoch"] for h in hist]
        rmse = [h.get("val_soh_rmse", h.get("val_rmse")) for h in hist]
        ax.plot(epochs, rmse, marker="o", color=COLORS["ours"], label="Validation RMSE")
        ax.axhline(PAPER_REFERENCE["paper_hybrid"]["soh_rmse"], color=COLORS["ref_paper"], ls=":", lw=1.2, label="Paper target")
        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation SOH RMSE")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Training Convergence", fontsize=13, y=1.02)
    return _save(fig, "fig04_training_convergence")


def generate_all_figures(report_path=None):
    ensure_dirs()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if report_path is None:
        report_path = os.path.join(RESULTS_DIR, "paper_experiment_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report not found: {report_path}. Run run_paper_experiment.py first.")

    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)

    saved = [plot_rmse_comparison(report), plot_training_curves(report)]

    ckpt_dir = os.path.join("checkpoints")
    has_ckpt = os.path.isdir(ckpt_dir) and any(f.endswith(".pt") for f in os.listdir(ckpt_dir))
    if has_ckpt:
        print("Collecting paper model predictions (requires checkpoints)...")
        all_preds = collect_all_predictions()
        save_json(all_preds, "validation_predictions.json")
        saved = [
            plot_soh_trajectories(all_preds),
            plot_soh_scatter(all_preds),
            *saved,
        ]
    else:
        print(
            "Skipping fig01/fig02 (no checkpoints). Run run_paper_experiment.py first, "
            "or use committed fig01/fig02 if present."
        )

    print("\n" + "=" * 60)
    print("PAPER REPRODUCTION FIGURES")
    print("=" * 60)
    for png, pdf in saved:
        print(f"  PNG: {png}")
        print(f"  PDF: {pdf}")
    print("=" * 60)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper reproduction figures.")
    parser.add_argument("--report", default=None, help="Path to paper_experiment_report.json")
    generate_all_figures(parser.parse_args().report)
