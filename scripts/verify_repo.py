#!/usr/bin/env python3
"""Point-by-point verification for paper-only repository (CI + supervisor)."""

from __future__ import annotations

import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("=" * 60)
    print("Paper repository verification")
    print("=" * 60)

    # 1. Core entry points
    for path in (
        "run_paper_experiment.py",
        "model_paper.py",
        "preprocess_paper.py",
        "generate_figures.py",
        "experiments/trainer.py",
        "experiments/paper_preprocessing.py",
    ):
        check(f"file: {path}", os.path.isfile(path))

    # 2. No MSc scripts on GitHub
    for forbidden in (
        "run_experiments.py",
        "model.py",
        "preprocess.py",
        "train.py",
    ):
        check(f"absent: {forbidden}", not os.path.isfile(forbidden))

    # 3. Imports
    for mod in ("numpy", "scipy", "torch", "matplotlib", "pandas", "openpyxl"):
        try:
            importlib.import_module(mod)
            check(f"import: {mod}", True)
        except ImportError:
            check(f"import: {mod}", False, "pip install -r requirements.txt")

    # 4. Paper experiment report
    report_path = "results/paper_experiment_report.json"
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        check("report.experiment", report.get("experiment") == "paper_reproduction")
        check("report.no_phase2", "next_step" not in report and "experiment_b_msc" not in str(report.get("experiment_config", {})))
        datasets = [r["dataset"] for r in report.get("results", [])]
        check("report.three_datasets", datasets == ["NASA", "Oxford", "CALCE"], str(datasets))
        check("report.eval_protocol", report.get("eval_protocol") == "cv5")
        check(
            "report.paper_doi",
            report.get("paper_doi") == "10.1038/s41598-026-39911-8",
            report.get("paper_doi", ""),
        )
        for r in report.get("results", []):
            m = r.get("metrics") or {}
            check(f"metrics.{r['dataset']}", m.get("rmse") is not None, f"rmse={m.get('rmse')}")
    else:
        check("report exists", False, report_path)

    # 5. Figures (paper naming only)
    figs = [
        "fig01_soh_trajectories.pdf",
        "fig02_soh_scatter.pdf",
        "fig03_soh_rmse_comparison.pdf",
        "fig04_training_convergence.pdf",
    ]
    for fig in figs:
        check(f"figure: {fig}", os.path.isfile(os.path.join("results/figures", fig)))

    # 6. Benchmark
    bench = "results/computational_benchmark.json"
    if os.path.isfile(bench):
        with open(bench, encoding="utf-8") as f:
            b = json.load(f)
        check("benchmark.paper_only", "msc_proposed" not in b and "paper_reproduction" in b)
    else:
        check("benchmark exists", False)

    # 7. Documentation (minimal set)
    for doc in ("README.md", "docs/SUPERVISOR_GUIDE.md", "docs/PAPER_METHODOLOGY.md", "docs/RESULTS.md"):
        check(f"doc: {doc}", os.path.isfile(doc))

    print("=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
