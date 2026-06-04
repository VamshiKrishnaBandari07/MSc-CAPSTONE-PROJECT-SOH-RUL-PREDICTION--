#!/usr/bin/env python3
"""Environment and dataset health check."""

import importlib
import os
import sys


def _status(ok, label, detail=""):
    mark = "OK" if ok else "MISSING"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def check_python():
    version = sys.version_info
    ok = version >= (3, 9)
    _status(ok, f"Python {version.major}.{version.minor}.{version.micro}", "3.9+ required")
    return ok


def check_imports():
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl"),
        ("pytest", "pytest"),
    ]
    all_ok = True
    for label, module in packages:
        try:
            importlib.import_module(module)
            _status(True, f"package: {label}")
        except ImportError:
            _status(False, f"package: {label}", "pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_datasets():
    root = os.getcwd()
    checks = [
        ("NASA", os.path.join(root, "data", "NASA"), ".mat", 4, "python download_data.py --nasa"),
        (
            "Oxford",
            os.path.join(root, "data", "Oxford", "Oxford_Battery_Degradation_Dataset_1.mat"),
            None,
            1,
            "python download_data.py --oxford",
        ),
        ("CALCE", os.path.join(root, "data", "CALCE"), "xlsx", 1, "python download_data.py --calce"),
    ]

    for name, path, ext, min_count, hint in checks:
        if name == "Oxford":
            ok = os.path.isfile(path) and os.path.getsize(path) > 1_000_000
            _status(ok, f"dataset: {name}", hint if not ok else f"file present ({os.path.getsize(path) // 1_000_000} MB)")
            continue

        if not os.path.isdir(path):
            _status(False, f"dataset: {name}", hint)
            continue

        if ext == "xlsx":
            count = sum(
                1
                for dp, _, files in os.walk(path)
                for f in files
                if f.lower().endswith((".xls", ".xlsx"))
            )
        else:
            suffix = ext if ext.startswith(".") else f".{ext}"
            count = sum(1 for f in os.listdir(path) if f.lower().endswith(suffix))

        ok = count >= min_count
        _status(ok, f"dataset: {name}", f"{count} file(s)" if ok else hint)


def check_results():
    root = os.getcwd()
    files = [
        "results/paper_experiment_report.json",
        "results/experiment_comparison_report.json",
        "results/computational_benchmark.json",
        "results/figures/fig04_soh_rmse_comparison.pdf",
    ]
    for rel in files:
        path = os.path.join(root, rel)
        _status(os.path.isfile(path), f"artifact: {rel}", "run run_experiments.py" if not os.path.isfile(path) else "")


def check_provenance():
    import json

    report_path = os.path.join(os.getcwd(), "results", "experiment_comparison_report.json")
    if not os.path.isfile(report_path):
        return
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    sources = report.get("data_sources", {})
    print("\n  Data sources recorded in experiment report:")
    for dataset, info in sources.items():
        paper = info.get("experiment_a_paper", "unknown")
        print(f"    {dataset}: {paper}")


def main():
    print("=" * 60)
    print("MSc Capstone — Setup Verification")
    print("=" * 60)
    print("\nEnvironment:")
    check_python()
    check_imports()
    print("\nDatasets (raw files are NOT in git — download required):")
    check_datasets()
    print("\nCommitted results:")
    check_results()
    check_provenance()
    print("\nNext steps (run in order):")
    print("  python download_data.py --all")
    print("  python run_paper_experiment.py --require-real --cpu   # Phase 1: Paper")
    print("  python run_experiments.py --msc-only --require-real --cpu   # Phase 2: MSc")
    print("  python generate_figures.py")
    print("  python scripts/sync_results_docs.py")
    print("  python -m pytest tests/ -v")
    print("\nOr one command: powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline.ps1")
    print("\nDocs: docs/CAPSTONE_OVERVIEW.md | docs/RESULTS.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
