"""Shared CLI helpers for experiment entry scripts."""

import argparse


def add_runtime_args(parser):
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (no GPU). Also auto-selected when CUDA is unavailable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: 4 on CPU, 16 paper / 8 MSc on GPU)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max training epochs (paper default 200, MSc default 25)",
    )
    parser.add_argument(
        "--dataset",
        choices=["NASA", "Oxford", "CALCE"],
        default=None,
        help="Run on a single dataset only (default: all three)",
    )
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Fail if real dataset files are missing (no synthetic fallback). Required for paper claims.",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Experiment A: stratified 5-fold CV (paper protocol, default for run_paper_experiment.py).",
    )
    parser.add_argument(
        "--chrono",
        action="store_true",
        help="Experiment A: fast 80/20 chronological split (supplementary, not paper Table 4 protocol).",
    )
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="Run Experiment A only (paper reproduction).",
    )
    parser.add_argument(
        "--msc-only",
        action="store_true",
        help="Run Experiments B+C only (MSc extension; run Experiment A first for baselines).",
    )
    return parser


def parse_runtime_args(description="Battery SOH experiment runner"):
    parser = argparse.ArgumentParser(description=description)
    add_runtime_args(parser)
    return parser.parse_args()


def paper_eval_protocol(args):
    """Resolve Experiment A evaluation protocol from CLI flags."""
    if args.chrono:
        return "chronological"
    if args.cv:
        return "cv5"
    return None  # caller applies script default
