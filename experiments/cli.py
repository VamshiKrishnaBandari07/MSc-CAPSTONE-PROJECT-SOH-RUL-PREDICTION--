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
    return parser


def parse_runtime_args(description="Battery SOH experiment runner"):
    parser = argparse.ArgumentParser(description=description)
    add_runtime_args(parser)
    return parser.parse_args()
