"""CLI helpers for paper reproduction."""

import argparse


def add_runtime_args(parser):
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs (default 200).")
    parser.add_argument(
        "--dataset",
        choices=["NASA", "Oxford", "CALCE"],
        default=None,
        help="Single dataset (default: all three).",
    )
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Fail if real dataset files are missing.",
    )
    parser.add_argument("--cv", action="store_true", help="Stratified 5-fold CV (paper protocol, default).")
    parser.add_argument(
        "--chrono",
        action="store_true",
        help="Chronological 80/20 split (supplementary debug only).",
    )
    parser.add_argument(
        "--paper-runs",
        type=int,
        default=None,
        metavar="N",
        help="Independent training runs averaged for Table 4 (default: 5 for cv5, 1 for chrono).",
    )
    return parser


def parse_runtime_args(description="Paper SOH reproduction (Scientific Reports 2026)"):
    parser = argparse.ArgumentParser(description=description)
    add_runtime_args(parser)
    return parser.parse_args()


def paper_eval_protocol(args):
    if args.chrono:
        return "chronological"
    if args.cv:
        return "cv5"
    return None
