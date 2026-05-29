import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(command: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(command))
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paper experiment first, then modified repo experiment, then compare both."
    )
    parser.add_argument("--output-dir", default="paper_exp/outputs/full_comparison")
    parser.add_argument("--raw-dir", default="data")
    parser.add_argument("--paper-datasets", nargs="+", default=["KaggleSDG7"])
    parser.add_argument("--modified-datasets", nargs="+", default=["NASA", "Oxford", "CALCE"])
    parser.add_argument("--prepare-kaggle", action="store_true", help="Download/prepare KaggleSDG7 before training.")
    parser.add_argument("--paper-epochs", type=int, default=300)
    parser.add_argument("--paper-folds", type=int, default=5)
    parser.add_argument("--paper-seq-len", type=int, default=128)
    parser.add_argument("--paper-batch-size", type=int, default=64)
    parser.add_argument("--modified-epochs", type=int, default=5)
    parser.add_argument("--modified-batch-size", type=int, default=8)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="Run a quick verification suite.")
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.paper_epochs = 1
    args.paper_folds = 2
    args.paper_seq_len = 64
    args.paper_batch_size = 8
    args.modified_epochs = 1
    args.modified_datasets = args.modified_datasets[:1]
    args.benchmark_repeats = min(args.benchmark_repeats, 10)
    return args


def main() -> None:
    args = apply_smoke_overrides(build_arg_parser().parse_args())
    output_dir = Path(args.output_dir)
    paper_output = output_dir / "01_paper_experiment"
    modified_output = output_dir / "02_modified_experiment"
    comparison_output = output_dir / "03_comparison"
    logs_dir = output_dir / "logs"

    if args.prepare_kaggle and "KaggleSDG7" in args.paper_datasets:
        _run(
            [
                sys.executable,
                "-m",
                "paper_exp.prepare_data",
                "--download-kaggle",
                "--datasets",
                "KaggleSDG7",
                "--raw-dir",
                args.raw_dir,
                "--output-dir",
                str(Path(args.raw_dir) / "processed"),
                "--seq-len",
                str(args.paper_seq_len),
            ],
            logs_dir / "00_prepare_kaggle.log",
        )

    _run(
        [
            sys.executable,
            "-m",
            "paper_exp.train",
            "--datasets",
            *args.paper_datasets,
            "--raw-dir",
            args.raw_dir,
            "--output-dir",
            str(paper_output),
            "--require-real-data",
            "--seq-len",
            str(args.paper_seq_len),
            "--n-folds",
            str(args.paper_folds),
            "--epochs",
            str(args.paper_epochs),
            "--batch-size",
            str(args.paper_batch_size),
            "--benchmark-repeats",
            str(args.benchmark_repeats),
        ],
        logs_dir / "01_paper_experiment.log",
    )

    _run(
        [
            sys.executable,
            "-m",
            "paper_exp.modified_experiment",
            "--datasets",
            *args.modified_datasets,
            "--epochs",
            str(args.modified_epochs),
            "--batch-size",
            str(args.modified_batch_size),
            "--benchmark-runs",
            str(args.benchmark_repeats),
            "--output-dir",
            str(modified_output),
        ],
        logs_dir / "02_modified_experiment.log",
    )

    _run(
        [
            sys.executable,
            "-m",
            "paper_exp.compare_results",
            "--paper-metrics",
            str(paper_output / "metrics.json"),
            "--modified-metrics",
            str(modified_output / "metrics.json"),
            "--output-dir",
            str(comparison_output),
        ],
        logs_dir / "03_compare_results.log",
    )

    print("\n================ comparison workflow complete ================")
    print(f"Paper metrics:     {paper_output / 'metrics.json'}")
    print(f"Modified metrics:  {modified_output / 'metrics.json'}")
    print(f"Comparison report: {comparison_output / 'comparison.md'}")


if __name__ == "__main__":
    main()

