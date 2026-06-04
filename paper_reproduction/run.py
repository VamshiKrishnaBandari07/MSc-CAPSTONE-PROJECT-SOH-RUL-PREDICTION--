"""Paper reproduction entry point (wrapper)."""

from run_paper_experiment import run_paper_experiment
from experiments.cli import parse_runtime_args, paper_eval_protocol
from experiments.config import DATASETS
from experiments.paper_config import PAPER_DEFAULT_EVAL
import time

if __name__ == "__main__":
    args = parse_runtime_args()
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
    print(f"\nRuntime: {(time.time() - started) / 60:.1f} minutes")
