import argparse
import json
from pathlib import Path
from typing import Dict, Optional

try:
    from .config import PAPER_TARGETS
except ImportError:
    from config import PAPER_TARGETS


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _delta(value: Optional[float], target: Optional[float]) -> Optional[float]:
    if value is None or target in (None, 0):
        return None
    return (value - target) / target * 100.0


def _paper_actuals(metrics: Dict) -> Dict[str, float]:
    return {
        "soh_rmse": metrics["summary"]["rmse"]["mean"],
        "soh_r2": metrics["summary"]["r2"]["mean"],
        "parameters_millions": metrics["parameter_count_millions"],
        "latency_ms": metrics["efficiency"]["latency_ms_per_sample"],
        "energy_mj": metrics["efficiency"]["estimated_energy_mj_per_sample"],
    }


def _modified_actuals(metrics: Dict) -> Dict[str, float]:
    return {
        "soh_rmse": metrics["summary"]["soh_rmse"]["mean"],
        "rul_rmse_cycles": metrics["summary"]["rul_rmse_cycles"]["mean"],
        "parameters_millions": metrics["parameter_count_millions"],
        "latency_ms": metrics["efficiency"]["latency_ms_per_sample"],
        "energy_mj": metrics["efficiency"]["estimated_energy_mj_per_sample"],
    }


def build_comparison(paper_metrics: Dict, modified_metrics: Dict) -> Dict[str, object]:
    paper_actuals = _paper_actuals(paper_metrics)
    modified_actuals = _modified_actuals(modified_metrics)
    return {
        "paper_reported_targets": PAPER_TARGETS,
        "paper_experiment": {
            "description": "First experiment: paper-aligned SOH-only CNN-TCN-LSTM-Attention reproduction.",
            "actuals": paper_actuals,
            "deltas_vs_paper_percent": {
                key: _delta(value, PAPER_TARGETS.get(key))
                for key, value in paper_actuals.items()
                if key in PAPER_TARGETS
            },
            "fold_metrics": paper_metrics.get("fold_metrics", []),
        },
        "modified_experiment": {
            "description": "Second experiment: existing repo modified/thesis joint SOH+RUL model with physics-informed loss.",
            "actuals": modified_actuals,
            "datasets": modified_metrics.get("datasets", {}),
            "deltas_vs_paper_percent": {
                key: _delta(value, PAPER_TARGETS.get(key))
                for key, value in modified_actuals.items()
                if key in PAPER_TARGETS
            },
        },
        "interpretation": [
            "Run the paper experiment first and compare it directly to the reported paper targets.",
            "Run the modified repository experiment second; it adds RUL prediction and physics-informed monotonicity, so its SOH metric is comparable but its RUL output is an additional task.",
            "Latency and energy are hardware-dependent; compare them only for the machine or embedded target where they were measured.",
        ],
    }


def render_markdown(comparison: Dict[str, object]) -> str:
    targets = comparison["paper_reported_targets"]
    paper = comparison["paper_experiment"]["actuals"]
    modified = comparison["modified_experiment"]["actuals"]

    lines = [
        "# Paper vs Modified Experiment Comparison",
        "",
        "## Run order",
        "",
        "1. **Paper experiment**: SOH-only CNN-TCN-LSTM-Attention reproduction.",
        "2. **Modified repo experiment**: existing joint SOH/RUL physics-informed model.",
        "3. **Comparison**: paper experiment vs reported paper targets, then modified experiment vs the same reference context.",
        "",
        "## Main metrics",
        "",
        "| Metric | Paper reported | Paper experiment run | Modified repo run |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| SOH RMSE | {_fmt(targets['soh_rmse'])} | "
            f"{_fmt(paper.get('soh_rmse'))} | {_fmt(modified.get('soh_rmse'))} |"
        ),
        (
            f"| SOH R2 | {_fmt(targets['soh_r2'])} | "
            f"{_fmt(paper.get('soh_r2'))} | n/a for joint report |"
        ),
        (
            f"| RUL RMSE (cycles) | n/a | n/a | "
            f"{_fmt(modified.get('rul_rmse_cycles'), digits=2)} |"
        ),
        (
            f"| Parameters (M) | {_fmt(targets['parameters_millions'], digits=3)} | "
            f"{_fmt(paper.get('parameters_millions'), digits=3)} | "
            f"{_fmt(modified.get('parameters_millions'), digits=3)} |"
        ),
        (
            f"| Latency (ms/sample) | {_fmt(targets['latency_ms'], digits=3)} | "
            f"{_fmt(paper.get('latency_ms'), digits=3)} | "
            f"{_fmt(modified.get('latency_ms'), digits=3)} |"
        ),
        (
            f"| Energy (mJ/sample) | {_fmt(targets['energy_mj'], digits=3)} | "
            f"{_fmt(paper.get('energy_mj'), digits=3)} | "
            f"{_fmt(modified.get('energy_mj'), digits=3)} |"
        ),
        "",
        "## Modified experiment per-dataset metrics",
        "",
        "| Dataset | SOH RMSE | RUL RMSE (cycles) |",
        "| --- | ---: | ---: |",
    ]

    for dataset_name, metrics in comparison["modified_experiment"].get("datasets", {}).items():
        lines.append(
            f"| {dataset_name} | {_fmt(metrics.get('soh_rmse'))} | "
            f"{_fmt(metrics.get('rul_rmse_cycles'), digits=2)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Paper targets are taken from the linked Scientific Reports article.",
            "- The paper experiment is the first baseline; the modified experiment is the existing repo extension.",
            "- Accuracy depends on dataset split, number of epochs, hardware, and whether the full paper training schedule is used.",
            "- Generated outputs are intentionally ignored by git; rerun the workflow to regenerate local reports.",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison(args: argparse.Namespace) -> Dict[str, object]:
    paper_metrics = _load_json(Path(args.paper_metrics))
    modified_metrics = _load_json(Path(args.modified_metrics))
    comparison = build_comparison(paper_metrics, modified_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    md_path = output_dir / "comparison.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    md_path.write_text(render_markdown(comparison), encoding="utf-8")

    print(f"Saved comparison JSON to {json_path}")
    print(f"Saved comparison report to {md_path}")
    return comparison


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare paper experiment metrics with modified repo metrics.")
    parser.add_argument("--paper-metrics", required=True)
    parser.add_argument("--modified-metrics", required=True)
    parser.add_argument("--output-dir", default="paper_exp/outputs/comparison")
    return parser


if __name__ == "__main__":
    write_comparison(build_arg_parser().parse_args())

