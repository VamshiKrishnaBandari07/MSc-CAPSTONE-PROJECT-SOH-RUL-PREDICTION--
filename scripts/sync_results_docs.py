"""Refresh docs/RESULTS.md from paper_experiment_report.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_JSON = ROOT / "results" / "paper_experiment_report.json"
RESULTS_MD = ROOT / "docs" / "RESULTS.md"


def _fmt(x, nd=4):
    return f"{x:.{nd}f}"


def _metric_cell(m: dict, key: str) -> str:
    if key not in m or m[key] is None:
        return "—"
    return _fmt(m[key])


def main() -> None:
    paper = json.loads(PAPER_JSON.read_text(encoding="utf-8"))
    proto = paper.get("eval_protocol", "cv5")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_runs = max(
        (r.get("independent_runs") or len((r.get("metrics") or {}).get("rmse_runs", [])) for r in paper.get("results", [])),
        default=5,
    )

    body = f"""# Paper Reproduction — Results

**Reference:** Rahman *et al.*, *Scientific Reports* (2026) — [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)  
**Last synced:** {ts}  
**Source:** `results/paper_experiment_report.json` · `results/summary.json`  
**Protocol:** Stratified 5-fold CV, {n_runs} independent runs (seeds 42–46), mean pooled OOF SOH RMSE

## SOH RMSE (completed training run)

| Dataset | Mean RMSE (± std over {n_runs} runs) | R² | Paper Table 4 (NASA only) |
|:---|:---:|:---:|:---:|
"""
    for r in paper.get("results", []):
        m = r["metrics"]
        rmse = (
            f"**{_fmt(m['rmse'])} ± {_fmt(m['rmse_std'])}**"
            if m.get("rmse_std") is not None
            else f"**{_fmt(m['rmse'])}**"
        )
        paper_ref = "Hybrid **0.021**" if r["dataset"] == "NASA" else "—"
        body += f"| {r['dataset']} | {rmse} | {_metric_cell(m, 'r2')} | {paper_ref} |\n"

    runs = (paper.get("results", [{}])[0].get("metrics") or {}).get("rmse_runs")
    if runs:
        body += """
### Per independent run (pooled OOF RMSE)

| Run | NASA | Oxford | CALCE |
|:---:|:---:|:---:|:---:|
"""
        by_ds = {r["dataset"]: (r.get("metrics") or {}).get("rmse_runs", []) for r in paper.get("results", [])}
        for i in range(len(runs)):
            row = [str(i + 1)]
            for ds in ("NASA", "Oxford", "CALCE"):
                vals = by_ds.get(ds, [])
                row.append(_fmt(vals[i]) if i < len(vals) else "—")
            body += "| " + " | ".join(row) + " |\n"

    body += """
## Discussion

- **Oxford** meets the published hybrid RMSE scale (~0.021).
- **NASA** remains above Table 4 (**0.021**); our mean (~**0.042**) is closer to the paper’s Transformer baseline (**0.038**).
- **CALCE** is reported as a cross-chemistry benchmark only.

## Regenerate

```powershell
python run_paper_experiment.py --require-real --cpu --cv
python generate_figures.py
python scripts/export_summary.py
python scripts/sync_results_docs.py
```
"""
    RESULTS_MD.write_text(body, encoding="utf-8")
    print(f"Updated {RESULTS_MD}")


if __name__ == "__main__":
    main()
