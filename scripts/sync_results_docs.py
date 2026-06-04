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


def main() -> None:
    paper = json.loads(PAPER_JSON.read_text(encoding="utf-8"))
    proto = paper.get("eval_protocol", "cv5")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = f"""# Paper Reproduction — Results

**Reference:** Rahman et al., *Scientific Reports* (2026) — [DOI 10.1038/s41598-026-39911-8](https://doi.org/10.1038/s41598-026-39911-8)  
**Last synced:** {ts}  
**Source:** `results/paper_experiment_report.json`  
**Protocol:** {proto}

## SOH RMSE (stratified 5-fold CV)

| Dataset | Our RMSE | Our R² | Paper hybrid | Transformer |
|:---|:---:|:---:|:---:|:---:|
"""
    for r in paper.get("results", []):
        m = r["metrics"]
        rmse = f"**{_fmt(m['rmse'])} ± {_fmt(m['rmse_std'])}**" if "rmse_std" in m else f"**{_fmt(m['rmse'])}**"
        body += f"| {r['dataset']} | {rmse} | {_fmt(m.get('r2', 0))} | 0.021 | 0.038 |\n"

    body += """
## Regenerate

```powershell
python run_paper_experiment.py --require-real --cpu
python generate_figures.py
python scripts/sync_results_docs.py
```
"""
    RESULTS_MD.write_text(body, encoding="utf-8")
    print(f"Updated {RESULTS_MD}")


if __name__ == "__main__":
    main()
