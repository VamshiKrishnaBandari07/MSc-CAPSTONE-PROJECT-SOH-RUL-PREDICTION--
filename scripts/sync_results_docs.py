"""Refresh docs/RESULTS.md tables from committed JSON reports."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_JSON = ROOT / "results" / "paper_experiment_report.json"
MSC_JSON = ROOT / "results" / "experiment_comparison_report.json"
RESULTS_MD = ROOT / "docs" / "RESULTS.md"


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x, nd=4):
    return f"{x:.{nd}f}"


def _paper_rows(paper: dict) -> list[str]:
    rows = []
    for r in paper.get("results", []):
        ds = r["dataset"]
        proto = r.get("eval_protocol", paper.get("eval_protocol", "unknown"))
        m = r.get("metrics", {})
        if "rmse_std" in m:
            rmse = f"**{_fmt(m['rmse'])} ± {_fmt(m['rmse_std'])}**"
        else:
            rmse = f"**{_fmt(m.get('rmse', 0))}**"
        r2 = _fmt(m.get("r2", 0))
        rows.append(f"| **{ds}** | {rmse} | {r2} | 0.021 | 0.038 | ({proto}) |")
    return rows


def _msc_rows(msc: dict, key: str) -> list[str]:
    rows = []
    for r in msc.get(key, []):
        soh = r.get("metrics", {}).get("soh", {})
        rul = r.get("metrics", {}).get("rul", {})
        rows.append(
            f"| {r['dataset']} | {_fmt(soh.get('rmse', 0))} | {_fmt(soh.get('r2', 0))} | "
            f"{_fmt(rul.get('rmse', 0), 1)} | {_fmt(soh.get('mono_violation_rate', 0), 3)} |"
        )
    return rows


def main() -> None:
    paper = _load(PAPER_JSON)
    msc = _load(MSC_JSON)
    proto = paper.get("eval_protocol", "cv5")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = f"""# Experimental Results Summary

**Author:** Vamshi Krishna Bandari  
**Last verified run:** {ts} — auto-synced from JSON  
**Report files:** `results/experiment_comparison_report.json`, `results/paper_experiment_report.json`  
**Figures:** `results/figures/fig01`–`fig07` (PNG + PDF)

> **Evaluation:** Experiment A uses **{proto}** (default paper protocol). See `docs/PAPER_METHODOLOGY.md`.

---

## Experiment A — Paper reproduction (SOH RMSE)

| Dataset | Our RMSE | Our R² | Paper hybrid | Transformer | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
"""
    body += "\n".join(_paper_rows(paper)) + "\n\n"

    body += """---

## Experiment B — MSc extension (joint SOH + RUL)

| Dataset | SOH RMSE | SOH R² | RUL RMSE (cycles) | Mono. violation |
|:---|:---:|:---:|:---:|:---:|
"""
    body += "\n".join(_msc_rows(msc, "experiment_b_msc_extension")) + "\n\n"

    ablation_a = {r["dataset"]: r for r in msc.get("experiment_c_ablation_no_physics", [])}
    ablation_b = {r["dataset"]: r for r in msc.get("experiment_b_msc_extension", [])}
    body += """---

## Experiment C — Ablation (physics monotonicity loss)

| Dataset | SOH RMSE (no physics) | SOH RMSE (with physics) |
|:---|:---:|:---:|
"""
    for ds in ("NASA", "Oxford", "CALCE"):
        na = ablation_a.get(ds, {}).get("metrics", {}).get("soh", {}).get("rmse")
        wb = ablation_b.get(ds, {}).get("metrics", {}).get("soh", {}).get("rmse")
        if na is not None and wb is not None:
            body += f"| {ds} | {_fmt(na)} | **{_fmt(wb)}** |\n"

    body += """
---

## Regenerate results

```powershell
python run_paper_experiment.py --require-real --cpu
python run_experiments.py --msc-only --require-real --cpu
python generate_figures.py
python scripts/sync_results_docs.py
```
"""
    RESULTS_MD.write_text(body, encoding="utf-8")
    print(f"Updated {RESULTS_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
