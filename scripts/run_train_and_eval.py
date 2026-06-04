"""Train paper model on NASA + Oxford + CALCE, then run tests and generate figures."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], env: dict | None = None) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    py = sys.executable
    started = time.time()
    log_path = root / "results" / "full_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    run(
        [
            py,
            str(root / "run_paper_experiment.py"),
            "--require-real",
            "--cpu",
            "--cv",
        ],
        env=env,
    )
    run([py, "-m", "pytest", f"{root}/tests", "-q", "--tb=short"])
    run([py, f"{root}/generate_figures.py"])
    run([py, f"{root}/scripts/export_summary.py"])
    run([py, f"{root}/scripts/sync_results_docs.py"])

    print(f"\nDone in {(time.time() - started) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
