"""Run improved paper protocol (grouped CV, global scale) and refresh all artifacts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    steps = [
        [PY, str(ROOT / "run_paper_experiment.py"), "--require-real", "--cpu", "--cv"],
        [PY, str(ROOT / "generate_figures.py")],
        [PY, str(ROOT / "scripts" / "sanitize_paper_report.py")],
        [PY, str(ROOT / "scripts" / "verify_repo.py")],
        [PY, "-m", "pytest", str(ROOT / "tests"), "-q", "--tb=line"],
    ]
    for cmd in steps:
        print("\n>>>", " ".join(cmd))
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            print(f"FAILED: {' '.join(cmd)}")
            return rc
    print("\nImproved experiment pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
