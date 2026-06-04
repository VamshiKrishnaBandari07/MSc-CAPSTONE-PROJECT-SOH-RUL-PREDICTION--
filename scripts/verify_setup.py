#!/usr/bin/env python3
"""Redirect to paper repository verification."""

import subprocess
import sys

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "scripts/verify_repo.py"]))
