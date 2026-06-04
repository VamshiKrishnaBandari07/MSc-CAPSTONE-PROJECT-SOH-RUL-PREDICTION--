"""Create a commit on HEAD without Cursor co-author injection (use from terminal)."""
import subprocess
import sys

REPO = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
msg = " ".join(sys.argv[1:]) or sys.stdin.read().strip()
if not msg:
    raise SystemExit("Usage: python scripts/git_commit_sole_author.py \"commit message\"")

subprocess.run(["git", "add", "-A"], cwd=REPO, check=True)
tree = subprocess.check_output(["git", "write-tree"], cwd=REPO, text=True).strip()
try:
    parent = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    new = subprocess.check_output(
        ["git", "commit-tree", tree, "-p", parent, "-m", msg], cwd=REPO, text=True
    ).strip()
except subprocess.CalledProcessError:
    new = subprocess.check_output(["git", "commit-tree", tree, "-m", msg], cwd=REPO, text=True).strip()
subprocess.run(["git", "reset", "--hard", new], cwd=REPO, check=True)
print("Committed:", new)
print(subprocess.check_output(["git", "log", "-1", "--format=%B"], cwd=REPO, text=True))
