"""Remove training data and extra notebooks from the git index (files stay on disk)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

KEEP_NOTEBOOK = "预测应用完整版.ipynb"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    out = subprocess.check_output(["git", "ls-files", "-z"])
    paths = [p.decode("utf-8") for p in out.split(b"\x00") if p]
    removed = 0
    for p in paths:
        drop = False
        if p.endswith(".joblib"):
            drop = True
        elif p.endswith(".csv"):
            drop = True
        elif p.endswith(".ipynb") and p != KEEP_NOTEBOOK:
            drop = True
        if drop:
            r = subprocess.run(["git", "rm", "--cached", "-q", "--", p], capture_output=True)
            if r.returncode == 0:
                removed += 1
    print(f"removed_from_index={removed}", file=sys.stderr)


if __name__ == "__main__":
    main()
