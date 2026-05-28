from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import DIST_APP_DIR, ROOT

SOURCE_DIR = DIST_APP_DIR
TARGET_DIR = ROOT / "dist" / "school_app_min"
ZIP_PATH = ROOT / "release" / "school_app_min_release.zip"


def main() -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE_DIR / "school_app.exe", TARGET_DIR / "school_app.exe")
    shutil.copytree(SOURCE_DIR / "_internal", TARGET_DIR / "_internal")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", TARGET_DIR.parent, TARGET_DIR.name)
    print(f"min_release_zip={ZIP_PATH}")


if __name__ == "__main__":
    main()
