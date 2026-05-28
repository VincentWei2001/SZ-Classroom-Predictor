from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import APP_ICON, PORTABLE_SCRIPT, ROOT

DIST_DIR = ROOT / "dist" / "school_app_light"
ZIP_PATH = ROOT / "release" / "school_app_light_portable.zip"

CONDA_DLLS = [
    "ffi.dll",
    "libcrypto-3-x64.dll",
    "libssl-3-x64.dll",
    "liblzma.dll",
    "libexpat.dll",
    "sqlite3.dll",
    "LIBBZ2.dll",
    "tcl86t.dll",
    "tk86t.dll",
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    if not APP_ICON.is_file():
        raise FileNotFoundError(f"缺少应用图标: {APP_ICON}")

    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--windowed",
            "--icon",
            str(APP_ICON),
            "--name",
            "school_app_light",
            "--paths",
            str(ROOT / "src"),
            "--add-data",
            f"{APP_ICON};assets",
            str(PORTABLE_SCRIPT),
        ]
    )

    built_dir = ROOT / "dist" / "school_app_light"
    if DIST_DIR != built_dir and built_dir.exists():
        if DIST_DIR.exists():
            shutil.rmtree(DIST_DIR)
        built_dir.rename(DIST_DIR)

    env_root = Path(sys.executable).resolve().parent
    target_root = DIST_DIR / "_internal"
    target_root.mkdir(parents=True, exist_ok=True)
    for dll_name in CONDA_DLLS:
        shutil.copy2(env_root / "Library" / "bin" / dll_name, target_root / dll_name)

    for candidate in [DIST_DIR / "school_app_models.bin", target_root / "school_app_models.bin"]:
        if candidate.exists():
            candidate.unlink()

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", DIST_DIR.parent, DIST_DIR.name)
    print(f"轻量压缩包已生成: {ZIP_PATH}")


if __name__ == "__main__":
    main()
