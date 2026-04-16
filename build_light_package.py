from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
PORTABLE_SCRIPT = WORKSPACE / "school_app_portable.py"
APP_ICON = WORKSPACE / "assets" / "app_icon.ico"
DIST_DIR = WORKSPACE / "dist" / "school_app_light"
ZIP_PATH = WORKSPACE / "school_app_light_portable.zip"
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
    subprocess.run(cmd, check=True, cwd=WORKSPACE)


def write_readme() -> None:
    (DIST_DIR / "README.txt").write_text(
        "\n".join(
            [
                "school_app 轻量发布说明",
                "",
                "1. 双击 school_app.exe 启动。",
                "2. 当前包未附带模型包，首次预测时会尝试自动下载 school_app_models.bin。",
                "3. 请在程序目录填写 model_bundle_url.txt，内容为模型包直链地址。",
                "4. 不要删除或拆分 _internal 文件夹。",
            ]
        ),
        encoding="utf-8",
    )


def write_url_placeholder() -> None:
    (DIST_DIR / "model_bundle_url.txt").write_text(
        "https://example.com/school_app_models.bin",
        encoding="utf-8",
    )


def remove_embedded_model_bundle() -> None:
    for candidate in [
        DIST_DIR / "school_app_models.bin",
        DIST_DIR / "_internal" / "school_app_models.bin",
    ]:
        if candidate.exists():
            candidate.unlink()


def copy_runtime_support_files() -> None:
    env_root = Path(sys.executable).resolve().parent
    dll_root = env_root / "Library" / "bin"
    target_root = DIST_DIR / "_internal"
    target_root.mkdir(parents=True, exist_ok=True)

    for dll_name in CONDA_DLLS:
        shutil.copy2(dll_root / dll_name, target_root / dll_name)

    xgboost_root = env_root / "Lib" / "site-packages" / "xgboost"
    xgboost_target = target_root / "xgboost"
    (xgboost_target / "lib").mkdir(parents=True, exist_ok=True)
    shutil.copy2(xgboost_root / "VERSION", xgboost_target / "VERSION")
    shutil.copy2(xgboost_root / "lib" / "xgboost.dll", xgboost_target / "lib" / "xgboost.dll")


def main() -> None:
    if not APP_ICON.is_file():
        raise FileNotFoundError(f"缺少应用图标: {APP_ICON}（可运行 python scripts/build_app_icon.py 生成）")

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
            "--add-data",
            f"{APP_ICON};assets",
            str(PORTABLE_SCRIPT),
        ]
    )

    built_dir = WORKSPACE / "dist" / "school_app_light"
    if DIST_DIR != built_dir and built_dir.exists():
        if DIST_DIR.exists():
            shutil.rmtree(DIST_DIR)
        built_dir.rename(DIST_DIR)

    copy_runtime_support_files()
    remove_embedded_model_bundle()
    write_readme()
    write_url_placeholder()

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", DIST_DIR.parent, DIST_DIR.name)
    print(f"轻量压缩包已生成: {ZIP_PATH}")


if __name__ == "__main__":
    main()
