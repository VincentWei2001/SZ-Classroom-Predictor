from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from secure_model_bundle import MODEL_BUNDLE_NAME, build_model_bundle

WORKSPACE = Path(__file__).resolve().parent
PORTABLE_SCRIPT = WORKSPACE / "school_app_portable.py"
DIST_DIR = WORKSPACE / "dist" / "school_app"
ZIP_PATH = WORKSPACE / "school_app_secure_portable.zip"
EXCLUDED_MODEL_FOLDERS = {
    "0312_2000_North0°_Overhang+Vertical(1)",
}

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


def copy_conda_dlls() -> None:
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


def write_readme() -> None:
    readme_path = DIST_DIR / "README.txt"
    readme_path.write_text(
        "\n".join(
            [
                "school_app 安全打包说明",
                "",
                "1. 运行方式",
                "双击同目录下的 school_app.exe 即可启动。",
                "",
                "2. 当前模型保护方式",
                "模型已打包为单个加密文件 school_app_models.bin。",
                "普通使用者不会直接看到各个模型文件夹和 joblib 文件。",
                "",
                "3. 重要说明",
                "请不要单独移动 school_app.exe。",
                "必须保留整个 school_app 文件夹结构不变。",
                "",
                "4. 建议部署方式",
                "将整个 school_app 文件夹复制到另一台 Windows 电脑后再运行。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    folder_names = sorted(
        p.name
        for p in WORKSPACE.iterdir()
        if p.is_dir() and p.name[:4].isdigit() and p.name not in EXCLUDED_MODEL_FOLDERS
    )
    bundle_info = build_model_bundle(
        WORKSPACE,
        WORKSPACE / MODEL_BUNDLE_NAME,
        folder_names=folder_names,
    )
    print(f"已生成加密模型包: {MODEL_BUNDLE_NAME} -> {bundle_info}")

    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--windowed",
            "--name",
            "school_app",
            "--onedir",
            "--add-data",
            f"{WORKSPACE / MODEL_BUNDLE_NAME};.",
            str(PORTABLE_SCRIPT),
        ]
    )

    copy_conda_dlls()
    write_readme()

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", DIST_DIR.parent, DIST_DIR.name)
    print(f"安全压缩包已生成: {ZIP_PATH}")


if __name__ == "__main__":
    main()
