from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import (
    APP_ICON,
    DIST_APP_DIR,
    MODELS_DIR,
    MODEL_BUNDLE_PATH,
    PACKAGING_DIR,
    SECURE_PORTABLE_LAYOUT,
    PORTABLE_SCRIPT,
    RELEASE_DIR,
    ROOT,
    ZIP_PATH,
)

sys.path.insert(0, str(ROOT / "src"))
from secure_model_bundle import MODEL_BUNDLE_NAME, build_model_bundle

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


def prune_pyside6_runtime() -> None:
    """Keep only Qt Widgets runtime files; extra Qt DLLs break DLL loading on Windows."""
    pyside_dir = DIST_APP_DIR / "_internal" / "PySide6"
    if not pyside_dir.is_dir():
        return

    keep_dlls = {
        "Qt6Core.dll",
        "Qt6Gui.dll",
        "Qt6Widgets.dll",
        "pyside6.abi3.dll",
        "MSVCP140.dll",
        "MSVCP140_1.dll",
        "MSVCP140_2.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "opengl32sw.dll",
    }
    keep_pyds = {"QtCore.pyd", "QtGui.pyd", "QtWidgets.pyd"}

    for item in pyside_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".dll" and item.name not in keep_dlls:
            item.unlink(missing_ok=True)
        if item.is_file() and item.suffix.lower() == ".pyd" and item.name not in keep_pyds:
            item.unlink(missing_ok=True)

    plugins_dir = pyside_dir / "plugins"
    keep_plugin_dirs = {"platforms", "imageformats", "iconengines", "styles"}
    if plugins_dir.is_dir():
        for child in plugins_dir.iterdir():
            if child.is_dir() and child.name not in keep_plugin_dirs:
                shutil.rmtree(child, ignore_errors=True)


def copy_qt_support_dlls() -> None:
    target_root = DIST_APP_DIR / "_internal"
    pyside_dir = target_root / "PySide6"
    if not pyside_dir.is_dir():
        return

    for dll_name in ("icuuc.dll", "icudt73.dll", "icuin73.dll", "zlib.dll"):
        src = target_root / dll_name
        if src.is_file():
            shutil.copy2(src, pyside_dir / dll_name)


def copy_conda_dlls() -> None:
    env_root = Path(sys.executable).resolve().parent
    dll_root = env_root / "Library" / "bin"
    target_root = DIST_APP_DIR / "_internal"
    target_root.mkdir(parents=True, exist_ok=True)

    for dll_name in CONDA_DLLS:
        shutil.copy2(dll_root / dll_name, target_root / dll_name)

    xgboost_root = env_root / "Lib" / "site-packages" / "xgboost"
    xgboost_target = target_root / "xgboost"
    (xgboost_target / "lib").mkdir(parents=True, exist_ok=True)
    shutil.copy2(xgboost_root / "VERSION", xgboost_target / "VERSION")
    shutil.copy2(xgboost_root / "lib" / "xgboost.dll", xgboost_target / "lib" / "xgboost.dll")


def write_readme() -> None:
    readme_path = DIST_APP_DIR / "README.txt"
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
                "",
                "3. 重要说明",
                "请不要单独移动 school_app.exe。",
                "必须保留整个 school_app 文件夹结构不变。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    folder_names = sorted(
        p.name
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and p.name[:4].isdigit()
    )
    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle_info = build_model_bundle(
        MODELS_DIR,
        MODEL_BUNDLE_PATH,
        folder_names=folder_names,
    )
    print(f"已生成加密模型包: {MODEL_BUNDLE_NAME} -> {bundle_info}")

    if not APP_ICON.is_file():
        raise FileNotFoundError(f"缺少应用图标: {APP_ICON}（可运行 python scripts/build_app_icon.py 生成）")

    if not PORTABLE_SCRIPT.is_file():
        raise FileNotFoundError(f"缺少 {PORTABLE_SCRIPT}，请先运行 scripts/build_school_app.py")

    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--noupx",
            "--windowed",
            "--icon",
            str(APP_ICON),
            "--name",
            "school_app",
            "--onedir",
            "--paths",
            str(ROOT / "src"),
            "--hidden-import",
            "sklearn_joblib_compat",
            "--hidden-import",
            "secure_model_bundle",
            "--hidden-import",
            "sklearn.ensemble._forest",
            "--hidden-import",
            "sklearn.tree._classes",
            "--hidden-import",
            "sklearn.ensemble._base",
            "--hidden-import",
            "PySide6.QtCore",
            "--hidden-import",
            "PySide6.QtGui",
            "--hidden-import",
            "PySide6.QtWidgets",
            "--hidden-import",
            "matplotlib.backends.backend_qtagg",
            "--additional-hooks-dir",
            str(PACKAGING_DIR),
            "--runtime-hook",
            str(PACKAGING_DIR / "pyi_rth_00_qt_dll_path.py"),
            "--exclude-module",
            "PySide6.QtWebEngineCore",
            "--exclude-module",
            "PySide6.QtWebEngineWidgets",
            "--exclude-module",
            "PySide6.QtWebEngineQuick",
            "--exclude-module",
            "PySide6.QtNetwork",
            "--exclude-module",
            "PySide6.QtQml",
            "--exclude-module",
            "PySide6.QtQuick",
            "--exclude-module",
            "PySide6.QtOpenGL",
            "--exclude-module",
            "PySide6.QtPdf",
            "--exclude-module",
            "PySide6.QtSvg",
            "--exclude-module",
            "PySide6.QtVirtualKeyboard",
            "--add-data",
            f"{MODEL_BUNDLE_PATH};.",
            "--add-data",
            f"{APP_ICON};assets",
            str(PORTABLE_SCRIPT),
        ]
    )

    copy_conda_dlls()
    copy_qt_support_dlls()
    prune_pyside6_runtime()
    write_readme()

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    portable_parent = SECURE_PORTABLE_LAYOUT.parent
    if portable_parent.exists():
        shutil.rmtree(portable_parent, ignore_errors=True)
    portable_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DIST_APP_DIR, SECURE_PORTABLE_LAYOUT)
    print(f"便携目录已同步: {SECURE_PORTABLE_LAYOUT}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", portable_parent, "school_app")
    print(f"安全压缩包已生成: {ZIP_PATH}")


if __name__ == "__main__":
    main()
