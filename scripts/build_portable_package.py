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
    PACKAGING_DIR,
    PORTABLE_LAYOUT,
    PORTABLE_SCRIPT,
    PORTABLE_ZIP_PATH,
    RELEASE_DIR,
    ROOT,
)

sys.path.insert(0, str(ROOT / "src"))
from scenario_registry import SCENARIO_FOLDERS  # noqa: E402
from secure_model_bundle import RUNTIME_MODEL_PREFIXES  # noqa: E402

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


def remove_bundled_conda_icu() -> None:
    """Drop the conda-forge ICU DLLs PyInstaller picks up off PATH.

    Qt6Core in this PySide6 build imports the *unversioned* ICU API
    (ucnv_open, ...) which only the Windows system ICU (System32\\icuuc.dll)
    provides. The conda ICU exports versioned names (ucnv_open_73) instead, so
    if it is bundled into _internal it shadows the good system DLL and Qt6Core
    fails to load with WinError 127 (procedure not found). Removing it lets the
    loader fall back to the correct system ICU.
    """
    target_root = DIST_APP_DIR / "_internal"
    for icu_dir in (target_root, target_root / "PySide6"):
        if not icu_dir.is_dir():
            continue
        for icu_dll in icu_dir.glob("icu*.dll"):
            icu_dll.unlink(missing_ok=True)
            print(f"  已移除错误打包的 conda ICU: {icu_dll.relative_to(DIST_APP_DIR)}")


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


def _runtime_files_in_folder(folder: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name == "X_train.joblib" or name.startswith(RUNTIME_MODEL_PREFIXES):
            files.append(path)
    return files


def copy_runtime_models(target_app_dir: Path) -> None:
    """Copy plaintext scenario model folders (runtime joblib only) beside the exe."""
    models_target = target_app_dir / "models"
    if models_target.exists():
        shutil.rmtree(models_target)
    models_target.mkdir(parents=True)

    copied = 0
    for folder_id in SCENARIO_FOLDERS:
        src_folder = MODELS_DIR / folder_id
        if not src_folder.is_dir():
            print(f"  警告: 缺少模型目录 {folder_id}")
            continue
        runtime_files = _runtime_files_in_folder(src_folder)
        if not runtime_files:
            print(f"  警告: {folder_id} 中无运行时 joblib 文件")
            continue
        dst_folder = models_target / folder_id
        dst_folder.mkdir(parents=True)
        for src_file in runtime_files:
            shutil.copy2(src_file, dst_folder / src_file.name)
        copied += 1

    if copied < len(SCENARIO_FOLDERS):
        print(f"已复制 {copied}/{len(SCENARIO_FOLDERS)} 个场景模型目录")
    else:
        print(f"已复制全部 {copied} 个场景模型目录到 models/")


def write_readme() -> None:
    readme_path = DIST_APP_DIR / "README.txt"
    readme_path.write_text(
        "\n".join(
            [
                "school_app 便携版说明",
                "",
                "1. 运行方式",
                "双击同目录下的 school_app.exe 即可启动。",
                "",
                "2. 模型文件",
                "预测模型位于 models/ 子目录（明文 joblib，可公开）。",
                "",
                "3. 重要说明",
                "请不要单独移动 school_app.exe。",
                "必须保留整个 school_app 文件夹结构不变（含 _internal 与 models）。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
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
            "scenario_registry",
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
            f"{APP_ICON};assets",
            str(PORTABLE_SCRIPT),
        ]
    )

    copy_conda_dlls()
    remove_bundled_conda_icu()
    prune_pyside6_runtime()
    copy_runtime_models(DIST_APP_DIR)
    write_readme()

    for stray_bin in [
        DIST_APP_DIR / "school_app_models.bin",
        DIST_APP_DIR / "_internal" / "school_app_models.bin",
    ]:
        stray_bin.unlink(missing_ok=True)

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    portable_parent = PORTABLE_LAYOUT.parent
    if portable_parent.exists():
        shutil.rmtree(portable_parent, ignore_errors=True)
    portable_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DIST_APP_DIR, PORTABLE_LAYOUT)
    print(f"便携目录已同步: {PORTABLE_LAYOUT}")

    if PORTABLE_ZIP_PATH.exists():
        PORTABLE_ZIP_PATH.unlink()
    shutil.make_archive(str(PORTABLE_ZIP_PATH.with_suffix("")), "zip", portable_parent, "school_app")
    print(f"便携压缩包已生成: {PORTABLE_ZIP_PATH}")


if __name__ == "__main__":
    main()
