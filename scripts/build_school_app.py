from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import NOTEBOOK_PATH, PORTABLE_SCRIPT, ROOT

BASE_PATH_REPLACEMENT = """
from pathlib import Path


def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    return str(Path(__file__).resolve().parent.parent)


BASE_PATH = get_base_path()


def _list_model_folders(base_path):
    src_dir = os.path.join(base_path, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from scenario_registry import list_available_scenarios

    scenarios = list_available_scenarios(base_path)
    if scenarios:
        return scenarios
    try:
        from secure_model_bundle import list_bundled_model_folders, model_bundle_exists

        if model_bundle_exists(base_path):
            return list_bundled_model_folders(base_path)
    except Exception:
        pass
    return scenarios


def _load_model_file(folder_marker, filename):
    if os.path.isdir(folder_marker):
        return joblib.load(os.path.join(folder_marker, filename))
    nested = os.path.join(folder_marker, filename)
    if os.path.isfile(nested):
        return joblib.load(nested)
    try:
        from secure_model_bundle import load_bundled_model
    except Exception as exc:
        raise FileNotFoundError(
            f"未找到模型文件: {folder_marker} / {filename}"
        ) from exc
    folder_name = os.path.basename(str(folder_marker).rstrip("\\\\/")) or str(folder_marker)
    return load_bundled_model(BASE_PATH, folder_name, filename)
""".strip()


PATTERN_REPLACEMENTS: list[tuple[str, str]] = [
    (
        "folders = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]",
        "folders = _list_model_folders(BASE_PATH)",
    ),
    (
        'joblib.load(os.path.join(folder, f"{m}_model_{t}.joblib"))',
        '_load_model_file(folder, f"{m}_model_{t}.joblib")',
    ),
    (
        'xtrain_path = os.path.join(folder, "X_train.joblib")\n'
        '                if os.path.exists(xtrain_path):\n'
        "                    cache['_feature_order'] = joblib.load(xtrain_path).columns.tolist()\n"
        '                else:\n'
        "                    cache['_feature_order'] = None",
        "try:\n"
        "                    cache['_feature_order'] = _load_model_file(folder, \"X_train.joblib\").columns.tolist()\n"
        "                except (FileNotFoundError, OSError):\n"
        "                    cache['_feature_order'] = None",
    ),
]


def export_notebook_to_script() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells: list[str] = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.strip():
            code_cells.append(source.rstrip())

    if not code_cells:
        raise RuntimeError("Notebook 中未找到可导出的代码单元。")

    script_content = "\n\n\n".join(code_cells) + "\n"

    if "def get_base_path():" not in script_content:
        import re

        pattern = re.compile(
            r'^BASE_PATH\s*=\s*r["\'].*?["\']\s*$',
            re.MULTILINE,
        )
        if not pattern.search(script_content):
            raise RuntimeError("未找到 BASE_PATH 定义，无法生成便携版脚本。")
        script_content = pattern.sub(
            BASE_PATH_REPLACEMENT + "\n",
            script_content,
            count=1,
        )
    else:
        script_content = script_content.replace(
            'from pathlib import Path\n',
            'from pathlib import Path\n',
            1,
        )

    for old, new in PATTERN_REPLACEMENTS:
        if old in script_content:
            script_content = script_content.replace(old, new)

    PORTABLE_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    PORTABLE_SCRIPT.write_text(script_content, encoding="utf-8")


if __name__ == "__main__":
    export_notebook_to_script()
    print(f"已生成便携版入口脚本: {PORTABLE_SCRIPT}")
