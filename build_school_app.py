from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
NOTEBOOK_PATH = WORKSPACE / "预测应用完整版.ipynb"
OUTPUT_SCRIPT_PATH = WORKSPACE / "school_app_portable.py"

BASE_PATH_REPLACEMENT = """
def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


BASE_PATH = get_base_path()
""".strip()


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
    original_base_path = 'BASE_PATH = r"C:\\Users\\GIGABYTE\\Desktop\\Test\\Analysis\\新（可用）\\CSV"'

    if original_base_path not in script_content:
        raise RuntimeError("未找到原始 BASE_PATH，无法生成便携版脚本。")

    script_content = script_content.replace(original_base_path, BASE_PATH_REPLACEMENT, 1)
    OUTPUT_SCRIPT_PATH.write_text(script_content, encoding="utf-8")


if __name__ == "__main__":
    export_notebook_to_script()
    print(f"已生成便携版入口脚本: {OUTPUT_SCRIPT_PATH}")
