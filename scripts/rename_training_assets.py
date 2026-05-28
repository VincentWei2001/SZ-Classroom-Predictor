from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import ARCHIVE_DIR, DATA_DIR, MODELS_DIR, ROOT

sys.path.insert(0, str(ROOT / "src"))
from scenario_registry import (  # noqa: E402
    ARCHIVE_LEGACY_FOLDERS,
    LEGACY_CSV_RENAMES,
    LEGACY_MODEL_FOLDER_RENAMES,
)


def _rename_path(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        print(f"  跳过（目标已存在）: {dst.name}")
        return
    action = "将重命名" if dry_run else "已重命名"
    print(f"  {action}: {src.name} -> {dst.name}")
    if not dry_run:
        src.rename(dst)


def rename_models(dry_run: bool) -> None:
    csv_dir = DATA_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    archive_models = ARCHIVE_DIR / "legacy_models"
    archive_models.mkdir(parents=True, exist_ok=True)

    print("模型目录:")
    for old_name, new_name in LEGACY_MODEL_FOLDER_RENAMES.items():
        _rename_path(MODELS_DIR / old_name, MODELS_DIR / new_name, dry_run)

    print("归档旧版（不再发布）:")
    for legacy_name in ARCHIVE_LEGACY_FOLDERS:
        src = MODELS_DIR / legacy_name
        if not src.is_dir():
            continue
        dst = archive_models / legacy_name
        label = "将移动" if dry_run else "已移动"
        print(f"  {label}: {legacy_name} -> archive/legacy_models/")
        if not dry_run:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))

    print("训练 CSV:")
    for old_name, new_name in LEGACY_CSV_RENAMES.items():
        _rename_path(csv_dir / old_name, csv_dir / new_name, dry_run)
        _rename_path(ROOT / old_name, csv_dir / new_name, dry_run)

    if not dry_run:
        unmapped = [
            p.name
            for p in MODELS_DIR.iterdir()
            if p.is_dir()
            and p.name not in LEGACY_MODEL_FOLDER_RENAMES.values()
            and p.name not in ARCHIVE_LEGACY_FOLDERS
            and p.name not in {v for v in LEGACY_MODEL_FOLDER_RENAMES.values()}
        ]
        # only warn about dirs that look like old experiment names
        legacy_like = [n for n in unmapped if len(n) >= 4 and n[:4].isdigit()]
        if legacy_like:
            print("未映射的模型目录（请手动处理）:", ", ".join(sorted(legacy_like)))


def main() -> None:
    parser = argparse.ArgumentParser(description="重命名训练 CSV 与模型目录为正式名称")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将执行的操作")
    args = parser.parse_args()
    rename_models(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
