from __future__ import annotations

import shutil
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
SOURCE_DIR = WORKSPACE / "dist" / "school_app"
TARGET_DIR = WORKSPACE / "dist" / "school_app_min"
ZIP_PATH = WORKSPACE / "school_app_min_release.zip"


def copy_required_items() -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE_DIR / "school_app.exe", TARGET_DIR / "school_app.exe")
    shutil.copytree(SOURCE_DIR / "_internal", TARGET_DIR / "_internal")
    readme_src = SOURCE_DIR / "README.txt"
    readme_dst = TARGET_DIR / "README.txt"
    if readme_src.exists():
        shutil.copy2(readme_src, readme_dst)
    else:
        readme_dst.write_text(
            "\n".join(
                [
                    "school_app 最小发布说明",
                    "",
                    "1. 双击 school_app.exe 启动。",
                    "2. 不要删除或拆分 _internal 文件夹。",
                    "3. 当前版本为最小发布目录，仅保留运行所需内容。",
                ]
            ),
            encoding="utf-8",
        )


def folder_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def main() -> None:
    copy_required_items()

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", TARGET_DIR.parent, TARGET_DIR.name)

    print(f"min_release_dir={TARGET_DIR}")
    print(f"min_release_dir_size_mb={folder_size_mb(TARGET_DIR):.2f}")
    print(f"min_release_zip={ZIP_PATH}")
    print(f"min_release_zip_size_mb={ZIP_PATH.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
