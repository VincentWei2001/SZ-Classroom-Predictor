# 项目目录说明

## 一眼看懂

| 目录 | 用途 |
|------|------|
| `app/` | 应用程序 Notebook（开发入口） |
| `src/` | Python 源码（`school_app_portable.py` 由脚本自动生成；`scenario_registry.py` 为场景命名表） |
| `models/` | 训练好的模型目录（正式名如 `south_0deg_base/`；明文 joblib） |
| `data/csv/` | 训练/分析用 CSV（正式名如 `south_0deg_base.csv`） |
| `scripts/` | 打包、发布、工具脚本 |
| `packaging/` | PyInstaller 运行时钩子 |
| `assets/` | 图标等资源 |
| `release/` | 可发布的 zip 与便携目录 |
| `archive/` | 旧 Notebook、归档模型、日志等 |
| `build/` `dist/` | PyInstaller 中间产物（可删，会重新生成） |

## 场景命名（10 组）

| 目录 / CSV 前缀 | 界面含义 |
|-----------------|----------|
| `south_0deg_base` | 南向 · 基准 |
| `south_0deg_overhang` | 南向 · 悬挑 |
| `south_0deg_vertical` | 南向 · 垂直 |
| `south_0deg_combined` | 南向 · 组合 |
| `south_0deg_frame` | 南向 · 框式 |
| `north_0deg_base` | 北向 · 基准 |
| `north_0deg_overhang` | 北向 · 悬挑 |
| `north_0deg_vertical` | 北向 · 垂直 |
| `north_0deg_combined` | 北向 · 组合（修正版训练数据） |
| `north_0deg_frame` | 北向 · 框式 |

从旧实验编号迁移：运行 `python scripts/rename_training_assets.py`（加 `--dry-run` 可预览）。

## 日常开发

```powershell
# 1. 在 app/ 里改 notebook 后，同步到可打包脚本
D:\Anaconda\envs\school_app\python.exe scripts\build_school_app.py

# 2. 本地运行（不要用系统 Python 3.12）
D:\Anaconda\envs\school_app\python.exe src\school_app_portable.py
# 或 .\scripts\run_local.ps1

# 3. 打 Windows 便携包（明文 models/，无加密）
D:\Anaconda\envs\school_app\python.exe scripts\build_portable_package.py
```

产物：`release\school_app_portable.zip`（内含 `school_app\models\` 与各场景 joblib）

## 旧版加密打包（可选）

```powershell
D:\Anaconda\envs\school_app\python.exe scripts\build_secure_package.py
```

产物：`release\school_app_secure_portable.zip`（单文件 `school_app_models.bin`）

## 发布到 GitHub

见 `release\README_GitHub_Release.txt`，或运行 `scripts\publish_release.ps1`（需 `GITHUB_TOKEN`）。
