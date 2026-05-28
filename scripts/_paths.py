"""Shared project paths for build scripts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

APP_DIR = ROOT / "app"
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
PACKAGING_DIR = ROOT / "packaging"
BUILD_DIR = ROOT / "build"
DIST_DIR = ROOT / "dist"
RELEASE_DIR = ROOT / "release"
ARCHIVE_DIR = ROOT / "archive"

NOTEBOOK_PATH = APP_DIR / "classroom_predictor_app.ipynb"
PORTABLE_SCRIPT = SRC_DIR / "school_app_portable.py"
APP_ICON = ASSETS_DIR / "app_icon.ico"
MODEL_BUNDLE_PATH = MODELS_DIR / "school_app_models.bin"

DIST_APP_DIR = DIST_DIR / "school_app"
ZIP_PATH = RELEASE_DIR / "school_app_secure_portable.zip"
SECURE_PORTABLE_LAYOUT = RELEASE_DIR / "school_app_secure_portable" / "school_app"
PORTABLE_ZIP_PATH = RELEASE_DIR / "school_app_portable.zip"
PORTABLE_LAYOUT = RELEASE_DIR / "school_app_portable" / "school_app"
