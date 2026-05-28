"""Canonical scenario folder names for classroom orientation + shading models."""
from __future__ import annotations

import os
from pathlib import Path

ORI_SOUTH = "南向 (South 0°)"
ORI_NORTH = "北向 (North 0°)"
SHADE_BASE = "基准 (Base)"
SHADE_OVERHANG = "悬挑 (Overhang)"
SHADE_VERTICAL = "垂直 (Vertical)"
SHADE_COMBINED = "组合 (O+V)"
SHADE_FRAME = "框式 (Frame)"

SCENARIO_FOLDERS: dict[str, tuple[str, str]] = {
    "south_0deg_base": (ORI_SOUTH, SHADE_BASE),
    "south_0deg_overhang": (ORI_SOUTH, SHADE_OVERHANG),
    "south_0deg_vertical": (ORI_SOUTH, SHADE_VERTICAL),
    "south_0deg_combined": (ORI_SOUTH, SHADE_COMBINED),
    "south_0deg_frame": (ORI_SOUTH, SHADE_FRAME),
    "north_0deg_base": (ORI_NORTH, SHADE_BASE),
    "north_0deg_overhang": (ORI_NORTH, SHADE_OVERHANG),
    "north_0deg_vertical": (ORI_NORTH, SHADE_VERTICAL),
    "north_0deg_combined": (ORI_NORTH, SHADE_COMBINED),
    "north_0deg_frame": (ORI_NORTH, SHADE_FRAME),
}

LEGACY_MODEL_FOLDER_RENAMES: dict[str, str] = {
    "0304_800_South0°(1)": "south_0deg_base",
    "0304_2000_South0°_Overhang(1)": "south_0deg_overhang",
    "0307_2000_South0°_Vertical(1)": "south_0deg_vertical",
    "0311_2000_South0°_Overhang+Vertical(1)": "south_0deg_combined",
    "0304_1500_South0°_Frame(1)": "south_0deg_frame",
    "0305_800_North0°(1)": "north_0deg_base",
    "0306_2000_North0°_Overhang(1)": "north_0deg_overhang",
    "0310_2000_North0°_Vertical(1)": "north_0deg_vertical",
    "0312_2000_North0°_Overhang+Vertical(1)_replaced": "north_0deg_combined",
    "0306_1500_North0°_Frame(1)": "north_0deg_frame",
}

LEGACY_CSV_RENAMES: dict[str, str] = {
    old + ".csv": new + ".csv" for old, new in LEGACY_MODEL_FOLDER_RENAMES.items()
}

ARCHIVE_LEGACY_FOLDERS = frozenset(
    {
        "0312_2000_North0°_Overhang+Vertical(1)",
    }
)

RUNTIME_MODEL_PREFIXES = ("xgb_model_", "lgbm_model_", "rf_model_", "meta_model_")
RUNTIME_MODEL_FILES = ("X_train.joblib",)


def list_available_scenarios(base_path: str | Path) -> list[str]:
    """Return relative paths like models/south_0deg_base for existing scenario dirs."""
    base_path = Path(base_path)
    found: list[str] = []
    for folder_id in SCENARIO_FOLDERS:
        for rel in (Path("models") / folder_id, Path(folder_id)):
            if (base_path / rel).is_dir():
                found.append(rel.as_posix())
                break
    return sorted(found)


def resolve_scenario_path(base_path: str | Path, rel_folder: str) -> Path:
    return Path(base_path) / rel_folder.replace("/", os.sep)
