from __future__ import annotations

import json
import os
import zlib
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
from cryptography.fernet import Fernet

MODEL_BUNDLE_NAME = "school_app_models.bin"
MODEL_BUNDLE_KEY = b"qtmK7t4N-VIMg9wPo6Am4Wrp1LBxGW4FQ5HOw3VTRIY="
RUNTIME_MODEL_PREFIXES = ("xgb_model_", "lgbm_model_", "rf_model_", "meta_model_")
MANIFEST_HEADER_BYTES = 8
MODEL_BUNDLE_ENV_VAR = "SCHOOL_APP_MODEL_BUNDLE"
MODEL_BUNDLE_URL_ENV_VAR = "SCHOOL_APP_MODEL_URL"
MODEL_BUNDLE_URL_FILE = "model_bundle_url.txt"


def _cipher() -> Fernet:
    return Fernet(MODEL_BUNDLE_KEY)


def _search_roots(base_path: str | Path) -> list[Path]:
    base_root = Path(base_path)
    roots = [base_root]
    if base_root.name.lower() == "_internal":
        roots.append(base_root.parent)
    return roots


def get_preferred_bundle_output_path(base_path: str | Path) -> Path:
    roots = _search_roots(base_path)
    return roots[-1] / MODEL_BUNDLE_NAME


def get_bundle_path(base_path: str | Path) -> Path:
    env_path = os.environ.get(MODEL_BUNDLE_ENV_VAR, "").strip()
    if env_path:
        env_bundle = Path(env_path)
        if env_bundle.is_file():
            return env_bundle

    for root in _search_roots(base_path):
        candidate = root / MODEL_BUNDLE_NAME
        if candidate.is_file():
            return candidate

    return get_preferred_bundle_output_path(base_path)


def get_model_bundle_url(base_path: str | Path) -> str:
    env_url = os.environ.get(MODEL_BUNDLE_URL_ENV_VAR, "").strip()
    if env_url:
        return env_url

    for root in _search_roots(base_path):
        url_file = root / MODEL_BUNDLE_URL_FILE
        if url_file.is_file():
            return url_file.read_text(encoding="utf-8").strip()

    return ""


def model_bundle_exists(base_path: str | Path) -> bool:
    return get_bundle_path(base_path).is_file()


@lru_cache(maxsize=8)
def load_bundle_manifest(base_path: str | Path) -> dict[str, dict[str, int]]:
    bundle_path = get_bundle_path(base_path)
    with bundle_path.open("rb") as f:
        manifest_len = int.from_bytes(f.read(MANIFEST_HEADER_BYTES), "big")
        manifest_bytes = f.read(manifest_len)
    return json.loads(manifest_bytes.decode("utf-8"))


def list_bundled_model_folders(base_path: str | Path) -> list[str]:
    manifest = load_bundle_manifest(base_path)
    folders = {rel_path.split("/", 1)[0] for rel_path in manifest}
    return sorted(folders)


def load_bundled_model(base_path: str | Path, folder_name: str, model_filename: str):
    rel_path = f"{folder_name}/{model_filename}"
    manifest = load_bundle_manifest(base_path)
    info = manifest.get(rel_path)
    if info is None:
        raise FileNotFoundError(f"加密模型包中不存在: {rel_path}")

    bundle_path = get_bundle_path(base_path)
    with bundle_path.open("rb") as f:
        f.seek(info["offset"])
        encrypted_payload = f.read(info["size"])

    compressed_bytes = _cipher().decrypt(encrypted_payload)
    raw_bytes = zlib.decompress(compressed_bytes)
    return joblib.load(BytesIO(raw_bytes))


def _iter_runtime_model_files(source_root: Path, folder_names: list[str]):
    for folder_name in folder_names:
        folder_path = source_root / folder_name
        if not folder_path.is_dir():
            continue
        for model_path in sorted(folder_path.glob("*.joblib")):
            if model_path.name.startswith(RUNTIME_MODEL_PREFIXES):
                yield folder_name, model_path


def build_model_bundle(
    source_root: str | Path,
    output_path: str | Path,
    folder_names: list[str] | None = None,
) -> dict[str, int]:
    source_root = Path(source_root)
    output_path = Path(output_path)

    if folder_names is None:
        folder_names = sorted(
            p.name for p in source_root.iterdir() if p.is_dir() and p.name[:4].isdigit()
        )

    encrypted_blobs: dict[str, bytes] = {}
    for folder_name, model_path in _iter_runtime_model_files(source_root, folder_names):
        rel_path = f"{folder_name}/{model_path.name}"
        compressed = zlib.compress(model_path.read_bytes(), level=9)
        encrypted_blobs[rel_path] = _cipher().encrypt(compressed)

    if not encrypted_blobs:
        raise RuntimeError("没有找到可打包的运行时模型文件。")

    relative_manifest: dict[str, dict[str, int]] = {}
    body_offset = 0
    for rel_path, encrypted_blob in encrypted_blobs.items():
        relative_manifest[rel_path] = {"offset": body_offset, "size": len(encrypted_blob)}
        body_offset += len(encrypted_blob)

    shift = MANIFEST_HEADER_BYTES
    for _ in range(10):
        manifest = {
            rel_path: {"offset": info["offset"] + shift, "size": info["size"]}
            for rel_path, info in relative_manifest.items()
        }
        manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        new_shift = MANIFEST_HEADER_BYTES + len(manifest_bytes)
        if new_shift == shift:
            break
        shift = new_shift
    else:
        raise RuntimeError("模型包清单偏移量收敛失败。")

    output_path.write_bytes(
        len(manifest_bytes).to_bytes(MANIFEST_HEADER_BYTES, "big")
        + manifest_bytes
        + b"".join(encrypted_blobs[rel_path] for rel_path in encrypted_blobs)
    )

    load_bundle_manifest.cache_clear()
    return {
        "folder_count": len(folder_names),
        "model_file_count": len(encrypted_blobs),
        "bundle_size": output_path.stat().st_size,
    }
