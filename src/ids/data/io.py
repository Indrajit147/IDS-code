"""Loaders for processed data. Constant columns (from the manifest) are excluded by default."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..utils.config import data_path
from .schema import BLE_FEATURES, FEATURES

FILES = {"full": "wifi_mqtt_full.parquet", "dev": "wifi_mqtt_dev.parquet",
         "ble": "ble.parquet", "wifi_profiling": "wifi_profiling.parquet"}
LABELS = ["y_attack19", "y_category", "y_binary"]


def processed_dir(cfg: dict) -> Path:
    """Folder holding data_manifest.json + parquet files.

    On Kaggle the uploaded dataset may nest it anywhere under /kaggle/input, so if
    the configured folder lacks the manifest we look for the manifest itself.
    """
    p = data_path(cfg, "processed")
    if (p / "data_manifest.json").exists() or cfg.get("env") != "kaggle":
        return p
    root = Path(cfg["paths"]["data_root"])
    hits = sorted(root.rglob("data_manifest.json")) if root.exists() else []
    return hits[0].parent if hits else p


def load_manifest(cfg: dict) -> dict:
    return json.loads((processed_dir(cfg) / "data_manifest.json").read_text(encoding="utf-8"))


def feature_columns(manifest: dict, exclude: list[str] | tuple = ()) -> list[str]:
    """Wi-Fi/MQTT model features: all minus constant columns minus `exclude`."""
    drop = set(manifest["cleaning"]["constant_columns"]) | set(exclude)
    return [f for f in FEATURES if f not in drop]


def model_features(cfg: dict, manifest: dict | None = None) -> list[str]:
    """Feature set used for modelling, honouring cfg.features.exclude_wifi (D6)."""
    return feature_columns(manifest or load_manifest(cfg), cfg.get("features", {}).get("exclude_wifi", []))


def ble_features(cfg: dict) -> list[str]:
    drop = set(cfg.get("features", {}).get("exclude_ble", []))
    return [f for f in BLE_FEATURES if f not in drop]


def load(cfg: dict, which: str = "dev", columns: list[str] | None = None) -> pd.DataFrame:
    """Default columns: row_id + model features + labels (Wi-Fi) or the full file (BLE etc.)."""
    path = processed_dir(cfg) / FILES[which]
    if columns is None and which in ("full", "dev"):
        columns = ["row_id", *model_features(cfg), *LABELS]
    # split_blocks + self_destruct free Arrow buffers column by column during
    # conversion instead of holding a full second copy (matters at 8.8M rows).
    return pq.read_table(path, columns=columns).to_pandas(split_blocks=True, self_destruct=True)
