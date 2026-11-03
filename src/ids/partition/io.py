"""Save/load partitions and materialize the rows for a (windows, clients, split) selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..data.io import FILES, LABELS, ble_features, model_features, processed_dir
from ..utils.config import data_path
from ..utils.jsonl import dumps
from .build import SRC_CSV, SRC_PROFILING, TEST, TRAIN, Partition

SPLIT_CODE = {"train": TRAIN, "test": TEST}


def partition_dir(cfg: dict, pool: str, timeline: str) -> Path:
    return data_path(cfg, "splits") / pool / timeline


def save_partition(part: Partition, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    part.assignments.to_parquet(out / "assignments.parquet", index=False)
    part.dpub.to_parquet(out / "dpub.parquet", index=False)
    if part.ble is not None:
        part.ble.to_parquet(out / "ble_assignments.parquet", index=False)
    (out / "window_definition.json").write_text(dumps(part.definition, indent=2), encoding="utf-8")
    (out / "client_partitions.json").write_text(dumps(part.meta, indent=2), encoding="utf-8")


def load_meta(cfg: dict, pool: str, timeline: str) -> dict:
    d = partition_dir(cfg, pool, timeline)
    return {"window_definition": json.loads((d / "window_definition.json").read_text(encoding="utf-8")),
            "client_partitions": json.loads((d / "client_partitions.json").read_text(encoding="utf-8"))}


def read_positions(path: Path, positions: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Rows at the given (sorted, unique) positions, reading one row group at a time."""
    pf = pq.ParquetFile(path)
    tables, start = [], 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        lo, hi = np.searchsorted(positions, [start, start + n])
        if hi > lo:
            tables.append(pf.read_row_group(g, columns=columns).take(positions[lo:hi] - start))
        start += n
    if not tables:
        return pd.DataFrame(columns=columns)
    return pa.concat_tables(tables).to_pandas(split_blocks=True, self_destruct=True)


def _sel(a: pd.DataFrame, windows, clients, split) -> pd.DataFrame:
    m = a["window"].isin(np.atleast_1d(windows))
    if clients is not None:
        m &= a["client"].isin(np.atleast_1d(clients))
    if split is not None:
        m &= a["split"] == SPLIT_CODE[split]
    return a[m]


def load_rows(cfg: dict, pool: str, timeline: str, windows, split: str | None = "train",
              clients=None, features: list[str] | None = None) -> pd.DataFrame:
    """Features + labels + (source, row_id, window, client, split) for the selection.

    Example: load_rows(cfg, "dev", "category_intro", windows=[0, 1], split="test", clients=3)
    """
    feats = features or model_features(cfg)
    a = pd.read_parquet(partition_dir(cfg, pool, timeline) / "assignments.parquet")
    a = _sel(a, windows, clients, split)
    out = []
    csv_sel = a[a["source"] == SRC_CSV]
    if len(csv_sel):
        src = processed_dir(cfg) / FILES["full" if pool == "full" else "dev"]
        cols = ["row_id", *feats, *LABELS]
        if pool == "full":  # row_id == position in the full parquet
            pos = np.sort(csv_sel["row_id"].to_numpy())
            data = read_positions(src, pos, cols)
        else:
            data = pd.read_parquet(src, columns=cols)
        out.append(csv_sel.drop(columns="y_attack19").merge(data, on="row_id", how="left"))
    prof_sel = a[a["source"] == SRC_PROFILING]
    if len(prof_sel):
        pos = np.sort(prof_sel["row_id"].unique())
        data = read_positions(processed_dir(cfg) / FILES["wifi_profiling"], pos, feats)
        data["row_id"] = pos.astype(np.int32)
        for lab in LABELS:
            data[lab] = np.int8(0)  # profiling captures are all benign
        out.append(prof_sel.drop(columns="y_attack19").merge(data, on="row_id", how="left"))
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(df) and df[feats].isna().any().any():
        raise RuntimeError("assignment rows missing from processed data (stale partition?)")
    return df


def load_dpub(cfg: dict, pool: str, timeline: str, upto_window: int | None = None,
              features: list[str] | None = None) -> pd.DataFrame:
    """D_pub rows; with window_mode=cumulative pass upto_window to hide future classes."""
    feats = features or model_features(cfg)
    d = pd.read_parquet(partition_dir(cfg, pool, timeline) / "dpub.parquet")
    if upto_window is not None:
        d = d[d["intro_window"] <= upto_window]
    src = processed_dir(cfg) / FILES["full" if pool == "full" else "dev"]
    cols = ["row_id", *feats, *LABELS]
    if pool == "full":
        data = read_positions(src, np.sort(d["row_id"].to_numpy()), cols)
    else:
        data = pd.read_parquet(src, columns=cols)
    return d.drop(columns="y_attack19").merge(data, on="row_id", how="left")


def load_ble_rows(cfg: dict, pool: str, timeline: str, windows, split: str | None = "train",
                  clients=None) -> pd.DataFrame:
    a = pd.read_parquet(partition_dir(cfg, pool, timeline) / "ble_assignments.parquet")
    a = _sel(a, windows, clients, split)
    ble = pd.read_parquet(processed_dir(cfg) / FILES["ble"], columns=[*ble_features(cfg), "y_category"])
    rows = ble.iloc[a["row_index"].to_numpy()].reset_index(drop=True)
    return pd.concat([a.reset_index(drop=True), rows], axis=1)
