"""Memory-safe CSV -> Parquet conversion with auditable statistics.

Pass 1 (stats): per-column min/max/sum/sumsq, NaN/inf counts, label consistency,
                and a seeded 10% sample to decide float32 vs float64 per column.
Pass 2 (write): typed Parquet with fixed integer labels plus row_id, and a
                64-bit hash per row for duplicate / label-conflict analysis.
Nothing is dropped here: constant columns are listed in the manifest and
excluded by the loader, so the conversion stays lossless and reversible.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..utils.seed import rng
from .schema import (ATTACK19_CODE, ATTACK19_TO_CATEGORY_CODE, FEATURES, MAJOR_TO_CATEGORY,
                     CATEGORY_CODE, RAW_LABEL_COLS, attack_category)

CHUNK = 1_000_000


def sha256_file(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(block):
            h.update(b)
    return h.hexdigest()


def _chunks(csv: Path, chunksize: int = CHUNK):
    dtypes = {c: "float64" for c in FEATURES} | {c: "str" for c in RAW_LABEL_COLS}
    yield from pd.read_csv(csv, dtype=dtypes, chunksize=chunksize)


def _check_header(csv: Path) -> None:
    header = pd.read_csv(csv, nrows=0).columns.tolist()
    expected = FEATURES + list(RAW_LABEL_COLS)
    if header != expected:
        raise ValueError(f"unexpected CSV header.\n got: {header}\n expected: {expected}")


def encode_labels(chunk: pd.DataFrame) -> pd.DataFrame:
    sub, major = chunk["Sub_Attack"], chunk["Major_Label"]
    unknown = set(sub.unique()) - set(ATTACK19_CODE)
    if unknown:
        raise ValueError(f"unknown Sub_Attack values: {sorted(unknown)}")
    for a, m in chunk[list(RAW_LABEL_COLS)].drop_duplicates().itertuples(index=False):
        if MAJOR_TO_CATEGORY.get(m) != attack_category(a):
            raise ValueError(f"label mismatch: Sub_Attack={a!r} Major_Label={m!r}")
    y19 = sub.map(ATTACK19_CODE).to_numpy(np.int8)
    ycat = np.asarray(ATTACK19_TO_CATEGORY_CODE, np.int8)[y19]
    return pd.DataFrame({"y_attack19": y19, "y_category": ycat,
                         "y_binary": (ycat != CATEGORY_CODE["benign"]).astype(np.int8)})


def scan_stats(csv: Path, seed: int, sample_frac: float = 0.10) -> tuple[dict, pd.DataFrame]:
    """Pass 1: full-data column stats + a seeded sample for the dtype decision."""
    _check_header(csv)
    n = 0
    mn = np.full(len(FEATURES), np.inf); mx = np.full(len(FEATURES), -np.inf)
    s = np.zeros(len(FEATURES)); ss = np.zeros(len(FEATURES))
    n_nan = np.zeros(len(FEATURES), np.int64); n_inf = np.zeros(len(FEATURES), np.int64)
    counts = np.zeros(len(ATTACK19_CODE), np.int64)
    samples = []
    for i, ch in enumerate(_chunks(csv)):
        X = ch[FEATURES].to_numpy()
        fin = np.isfinite(X)
        n_nan += np.isnan(X).sum(0); n_inf += np.isinf(X).sum(0)
        Xf = np.where(fin, X, np.nan)
        mn = np.fmin(mn, np.nanmin(Xf, 0)); mx = np.fmax(mx, np.nanmax(Xf, 0))
        s += np.nansum(Xf, 0); ss += np.nansum(Xf * Xf, 0)
        lab = encode_labels(ch)
        counts += np.bincount(lab["y_attack19"], minlength=len(counts))
        take = rng(seed, "m1_dtype_sample", i).random(len(ch)) < sample_frac
        samples.append(ch.loc[take, FEATURES])
        n += len(ch)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean**2, 0))
    stats = {
        "n_rows": n,
        "n_cols": len(FEATURES) + len(RAW_LABEL_COLS),
        "class_counts_attack19": dict(zip(ATTACK19_CODE, counts.tolist())),
        "columns": {f: {"min": float(mn[j]), "max": float(mx[j]), "mean": float(mean[j]),
                        "std": float(std[j]), "n_nan": int(n_nan[j]), "n_inf": int(n_inf[j])}
                    for j, f in enumerate(FEATURES)},
    }
    return stats, pd.concat(samples, ignore_index=True)


def choose_dtypes(sample: pd.DataFrame, tol: float = 0.001) -> dict[str, str]:
    """float32 unless the cast merges >tol of a column's distinct values (precision loss)."""
    out = {}
    for c in sample.columns:
        x = sample[c].to_numpy()
        u64 = len(np.unique(x))
        u32 = len(np.unique(x.astype(np.float32)))
        out[c] = "float64" if u32 < (1 - tol) * u64 else "float32"
    return out


def write_parquet(csv: Path, out: Path, dtypes: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Pass 2: write typed Parquet; return per-row feature hashes and y_attack19."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    writer = None
    hashes, labels = [], []
    offset = 0
    try:
        for ch in _chunks(csv):
            lab = encode_labels(ch)
            hashes.append(pd.util.hash_pandas_object(ch[FEATURES], index=False).to_numpy())
            labels.append(lab["y_attack19"].to_numpy())
            df = ch[FEATURES].astype(dtypes)
            df.insert(0, "row_id", np.arange(offset, offset + len(ch), dtype=np.int32))
            df = pd.concat([df.reset_index(drop=True), lab], axis=1)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=CHUNK)
            offset += len(ch)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    return np.concatenate(hashes), np.concatenate(labels)


def duplicate_report(hashes: np.ndarray, y: np.ndarray) -> dict:
    """Exact duplicate rows (by feature hash) and duplicates with conflicting labels."""
    df = pd.DataFrame({"h": hashes, "y": y})
    g = df.groupby("h")["y"]
    size, nlab = g.size(), g.nunique()
    dup_groups = size[size > 1]
    conflict = nlab[nlab > 1]
    return {
        "n_rows": int(len(df)),
        "n_unique_feature_rows": int(len(size)),
        "n_duplicate_rows": int((size - 1).clip(lower=0).sum()),
        "n_duplicate_groups": int(len(dup_groups)),
        "n_conflicting_label_groups": int(len(conflict)),
        "n_rows_in_conflicting_groups": int(size[conflict.index].sum()),
        "duplicate_rows_by_class": (df[df.duplicated("h", keep="first")]["y"]
                                    .value_counts().sort_index().to_dict()),
    }
