"""Stratified development subsample that keeps rare classes whole."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..utils.seed import rng


def stratified_indices(y: np.ndarray, frac: float, floor: int, seed: int) -> tuple[np.ndarray, dict]:
    """Per class: keep all rows if n <= floor, else max(floor, frac*n) rows at random.

    Returns sorted row positions and per-class {n_full, n_sample, weight}, where
    weight = n_full / n_sample restores the full-data class priors if needed.
    """
    picks, info = [], {}
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = len(idx) if len(idx) <= floor else max(floor, int(round(frac * len(idx))))
        chosen = idx if k == len(idx) else rng(seed, "dev_sample", int(c)).choice(idx, k, replace=False)
        picks.append(chosen)
        info[int(c)] = {"n_full": int(len(idx)), "n_sample": int(k), "weight": len(idx) / k}
    return np.sort(np.concatenate(picks)), info


def take_rows(src: Path, dst: Path, positions: np.ndarray) -> int:
    """Copy the given row positions from src to dst, one row group at a time."""
    pf = pq.ParquetFile(src)
    writer, start, written = None, 0, 0
    try:
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            lo, hi = np.searchsorted(positions, [start, start + n])
            if hi > lo:
                t = pf.read_row_group(g).take(positions[lo:hi] - start)
                if writer is None:
                    writer = pq.ParquetWriter(dst, t.schema, compression="zstd")
                writer.write_table(t)
                written += t.num_rows
            start += n
    finally:
        if writer is not None:
            writer.close()
    return written
