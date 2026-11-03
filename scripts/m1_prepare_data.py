"""M1: ingest, clean, label-harmonize CICIoMT2024 (+ Zenodo BLE / Wi-Fi profiling).

Outputs (data/processed/):
  wifi_mqtt_full.parquet   all 8.78M rows, row_id + 45 features + y_attack19/y_category/y_binary
  wifi_mqtt_dev.parquet    stratified dev subsample (rare classes kept whole)
  ble.parquet              BLE packet records, DoS relabelled from 'UNKNOWN'
  wifi_profiling.parquet   benign device/scenario captures (device identities for M2)
  data_manifest.json       hashes, counts, dtypes, constant columns, duplicates, sampling weights
A copy of the manifest goes to reports/m1/ (tracked in git).

Usage: python scripts/m1_prepare_data.py [--env local|kaggle] [--force] [--set key=value ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ids.data import external, ingest, sample
from ids.data.io import FILES, processed_dir
from ids.data.schema import ATTACKS19, FEATURES, label_maps
from ids.utils import Run, data_path, load_config, set_global_seed
from ids.utils.config import REPO_ROOT
from ids.utils.jsonl import dumps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild full parquet even if present")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(env=args.env, overrides=args.set)
    seed = cfg["seed"]
    set_global_seed(seed)
    raw = data_path(cfg, "raw_csv")
    zen = data_path(cfg, "zenodo_dir")
    out = processed_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    full_pq = out / FILES["full"]

    with Run(cfg, "m1_prepare_data") as run:
        log, t0 = run.logger, time.time()
        manifest: dict = {"label_maps": label_maps(), "features_all": FEATURES}

        log.info("sha256 of raw CSV ...")
        manifest["raw"] = {"path": str(raw), "bytes": raw.stat().st_size,
                           "sha256": ingest.sha256_file(raw)}

        log.info("pass 1: column stats + dtype sample ...")
        stats, dsample = ingest.scan_stats(raw, seed)
        dtypes = ingest.choose_dtypes(dsample)
        del dsample
        manifest["raw"].update(n_rows=stats["n_rows"], n_cols=stats["n_cols"])
        manifest["class_counts_attack19"] = stats["class_counts_attack19"]
        manifest["column_stats"] = stats["columns"]
        manifest["dtypes"] = dtypes
        const = [c for c, s in stats["columns"].items() if s["min"] == s["max"]]
        manifest["cleaning"] = {
            "constant_columns": const,
            "n_nan_total": sum(s["n_nan"] for s in stats["columns"].values()),
            "n_inf_total": sum(s["n_inf"] for s in stats["columns"].values()),
            "policy": "lossless parquet; constant columns excluded by loader; duplicates reported, not dropped",
        }
        log.info("constant=%s float64=%s (%.0fs)", const,
                 [c for c, d in dtypes.items() if d == "float64"], time.time() - t0)

        if full_pq.exists() and not args.force:
            log.info("full parquet exists, skipping pass 2 (use --force to rebuild)")
            rebuilt = False
        else:
            log.info("pass 2: writing %s ...", full_pq)
            hashes, y19 = ingest.write_parquet(raw, full_pq, dtypes)
            rebuilt = True
            dup = ingest.duplicate_report(hashes, y19)
            dup["duplicate_rows_by_class"] = {ATTACKS19[k]: v for k, v in dup["duplicate_rows_by_class"].items()}
            manifest["duplicates"] = dup
            del hashes, y19
            log.info("duplicates: %s rows, %s conflicting-label groups (%.0fs)",
                     dup["n_duplicate_rows"], dup["n_conflicting_label_groups"], time.time() - t0)
        manifest["full_parquet"] = {"path": str(full_pq), "bytes": full_pq.stat().st_size,
                                    "n_rows": pq.ParquetFile(full_pq).metadata.num_rows}

        log.info("dev sample ...")
        y = pq.read_table(full_pq, columns=["y_attack19"])["y_attack19"].to_numpy()
        frac, floor = cfg["resources"]["dev_sample_frac"], cfg["resources"]["dev_sample_floor"]
        pos, info = sample.stratified_indices(y, frac, floor, seed)
        n_dev = sample.take_rows(full_pq, out / FILES["dev"], pos)
        manifest["dev_sample"] = {"frac": frac, "floor": floor, "n_rows": n_dev,
                                  "per_class": {ATTACKS19[c]: v for c, v in info.items()}}
        del y, pos

        log.info("Zenodo: BLE + Wi-Fi profiling ...")
        manifest["ble"] = external.process_ble(zen / "bluetooth_clean.parquet", out / FILES["ble"])
        manifest["wifi_profiling"] = external.process_wifi_profiling(
            zen / "wifi_mqtt_clean.parquet", out / FILES["wifi_profiling"])
        manifest["zenodo_vs_csv"] = external.compare_wifi_with_csv(
            zen / "wifi_mqtt_clean.parquet", manifest["class_counts_attack19"])

        man_path = out / "data_manifest.json"
        if not rebuilt and man_path.exists():  # keep duplicate stats from the earlier build
            old = json.loads(man_path.read_text(encoding="utf-8"))
            if "duplicates" in old:
                manifest["duplicates"] = old["duplicates"]
        man_path.write_text(dumps(manifest, indent=2), encoding="utf-8")
        rep = REPO_ROOT / "reports" / "m1"
        rep.mkdir(parents=True, exist_ok=True)
        shutil.copy(man_path, rep / "data_manifest.json")
        run.save_json("data_manifest.json", manifest)
        run.log("m1_done", seconds=round(time.time() - t0, 1), n_dev=n_dev)
        log.info("M1 data prep done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
