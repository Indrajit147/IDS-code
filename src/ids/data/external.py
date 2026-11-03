"""Zenodo CICIoMT2024 derivative (10.5281/zenodo.19452870, CC BY 4.0).

- Bluetooth: packet-level BLE records. Zenodo labels the DoS rows 'UNKNOWN';
  every such row comes from a Bluetooth_DoS_*.csv capture, so we relabel to DoS
  and fail loudly if that invariant ever breaks.
- Wi-Fi profiling: benign captures named by device/scenario, used in M2 as real
  device identities for device-based non-IID partitioning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .schema import BLE_FEATURES, CATEGORY_CODE, FEATURES

ZENODO_WIFI_LABEL_TO_ATTACK19 = {
    "Benign": "Benign",
    "DDoS ICMP Flood": "DDoS-ICMP", "DDoS SYN Flood": "DDoS-SYN",
    "DDoS TCP Flood": "DDoS-TCP", "DDoS UDP Flood": "DDoS-UDP",
    "DoS ICMP Flood": "DoS-ICMP", "DoS SYN Flood": "DoS-SYN",
    "DoS TCP Flood": "DoS-TCP", "DoS UDP Flood": "DoS-UDP",
    "Recon OSScan": "Recon-OS-Scan", "Recon PingSweep": "Recon-Ping-Sweep",
    "Recon PortScan": "Recon-Port-Scan", "Recon VulScan": "Recon-VulScan",
    "MQTT DDoS Connect Flood": "MQTT-DDoS-Connect-Flood",
    "MQTT DDoS Publish Flood": "MQTT-DDoS-Publish-Flood",
    "MQTT DoS Connect Flood": "MQTT-DoS-Connect-Flood",
    "MQTT DoS Publish Flood": "MQTT-DoS-Publish-Flood",
    "MQTT Malformed Data": "MQTT-Malformed-Data",
    "ARP Spoofing": "ARP-Spoofing",
}


def process_ble(src: Path, dst: Path) -> dict:
    df = pd.read_parquet(src)
    is_dos_file = df["source_file"].str.contains("DoS", regex=False)
    bad = df[(df["label"] == "UNKNOWN") != is_dos_file]
    if len(bad):
        raise ValueError(f"{len(bad)} BLE rows break the UNKNOWN<->DoS-file invariant")
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    out = df[["timestamp", *BLE_FEATURES, "split", "source_file"]].copy()
    out["y_binary"] = is_dos_file.astype(np.int8)
    out["y_category"] = np.where(is_dos_file, CATEGORY_CODE["dos"], CATEGORY_CODE["benign"]).astype(np.int8)
    out["y_attack"] = np.where(is_dos_file, "BLE-DoS", "Benign")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, compression="zstd", index=False)
    return {
        "source": str(src), "n_rows": int(len(out)),
        "dropped_all_nan_columns": empty_cols,
        "features": BLE_FEATURES,
        "counts_by_label_split": (out.groupby(["y_attack", "split"]).size()
                                  .rename("n").reset_index().to_dict("records")),
        "n_nan_by_feature": {c: int(out[c].isna().sum()) for c in BLE_FEATURES},
    }


def process_wifi_profiling(src: Path, dst: Path) -> dict:
    """Benign device/scenario profiling captures, renamed to the CSV feature names."""
    lower = {f.lower(): f for f in FEATURES}
    parts = []  # stream batches: the full file would need ~3.6 GB in memory
    for b in pq.ParquetFile(src).iter_batches(batch_size=500_000):
        d = b.to_pandas()
        parts.append(d[d["split"] == "profiling"])
    df = pd.concat(parts, ignore_index=True).rename(columns=lower)
    labels = df["label"].unique().tolist()
    if labels != ["Benign"]:
        raise ValueError(f"profiling split expected benign only, got {labels}")
    df["device_scenario"] = df["source_file"].str.replace(".pcap.csv", "", regex=False)
    keep = [*FEATURES, "time_to_live", "device_scenario", "source_file"]
    df[keep].to_parquet(dst, compression="zstd", index=False)
    vc = df["device_scenario"].value_counts()
    return {"source": str(src), "n_rows": int(len(df)), "n_device_scenarios": int(len(vc)),
            "rows_per_device_scenario": vc.to_dict()}


def compare_wifi_with_csv(src: Path, csv_counts: dict[str, int]) -> dict:
    """Per-class row counts: Zenodo train+test vs our CSV (consistency audit)."""
    t = pq.read_table(src, columns=["label", "split"], filters=[("split", "!=", "profiling")])
    z = t.to_pandas()["label"].map(ZENODO_WIFI_LABEL_TO_ATTACK19)
    if z.isna().any():
        raise ValueError("unmapped Zenodo Wi-Fi label")
    zc = z.value_counts().to_dict()
    rows = [{"class": a, "csv": int(csv_counts.get(a, 0)), "zenodo_train_test": int(zc.get(a, 0)),
             "diff": int(zc.get(a, 0)) - int(csv_counts.get(a, 0))} for a in csv_counts]
    return {"per_class": rows, "csv_total": int(sum(csv_counts.values())),
            "zenodo_total": int(sum(zc.values()))}
