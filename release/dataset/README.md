# CICIoMT2024, processed for drift-adaptive federated IDS experiments

The processed input of the project's experiment pipeline. Notebooks A and B in the repository's `notebooks/` folder attach this dataset and check every file against [`SHA256SUMS`](SHA256SUMS) before running.

**Before publishing:** the uploader checks the Canadian Institute for Cybersecurity's terms for redistributing CICIoMT2024 derivatives and sets the licence field to match. This card doesn't choose a licence.

## Files

| file | rows | what it is |
|---|---:|---|
| `wifi_mqtt_full.parquet` | 8,775,013 | All Wi-Fi/MQTT flow records of CICIoMT2024: 45 features, the original label, and integer labels `y_binary` (2), `y_category` (6) and `y_attack19` (19). The original row order is kept: the official train block, then the official test block (from row 7,160,831). |
| `wifi_mqtt_dev.parquet` | 568,146 | A stratified development subsample of the file above: 5% of each class, at least 20,000 rows per class, rare classes whole. |
| `ble.parquet` | 1,547,894 | Bluetooth packet rows, with `UNKNOWN` relabelled to `BLE-DoS` (every such row comes from a DoS capture file). |
| `wifi_profiling.parquet` | 361,081 | Benign Wi-Fi rows from 42 device and scenario captures, with device identities. |
| `data_manifest.json` | | Label maps, feature list, dtypes, column statistics, cleaning and duplicate report, and the raw CSV's SHA-256. |

## How it was built

`scripts/m1_prepare_data.py` in the repository (about 100 s on a laptop):
- **Input:** a merge of CICIoMT2024's 72 Wi-Fi/MQTT attack CSV files, `ciciomt2024_full_8p7m.csv` (SHA-256 `f7cb6fe30c4e4f0cb415a5e888f9bfcc7a652189a094447c72b8428b6b8f0eb6`). The merge was done by hand and isn't scripted.
- **Conversion:** read in chunks and written to Parquet.
  - Features are stored as float32, except where the cast would merge more than 0.1% of a column's distinct values. Those eight columns stay float64: `Rate`, `Srate`, `AVG`, `Std`, `IAT`, `Magnitue`, `Radius`, `Covariance`.
  - Labels are given fixed integer codes, and their consistency is asserted on every chunk.
  - Duplicates (7,379 rows) and the constant column `Drate` are reported, not removed.
- **Check against a second copy:** the merged rows were compared with an independent published copy of CICIoMT2024 ([Zenodo record 19452870](https://zenodo.org/records/19452870)) on all 45 features and every label. Both hold the same 8,767,634 unique rows, and the labels map one-to-one.

`ble.parquet` and `wifi_profiling.parquet` come from that Zenodo record (`bluetooth_clean.parquet` and `wifi_mqtt_clean.parquet`). They carry its CC BY 4.0 licence, which requires attribution.

## Caution

`IAT` behaves like a capture timestamp, not an inter-arrival time. On its own it reaches 0.83 macro-F1, and with it DoS and DDoS become trivially separable. The project excludes it from modelling, and anyone using this data should check for the same shortcut.

## Cite

- Dadkhah, S. et al. CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT. *Internet of Things* 28, 101351 (2024). [doi:10.1016/j.iot.2024.101351](https://doi.org/10.1016/j.iot.2024.101351)
- Engel, V. J. L., Angela, D., Latif, A., Kristyan, S. A. CICIoMT2024 Cleaned Dataset and EDA Outputs. Zenodo (2026), version record [doi:10.5281/zenodo.19452870](https://doi.org/10.5281/zenodo.19452870) (for `ble.parquet` and `wifi_profiling.parquet`).
