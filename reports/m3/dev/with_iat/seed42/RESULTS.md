# M3 results — edge ensemble baselines (`dev` pool)

Run `20261116-111249_m3_baselines_dev`; target `y_category` (6 classes); timeline `category_intro`; 44 features (D6: IAT excluded).
Model: XGBoost + LightGBM with balanced class weights (capped at 50), early stopping on a 10% stratified validation split, fusion weight chosen by validation log-loss, temperature-calibrated.

## E1 centralized vs static, on each window's own test set

| window | new classes | E1 centralized macro-F1 | E1 accuracy | static macro-F1 | static accuracy | E1 n_train | w_xgb | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | Benign, DoS-* | 0.9993 | 0.9994 | 0.9993 | 0.9994 | 22,628 | 0.0 | 0.14 |
| t1 | DDoS-* | 0.9998 | 0.9998 | 0.4624 | 0.3516 | 86,930 | 0.0 | 0.91 |
| t2 | Recon-Port-Scan, Recon-Ping-Sweep | 0.9960 | 0.9984 | 0.3188 | 0.3326 | 154,897 | 0.0 | 0.84 |
| t3 | MQTT-* | 0.9960 | 0.9981 | 0.1688 | 0.2579 | 242,526 | 1.0 | 1.01 |
| t4 | ARP-Spoofing | 0.9877 | 0.9948 | 0.1302 | 0.2406 | 336,493 | 0.5 | 1.03 |
| t5 | Recon-OS-Scan, Recon-VulScan | 0.9810 | 0.9892 | 0.1390 | 0.2625 | 454,648 | 0.7 | 1.05 |

- *E1 centralized* = retrained at every window on all clients' train data for windows 0..t (the upper bound).
- *Static* = trained on t0 only and never updated, which quantifies drift degradation.
- Window-averaged macro-F1: **E1 0.9933**, **static 0.3698**.
- E1 macro-F1 on the cumulative test set (windows 0..t, row-weighted) per t: t0 0.9993, t1 0.9998, t2 0.9979, t3 0.9972, t4 0.9939, t5 0.9893.

![static vs centralized](static_vs_centralized.png)

![matrix](macro_f1_matrix.png)

Each matrix row is one model (trained through window t) and each column is a window's test set. Cells right of the diagonal are future windows the model hasn't seen.

## Ensemble components (averaged over the diagonal)

| variant | mean macro-F1 | mean accuracy | mean ECE |
|---|---:|---:|---:|
| xgb | 0.9933 | 0.9966 | 0.0010 |
| lgb | 0.9934 | 0.9966 | 0.0009 |
| fused | 0.9933 | 0.9966 | 0.0009 |
| calibrated | 0.9933 | 0.9966 | 0.0009 |

## Per-class F1 of E1 on each window (— = class not yet present)

| window | benign | ddos | dos | recon | mqtt | spoofing |
|---|---:|---:|---:|---:|---:|---:|
| t0 | 0.999 | — | 0.999 | — | — | — |
| t1 | 1.000 | 1.000 | 0.999 | — | — | — |
| t2 | 0.996 | 1.000 | 0.999 | 0.990 | — | — |
| t3 | 0.993 | 1.000 | 0.999 | 0.990 | 0.998 | — |
| t4 | 0.991 | 1.000 | 0.999 | 0.974 | 0.997 | 0.965 |
| t5 | 0.989 | 1.000 | 1.000 | 0.976 | 0.993 | 0.928 |

![confusion](confusion_last_window.png)

## Edge cost

| model | single-flow P50 (ms) | P95 (ms) | batched (µs/flow) | XGB size (KB) | LGBM size (KB) |
|---|---:|---:|---:|---:|---:|
| trained through t0 | 5.56 | 7.58 | 3.04 | 195 | 733 |
| trained through t5 | 7.06 | 8.04 | 16.03 | 9,979 | 3,763 |

Measured on a machine with 12 logical CPUs, using all of them, with the full fused + calibrated ensemble. Single-flow latency includes Python call overhead.

## BLE edge model (binary, separate feature space: packet_length, inter_packet_gap, hci_type, hci_event_code, l2cap_length)

| trained through | eval window | macro-F1 | accuracy | AUROC | ECE |
|---|---|---:|---:|---:|---:|
| t4 | t4 | 0.9934 | 0.9947 | 0.9990 | 0.0054 |
| t4 | t5 | 0.9953 | 0.9961 | 0.9992 | 0.0062 |
| t5 | t4 | 0.9918 | 0.9933 | 0.9996 | 0.0054 |
| t5 | t5 | 0.9928 | 0.9941 | 0.9995 | 0.0054 |
