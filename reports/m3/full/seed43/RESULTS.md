# M3 results — edge ensemble baselines (`full` pool)

Run `20261114-192216_m3_baselines_full`; target `y_category` (6 classes); timeline `category_intro`; 43 features (D6: IAT excluded).
Model: XGBoost + LightGBM with balanced class weights (capped at 50), early stopping on a 10% stratified validation split, fusion weight chosen by validation log-loss, temperature-calibrated.

## E1 centralized vs static, on each window's own test set

| window | new classes | E1 centralized macro-F1 | E1 accuracy | static macro-F1 | static accuracy | E1 n_train | w_xgb | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | Benign, DoS-* | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 332,955 | 0.0 | 0.22 |
| t1 | DDoS-* | 0.8313 | 0.8054 | 0.4626 | 0.2855 | 1,499,308 | 0.4 | 0.88 |
| t2 | Recon-Port-Scan, Recon-Ping-Sweep | 0.8689 | 0.8053 | 0.3278 | 0.2811 | 2,683,809 | 0.6 | 0.82 |
| t3 | MQTT-* | 0.8946 | 0.8179 | 0.2562 | 0.2638 | 3,945,919 | 0.3 | 0.84 |
| t4 | ARP-Spoofing | 0.8819 | 0.8167 | 0.2072 | 0.2625 | 5,214,366 | 0.3 | 0.82 |
| t5 | Recon-OS-Scan, Recon-VulScan | 0.8722 | 0.8179 | 0.2010 | 0.2640 | 6,507,448 | 0.1 | 0.83 |

- *E1 centralized* = retrained at every window on all clients' train data for windows 0..t (the upper bound).
- *Static* = trained on t0 only and never updated, which quantifies drift degradation.
- Window-averaged macro-F1: **E1 0.8915**, **static 0.4091**.
- E1 macro-F1 on the cumulative test set (windows 0..t, row-weighted) per t: t0 1.0000, t1 0.8536, t2 0.8591, t3 0.8713, t4 0.8715, t5 0.8717.

![static vs centralized](static_vs_centralized.png)

![matrix](macro_f1_matrix.png)

Each matrix row is one model (trained through window t) and each column is a window's test set. Cells right of the diagonal are future windows the model hasn't seen.

## Ensemble components (averaged over the diagonal)

| variant | mean macro-F1 | mean accuracy | mean ECE |
|---|---:|---:|---:|
| xgb | 0.8908 | 0.8433 | 0.0249 |
| lgb | 0.8908 | 0.8430 | 0.0229 |
| fused | 0.8915 | 0.8439 | 0.0239 |
| calibrated | 0.8915 | 0.8439 | 0.0164 |

## Per-class F1 of E1 on each window (— = class not yet present)

| window | benign | ddos | dos | recon | mqtt | spoofing |
|---|---:|---:|---:|---:|---:|---:|
| t0 | 1.000 | — | 1.000 | — | — | — |
| t1 | 1.000 | 0.855 | 0.639 | — | — | — |
| t2 | 0.997 | 0.853 | 0.635 | 0.991 | — | — |
| t3 | 0.995 | 0.854 | 0.637 | 0.990 | 0.998 | — |
| t4 | 0.988 | 0.853 | 0.634 | 0.978 | 0.997 | 0.841 |
| t5 | 0.984 | 0.852 | 0.634 | 0.964 | 0.997 | 0.802 |

![confusion](confusion_last_window.png)

## Edge cost

| model | single-flow P50 (ms) | P95 (ms) | batched (µs/flow) | XGB size (KB) | LGBM size (KB) |
|---|---:|---:|---:|---:|---:|
| trained through t0 | 6.00 | 9.08 | 10.99 | 274 | 1,085 |
| trained through t5 | 5.98 | 8.65 | 113.61 | 27,189 | 9,293 |

Measured on a machine with 4 logical CPUs, using all of them, with the full fused + calibrated ensemble. Single-flow latency includes Python call overhead.

## BLE edge model (binary, separate feature space: packet_length, inter_packet_gap, hci_type, hci_event_code, l2cap_length)

| trained through | eval window | macro-F1 | accuracy | AUROC | ECE |
|---|---|---:|---:|---:|---:|
| t4 | t4 | 0.9938 | 0.9959 | 0.9998 | 0.0051 |
| t4 | t5 | 0.9930 | 0.9954 | 0.9998 | 0.0050 |
| t5 | t4 | 0.9938 | 0.9959 | 0.9998 | 0.0052 |
| t5 | t5 | 0.9931 | 0.9954 | 0.9998 | 0.0048 |
