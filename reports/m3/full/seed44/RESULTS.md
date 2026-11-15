# M3 results — edge ensemble baselines (`full` pool)

Run `20261114-192242_m3_baselines_full`; target `y_category` (6 classes); timeline `category_intro`; 43 features (D6: IAT excluded).
Model: XGBoost + LightGBM with balanced class weights (capped at 50), early stopping on a 10% stratified validation split, fusion weight chosen by validation log-loss, temperature-calibrated.

## E1 centralized vs static, on each window's own test set

| window | new classes | E1 centralized macro-F1 | E1 accuracy | static macro-F1 | static accuracy | E1 n_train | w_xgb | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | Benign, DoS-* | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 332,955 | 0.0 | 0.22 |
| t1 | DDoS-* | 0.8308 | 0.8044 | 0.4626 | 0.2855 | 1,499,308 | 0.4 | 0.87 |
| t2 | Recon-Port-Scan, Recon-Ping-Sweep | 0.8690 | 0.8060 | 0.3278 | 0.2811 | 2,683,809 | 0.8 | 0.82 |
| t3 | MQTT-* | 0.8947 | 0.8182 | 0.2561 | 0.2638 | 3,945,919 | 0.2 | 0.84 |
| t4 | ARP-Spoofing | 0.8834 | 0.8157 | 0.2072 | 0.2625 | 5,214,366 | 0.3 | 0.82 |
| t5 | Recon-OS-Scan, Recon-VulScan | 0.8716 | 0.8194 | 0.2009 | 0.2640 | 6,507,448 | 0.3 | 0.82 |

- *E1 centralized* = retrained at every window on all clients' train data for windows 0..t (the upper bound).
- *Static* = trained on t0 only and never updated, which quantifies drift degradation.
- Window-averaged macro-F1: **E1 0.8916**, **static 0.4091**.
- E1 macro-F1 on the cumulative test set (windows 0..t, row-weighted) per t: t0 1.0000, t1 0.8535, t2 0.8592, t3 0.8714, t4 0.8718, t5 0.8714.

![static vs centralized](static_vs_centralized.png)

![matrix](macro_f1_matrix.png)

Each matrix row is one model (trained through window t) and each column is a window's test set. Cells right of the diagonal are future windows the model hasn't seen.

## Ensemble components (averaged over the diagonal)

| variant | mean macro-F1 | mean accuracy | mean ECE |
|---|---:|---:|---:|
| xgb | 0.8908 | 0.8425 | 0.0237 |
| lgb | 0.8915 | 0.8440 | 0.0222 |
| fused | 0.8916 | 0.8439 | 0.0237 |
| calibrated | 0.8916 | 0.8439 | 0.0174 |

## Per-class F1 of E1 on each window (— = class not yet present)

| window | benign | ddos | dos | recon | mqtt | spoofing |
|---|---:|---:|---:|---:|---:|---:|
| t0 | 1.000 | — | 1.000 | — | — | — |
| t1 | 1.000 | 0.854 | 0.638 | — | — | — |
| t2 | 0.997 | 0.853 | 0.635 | 0.990 | — | — |
| t3 | 0.995 | 0.854 | 0.637 | 0.991 | 0.998 | — |
| t4 | 0.989 | 0.852 | 0.633 | 0.980 | 0.997 | 0.849 |
| t5 | 0.983 | 0.854 | 0.635 | 0.963 | 0.997 | 0.798 |

![confusion](confusion_last_window.png)

## Edge cost

| model | single-flow P50 (ms) | P95 (ms) | batched (µs/flow) | XGB size (KB) | LGBM size (KB) |
|---|---:|---:|---:|---:|---:|
| trained through t0 | 6.00 | 8.83 | 11.25 | 295 | 1,071 |
| trained through t5 | 6.03 | 8.90 | 103.93 | 27,018 | 7,927 |

Measured on a machine with 4 logical CPUs, using all of them, with the full fused + calibrated ensemble. Single-flow latency includes Python call overhead.

## BLE edge model (binary, separate feature space: packet_length, inter_packet_gap, hci_type, hci_event_code, l2cap_length)

| trained through | eval window | macro-F1 | accuracy | AUROC | ECE |
|---|---|---:|---:|---:|---:|
| t4 | t4 | 0.9941 | 0.9961 | 0.9998 | 0.0054 |
| t4 | t5 | 0.9932 | 0.9955 | 0.9998 | 0.0053 |
| t5 | t4 | 0.9938 | 0.9959 | 0.9998 | 0.0050 |
| t5 | t5 | 0.9931 | 0.9954 | 0.9998 | 0.0049 |
