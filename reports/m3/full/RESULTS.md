# M3 results — edge ensemble baselines (`full` pool)

Run `20261104-154549_m3_baselines_full`; target `y_category` (6 classes); timeline `category_intro`; 43 features (D6: IAT excluded).
Model: XGBoost + LightGBM with balanced class weights (capped at 50), early stopping on a 10% stratified validation split, fusion weight chosen by validation log-loss, temperature-calibrated.

## E1 centralized vs static, on each window's own test set

| window | new classes | E1 centralized macro-F1 | E1 accuracy | static macro-F1 | static accuracy | E1 n_train | w_xgb | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | Benign, DoS-* | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 332,955 | 0.9 | 0.90 |
| t1 | DDoS-* | 0.8311 | 0.8053 | 0.4626 | 0.2854 | 1,499,308 | 0.4 | 0.87 |
| t2 | Recon-Port-Scan, Recon-Ping-Sweep | 0.8683 | 0.8040 | 0.3369 | 0.2811 | 2,683,809 | 0.9 | 0.83 |
| t3 | MQTT-* | 0.8948 | 0.8186 | 0.2631 | 0.2638 | 3,945,919 | 0.3 | 0.83 |
| t4 | ARP-Spoofing | 0.8827 | 0.8147 | 0.2127 | 0.2625 | 5,214,366 | 0.9 | 0.82 |
| t5 | Recon-OS-Scan, Recon-VulScan | 0.8718 | 0.8175 | 0.2089 | 0.2640 | 6,507,448 | 0.1 | 0.83 |

- *E1 centralized* = retrained at every window on all clients' train data for windows 0..t (the upper bound).
- *Static* = trained on t0 only and never updated, which quantifies drift degradation.
- Window-averaged macro-F1: **E1 0.8915**, **static 0.4140**.
- E1 macro-F1 on the cumulative test set (windows 0..t, row-weighted) per t: t0 0.9999, t1 0.8535, t2 0.8586, t3 0.8714, t4 0.8712, t5 0.8713.

![static vs centralized](static_vs_centralized.png)

![matrix](macro_f1_matrix.png)

Each matrix row is one model (trained through window t) and each column is a window's test set. Cells right of the diagonal are future windows the model hasn't seen.

## Ensemble components (averaged over the diagonal)

| variant | mean macro-F1 | mean accuracy | mean ECE |
|---|---:|---:|---:|
| xgb | 0.8910 | 0.8429 | 0.0245 |
| lgb | 0.8906 | 0.8422 | 0.0226 |
| fused | 0.8915 | 0.8433 | 0.0239 |
| calibrated | 0.8915 | 0.8433 | 0.0172 |

## Per-class F1 of E1 on each window (— = class not yet present)

| window | benign | ddos | dos | recon | mqtt | spoofing |
|---|---:|---:|---:|---:|---:|---:|
| t0 | 1.000 | — | 1.000 | — | — | — |
| t1 | 1.000 | 0.855 | 0.638 | — | — | — |
| t2 | 0.998 | 0.852 | 0.634 | 0.991 | — | — |
| t3 | 0.995 | 0.854 | 0.637 | 0.990 | 0.998 | — |
| t4 | 0.988 | 0.851 | 0.632 | 0.980 | 0.997 | 0.848 |
| t5 | 0.984 | 0.852 | 0.634 | 0.964 | 0.997 | 0.801 |

![confusion](confusion_last_window.png)

## Edge cost

| model | single-flow P50 (ms) | P95 (ms) | batched (µs/flow) | XGB size (KB) | LGBM size (KB) |
|---|---:|---:|---:|---:|---:|
| trained through t0 | 5.99 | 9.49 | 3.26 | 212 | 619 |
| trained through t5 | 6.01 | 8.65 | 74.03 | 27,352 | 9,165 |

Measured on a machine with 4 logical CPUs, using all of them, with the full fused + calibrated ensemble. Single-flow latency includes Python call overhead.

## BLE edge model (binary, separate feature space: packet_length, inter_packet_gap, hci_type, hci_event_code, l2cap_length)

| trained through | eval window | macro-F1 | accuracy | AUROC | ECE |
|---|---|---:|---:|---:|---:|
| t4 | t4 | 0.9931 | 0.9955 | 0.9998 | 0.0048 |
| t4 | t5 | 0.9925 | 0.9951 | 0.9998 | 0.0046 |
| t5 | t4 | 0.9944 | 0.9963 | 0.9998 | 0.0049 |
| t5 | t5 | 0.9936 | 0.9958 | 0.9998 | 0.0050 |
