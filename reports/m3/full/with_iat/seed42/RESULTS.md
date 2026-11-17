# M3 results — edge ensemble baselines (`full` pool)

Run `20261117-024744_m3_baselines_full`; target `y_category` (6 classes); timeline `category_intro`; 44 features (D6: IAT excluded).
Model: XGBoost + LightGBM with balanced class weights (capped at 50), early stopping on a 10% stratified validation split, fusion weight chosen by validation log-loss, temperature-calibrated.

## E1 centralized vs static, on each window's own test set

| window | new classes | E1 centralized macro-F1 | E1 accuracy | static macro-F1 | static accuracy | E1 n_train | w_xgb | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | Benign, DoS-* | 0.9999 | 1.0000 | 0.9999 | 1.0000 | 332,955 | 1.0 | 1.04 |
| t1 | DDoS-* | 0.9997 | 0.9997 | 0.4626 | 0.2855 | 1,499,308 | 0.5 | 0.99 |
| t2 | Recon-Port-Scan, Recon-Ping-Sweep | 0.9972 | 0.9995 | 0.3367 | 0.2811 | 2,683,809 | 0.5 | 0.70 |
| t3 | MQTT-* | 0.9972 | 0.9992 | 0.2614 | 0.2638 | 3,945,919 | 0.2 | 0.87 |
| t4 | ARP-Spoofing | 0.9803 | 0.9985 | 0.2114 | 0.2625 | 5,214,366 | 0.2 | 0.92 |
| t5 | Recon-OS-Scan, Recon-VulScan | 0.9729 | 0.9978 | 0.2076 | 0.2640 | 6,507,448 | 0.6 | 0.89 |

- *E1 centralized* = retrained at every window on all clients' train data for windows 0..t (the upper bound).
- *Static* = trained on t0 only and never updated, which quantifies drift degradation.
- Window-averaged macro-F1: **E1 0.9912**, **static 0.4133**.
- E1 macro-F1 on the cumulative test set (windows 0..t, row-weighted) per t: t0 0.9999, t1 0.9998, t2 0.9985, t3 0.9981, t4 0.9930, t5 0.9885.

![static vs centralized](static_vs_centralized.png)

![matrix](macro_f1_matrix.png)

Each matrix row is one model (trained through window t) and each column is a window's test set. Cells right of the diagonal are future windows the model hasn't seen.

## Ensemble components (averaged over the diagonal)

| variant | mean macro-F1 | mean accuracy | mean ECE |
|---|---:|---:|---:|
| xgb | 0.9911 | 0.9991 | 0.0003 |
| lgb | 0.9909 | 0.9991 | 0.0003 |
| fused | 0.9912 | 0.9991 | 0.0003 |
| calibrated | 0.9912 | 0.9991 | 0.0002 |

## Per-class F1 of E1 on each window (— = class not yet present)

| window | benign | ddos | dos | recon | mqtt | spoofing |
|---|---:|---:|---:|---:|---:|---:|
| t0 | 1.000 | — | 1.000 | — | — | — |
| t1 | 1.000 | 1.000 | 0.999 | — | — | — |
| t2 | 0.998 | 1.000 | 0.999 | 0.992 | — | — |
| t3 | 0.997 | 1.000 | 0.999 | 0.992 | 0.999 | — |
| t4 | 0.993 | 1.000 | 0.999 | 0.988 | 0.998 | 0.904 |
| t5 | 0.991 | 1.000 | 0.999 | 0.981 | 0.998 | 0.868 |

![confusion](confusion_last_window.png)

## Edge cost

| model | single-flow P50 (ms) | P95 (ms) | batched (µs/flow) | XGB size (KB) | LGBM size (KB) |
|---|---:|---:|---:|---:|---:|
| trained through t0 | 6.26 | 12.12 | 6.47 | 232 | 655 |
| trained through t5 | 6.02 | 18.01 | 40.23 | 11,951 | 3,892 |

Measured on a machine with 4 logical CPUs, using all of them, with the full fused + calibrated ensemble. Single-flow latency includes Python call overhead.

## BLE edge model (binary, separate feature space: packet_length, inter_packet_gap, hci_type, hci_event_code, l2cap_length)

| trained through | eval window | macro-F1 | accuracy | AUROC | ECE |
|---|---|---:|---:|---:|---:|
| t4 | t4 | 0.9931 | 0.9955 | 0.9998 | 0.0048 |
| t4 | t5 | 0.9925 | 0.9951 | 0.9998 | 0.0046 |
| t5 | t4 | 0.9944 | 0.9963 | 0.9998 | 0.0049 |
| t5 | t5 | 0.9936 | 0.9958 | 0.9998 | 0.0050 |
