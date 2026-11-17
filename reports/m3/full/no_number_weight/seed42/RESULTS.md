# M3 results — edge ensemble baselines (`full` pool)

Run `20261117-024811_m3_baselines_full`; target `y_category` (6 classes); timeline `category_intro`; 41 features (D6: IAT excluded).
Model: XGBoost + LightGBM with balanced class weights (capped at 50), early stopping on a 10% stratified validation split, fusion weight chosen by validation log-loss, temperature-calibrated.

## E1 centralized vs static, on each window's own test set

| window | new classes | E1 centralized macro-F1 | E1 accuracy | static macro-F1 | static accuracy | E1 n_train | w_xgb | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | Benign, DoS-* | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 332,955 | 1.0 | 1.00 |
| t1 | DDoS-* | 0.8315 | 0.8059 | 0.4624 | 0.2854 | 1,499,308 | 0.3 | 0.87 |
| t2 | Recon-Port-Scan, Recon-Ping-Sweep | 0.8690 | 0.8057 | 0.3394 | 0.2811 | 2,683,809 | 0.3 | 0.83 |
| t3 | MQTT-* | 0.8951 | 0.8193 | 0.2358 | 0.2638 | 3,945,919 | 0.2 | 0.84 |
| t4 | ARP-Spoofing | 0.8823 | 0.8143 | 0.1916 | 0.2625 | 5,214,366 | 0.7 | 0.81 |
| t5 | Recon-OS-Scan, Recon-VulScan | 0.8707 | 0.8179 | 0.1916 | 0.2640 | 6,507,448 | 0.6 | 0.81 |

- *E1 centralized* = retrained at every window on all clients' train data for windows 0..t (the upper bound).
- *Static* = trained on t0 only and never updated, which quantifies drift degradation.
- Window-averaged macro-F1: **E1 0.8914**, **static 0.4034**.
- E1 macro-F1 on the cumulative test set (windows 0..t, row-weighted) per t: t0 0.9999, t1 0.8537, t2 0.8595, t3 0.8717, t4 0.8709, t5 0.8708.

![static vs centralized](static_vs_centralized.png)

![matrix](macro_f1_matrix.png)

Each matrix row is one model (trained through window t) and each column is a window's test set. Cells right of the diagonal are future windows the model hasn't seen.

## Ensemble components (averaged over the diagonal)

| variant | mean macro-F1 | mean accuracy | mean ECE |
|---|---:|---:|---:|
| xgb | 0.8909 | 0.8431 | 0.0252 |
| lgb | 0.8903 | 0.8426 | 0.0227 |
| fused | 0.8914 | 0.8438 | 0.0233 |
| calibrated | 0.8914 | 0.8438 | 0.0170 |

## Per-class F1 of E1 on each window (— = class not yet present)

| window | benign | ddos | dos | recon | mqtt | spoofing |
|---|---:|---:|---:|---:|---:|---:|
| t0 | 1.000 | — | 1.000 | — | — | — |
| t1 | 1.000 | 0.856 | 0.639 | — | — | — |
| t2 | 0.997 | 0.853 | 0.636 | 0.990 | — | — |
| t3 | 0.995 | 0.855 | 0.638 | 0.990 | 0.998 | — |
| t4 | 0.988 | 0.851 | 0.631 | 0.980 | 0.997 | 0.847 |
| t5 | 0.983 | 0.852 | 0.633 | 0.963 | 0.997 | 0.796 |

![confusion](confusion_last_window.png)

## Edge cost

| model | single-flow P50 (ms) | P95 (ms) | batched (µs/flow) | XGB size (KB) | LGBM size (KB) |
|---|---:|---:|---:|---:|---:|
| trained through t0 | 6.07 | 12.12 | 9.10 | 609 | 675 |
| trained through t5 | 6.37 | 18.13 | 97.52 | 27,633 | 6,602 |

Measured on a machine with 4 logical CPUs, using all of them, with the full fused + calibrated ensemble. Single-flow latency includes Python call overhead.

## BLE edge model (binary, separate feature space: packet_length, inter_packet_gap, hci_type, hci_event_code, l2cap_length)

| trained through | eval window | macro-F1 | accuracy | AUROC | ECE |
|---|---|---:|---:|---:|---:|
| t4 | t4 | 0.9931 | 0.9955 | 0.9998 | 0.0048 |
| t4 | t5 | 0.9925 | 0.9951 | 0.9998 | 0.0046 |
| t5 | t4 | 0.9944 | 0.9963 | 0.9998 | 0.0049 |
| t5 | t5 | 0.9936 | 0.9958 | 0.9998 | 0.0050 |
