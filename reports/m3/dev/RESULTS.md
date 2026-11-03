# M3 results — edge ensemble baselines (`dev` pool)

Run `20261104-013856_m3_baselines_dev`; target `y_category` (6 classes); timeline `category_intro`; 43 features (D6: IAT excluded).
Model: XGBoost + LightGBM with balanced class weights (capped at 50), early stopping on a 10% stratified validation split, fusion weight chosen by validation log-loss, temperature-calibrated.

## E1 centralized vs static, on each window's own test set

| window | new classes | E1 centralized macro-F1 | E1 accuracy | static macro-F1 | static accuracy | E1 n_train | w_xgb | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | Benign, DoS-* | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 22,399 | 0.0 | 0.19 |
| t1 | DDoS-* | 0.8203 | 0.8126 | 0.4613 | 0.3495 | 86,463 | 0.5 | 0.90 |
| t2 | Recon-Port-Scan, Recon-Ping-Sweep | 0.8621 | 0.8223 | 0.3252 | 0.3312 | 154,048 | 0.7 | 0.81 |
| t3 | MQTT-* | 0.8875 | 0.8614 | 0.2310 | 0.2581 | 240,776 | 0.6 | 0.83 |
| t4 | ARP-Spoofing | 0.8901 | 0.8629 | 0.1621 | 0.2411 | 333,586 | 0.3 | 0.86 |
| t5 | Recon-OS-Scan, Recon-VulScan | 0.8834 | 0.8837 | 0.1597 | 0.2645 | 449,958 | 0.3 | 0.89 |

- *E1 centralized* = retrained at every window on all clients' train data for windows 0..t (the upper bound).
- *Static* = trained on t0 only and never updated, which quantifies drift degradation.
- Window-averaged macro-F1: **E1 0.8905**, **static 0.3899**.
- E1 macro-F1 on the cumulative test set (windows 0..t, row-weighted) per t: t0 1.0000, t1 0.8482, t2 0.8535, t3 0.8662, t4 0.8702, t5 0.8738.

![static vs centralized](static_vs_centralized.png)

![matrix](macro_f1_matrix.png)

Each matrix row is one model (trained through window t) and each column is a window's test set. Cells right of the diagonal are future windows the model hasn't seen.

## Ensemble components (averaged over the diagonal)

| variant | mean macro-F1 | mean accuracy | mean ECE |
|---|---:|---:|---:|
| xgb | 0.8901 | 0.8733 | 0.0163 |
| lgb | 0.8897 | 0.8727 | 0.0183 |
| fused | 0.8905 | 0.8738 | 0.0187 |
| calibrated | 0.8905 | 0.8738 | 0.0140 |

## Per-class F1 of E1 on each window (— = class not yet present)

| window | benign | ddos | dos | recon | mqtt | spoofing |
|---|---:|---:|---:|---:|---:|---:|
| t0 | 1.000 | — | 1.000 | — | — | — |
| t1 | 1.000 | 0.846 | 0.615 | — | — | — |
| t2 | 0.996 | 0.847 | 0.617 | 0.988 | — | — |
| t3 | 0.987 | 0.849 | 0.625 | 0.980 | 0.995 | — |
| t4 | 0.983 | 0.848 | 0.623 | 0.962 | 0.991 | 0.934 |
| t5 | 0.984 | 0.852 | 0.625 | 0.962 | 0.989 | 0.889 |

![confusion](confusion_last_window.png)

## Edge cost

| model | single-flow P50 (ms) | P95 (ms) | batched (µs/flow) | XGB size (KB) | LGBM size (KB) |
|---|---:|---:|---:|---:|---:|
| trained through t0 | 8.25 | 9.96 | 3.20 | 150 | 745 |
| trained through t5 | 9.30 | 10.73 | 24.98 | 19,072 | 7,219 |

Measured on this machine's CPU (12 threads) with the full fused + calibrated ensemble. Single-flow latency includes Python call overhead.

## BLE edge model (binary, separate feature space: packet_length, inter_packet_gap, hci_type, hci_event_code, l2cap_length)

| trained through | eval window | macro-F1 | accuracy | AUROC | ECE |
|---|---|---:|---:|---:|---:|
| t4 | t4 | 0.9934 | 0.9947 | 0.9990 | 0.0055 |
| t4 | t5 | 0.9951 | 0.9960 | 0.9992 | 0.0063 |
| t5 | t4 | 0.9918 | 0.9933 | 0.9996 | 0.0047 |
| t5 | t5 | 0.9926 | 0.9940 | 0.9995 | 0.0047 |
