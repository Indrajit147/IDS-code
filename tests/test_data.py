import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from ids.data import external, ingest, sample
from ids.data.schema import ATTACK19_CODE, ATTACKS19, CATEGORY_CODE, FEATURES


def _toy_csv(path, n=500, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame(r.random((n, len(FEATURES))), columns=FEATURES)
    df["Drate"] = 0.0                                   # constant column
    df["IAT"] = 1.6e8 + r.random(n) * 10                # float32 would merge these values
    subs = r.choice(["Benign", "DDoS-UDP", "MQTT-Malformed-Data", "ARP-Spoofing"], n)
    major = {"Benign": "Benign", "DDoS-UDP": "DDoS", "MQTT-Malformed-Data": "MQTT",
             "ARP-Spoofing": "Spoofing"}
    df["Sub_Attack"] = subs
    df["Major_Label"] = [major[s] for s in subs]
    df.iloc[1] = df.iloc[0]                              # one exact duplicate (same label)
    df.to_csv(path, index=False)
    return df


def test_label_schema_consistent():
    assert len(ATTACKS19) == 19 and ATTACK19_CODE["Benign"] == 0
    assert len(FEATURES) == 45


def test_encode_labels_and_mismatch():
    ch = pd.DataFrame({"Sub_Attack": ["Benign", "DoS-SYN", "Recon-VulScan", "MQTT-DoS-Connect-Flood"],
                       "Major_Label": ["Benign", "DoS", "Recon", "MQTT"]})
    lab = ingest.encode_labels(ch)
    assert lab["y_category"].tolist() == [CATEGORY_CODE[c] for c in ["benign", "dos", "recon", "mqtt"]]
    assert lab["y_binary"].tolist() == [0, 1, 1, 1]
    with pytest.raises(ValueError):
        ingest.encode_labels(pd.DataFrame({"Sub_Attack": ["DoS-SYN"], "Major_Label": ["DDoS"]}))
    with pytest.raises(ValueError):
        ingest.encode_labels(pd.DataFrame({"Sub_Attack": ["BLE-DoS"], "Major_Label": ["DoS"]}))


def test_csv_to_parquet_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHUNK", 128)            # force several chunks / row groups
    csv = tmp_path / "toy.csv"
    df = _toy_csv(csv)
    stats, smp = ingest.scan_stats(csv, seed=1, sample_frac=1.0)
    assert stats["n_rows"] == len(df)
    assert stats["columns"]["Drate"]["min"] == stats["columns"]["Drate"]["max"] == 0.0
    dtypes = ingest.choose_dtypes(smp)
    assert dtypes["IAT"] == "float64" and dtypes["Rate"] == "float32"

    out = tmp_path / "toy.parquet"
    hashes, y = ingest.write_parquet(csv, out, dtypes)
    t = pq.read_table(out).to_pandas()
    assert t["row_id"].tolist() == list(range(len(df)))
    assert pq.ParquetFile(out).metadata.num_row_groups > 1
    np.testing.assert_allclose(t["Rate"], df["Rate"], rtol=1e-6)
    # float64 kept: exact up to the CSV text parser's last-ulp rounding (float32 would be ~1e-8)
    np.testing.assert_allclose(t["IAT"], df["IAT"], rtol=1e-15)
    assert (t["y_attack19"].map(dict(enumerate(ATTACKS19))) == df["Sub_Attack"]).all()

    dup = ingest.duplicate_report(hashes, y)
    assert dup["n_duplicate_rows"] == 1 and dup["n_conflicting_label_groups"] == 0
    dup2 = ingest.duplicate_report(np.array([5, 5, 7], np.uint64), np.array([0, 3, 3]))
    assert dup2["n_conflicting_label_groups"] == 1 and dup2["n_rows_in_conflicting_groups"] == 2


def test_stratified_indices_floor_and_weights(tmp_path):
    y = np.array([0] * 1000 + [1] * 50 + [2] * 5000)
    pos, info = sample.stratified_indices(y, frac=0.1, floor=100, seed=3)
    assert info[1]["n_sample"] == 50                     # rare class kept whole
    assert info[0]["n_sample"] == 100                    # floor dominates 10%
    assert info[2]["n_sample"] == 500                    # 10% dominates floor
    assert info[2]["weight"] == pytest.approx(10.0)
    assert np.all(np.diff(pos) > 0)
    pos2, _ = sample.stratified_indices(y, 0.1, 100, seed=3)
    np.testing.assert_array_equal(pos, pos2)             # deterministic


def test_take_rows(tmp_path):
    import pyarrow as pa
    src, dst = tmp_path / "s.parquet", tmp_path / "d.parquet"
    w = pq.ParquetWriter(src, pa.schema([("a", pa.int64())]))
    for g in range(4):
        w.write_table(pa.table({"a": np.arange(g * 10, g * 10 + 10)}))
    w.close()
    n = sample.take_rows(src, dst, np.array([0, 9, 10, 25, 39]))
    assert n == 5 and pq.read_table(dst)["a"].to_pylist() == [0, 9, 10, 25, 39]


def test_ble_relabel(tmp_path):
    src = tmp_path / "bt.parquet"
    df = pd.DataFrame({"timestamp": [1.0, 2.0, 3.0], "packet_length": [8, 9, 10],
                       "inter_packet_gap": [0.0, 0.1, 0.2], "hci_type": [1, 4, 2],
                       "hci_event_code": [19.0, 14.0, np.nan], "l2cap_length": [27.0, 27.0, 5.0],
                       "ble_access_addr": [np.nan] * 3, "ble_pdu_type": [np.nan] * 3,
                       "cumulative_packet_count": [1, 2, 3],
                       "label": ["Benign", "UNKNOWN", "UNKNOWN"],
                       "source_file": ["Bluetooth_Benign_train.csv", "Bluetooth_DoS_train.csv",
                                       "Bluetooth_DoS_test.csv"],
                       "protocol": ["Bluetooth"] * 3, "split": ["train", "train", "test"]})
    df.to_parquet(src)
    rep = external.process_ble(src, tmp_path / "ble.parquet")
    out = pd.read_parquet(tmp_path / "ble.parquet")
    assert out["y_attack"].tolist() == ["Benign", "BLE-DoS", "BLE-DoS"]
    assert out["y_category"].tolist() == [CATEGORY_CODE["benign"], CATEGORY_CODE["dos"], CATEGORY_CODE["dos"]]
    assert set(rep["dropped_all_nan_columns"]) == {"ble_access_addr", "ble_pdu_type"}
    df.loc[0, "label"] = "UNKNOWN"                       # UNKNOWN outside a DoS file -> must fail
    df.to_parquet(src)
    with pytest.raises(ValueError):
        external.process_ble(src, tmp_path / "ble2.parquet")


def test_feature_exclusion():
    from ids.data.io import ble_features, feature_columns, model_features
    from ids.utils import load_config
    man = {"cleaning": {"constant_columns": ["Drate"]}}
    assert "Drate" not in feature_columns(man) and len(feature_columns(man)) == 44
    cfg = load_config(env="local")
    feats = model_features(cfg, man)
    assert "IAT" not in feats and "Drate" not in feats and len(feats) == 43
    cfg2 = load_config(env="local", overrides=["features.exclude_wifi=[]"])
    assert "IAT" in model_features(cfg2, man)
    assert "cumulative_packet_count" not in ble_features(cfg) and "timestamp" not in ble_features(cfg)
