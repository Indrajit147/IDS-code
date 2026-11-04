import numpy as np
import pandas as pd
import pytest

from ids.data.schema import ATTACK19_CODE, ATTACKS19
from ids.partition.allocate import drift_shares, jsd, proportional_counts, train_test_mask
from ids.partition.build import (SRC_CSV, SRC_PROFILING, TEST, TRAIN, build_partition,
                                 device_group, partition_stats, select_dpub)
from ids.partition.timeline import intro_windows
from ids.utils import load_config

W = ["t0", "t1", "t2", "t3", "t4", "t5"]


def test_intro_windows_validation():
    cfg = load_config(env="local")
    intro = intro_windows(cfg["timeline"]["definitions"]["category_intro"], W)
    assert intro["Benign"] == 0 and intro["DoS-SYN"] == 0 and intro["DDoS-UDP"] == 1
    assert intro["Recon-Port-Scan"] == 2 and intro["Recon-OS-Scan"] == 5 and intro["ARP-Spoofing"] == 4
    alt = intro_windows(cfg["timeline"]["definitions"]["alt_order"], W)
    assert alt["Recon-VulScan"] == 4 and alt["MQTT-Malformed-Data"] == 1
    with pytest.raises(ValueError, match="twice"):
        intro_windows({"t0": ["Benign", "DoS-*"], "t1": ["DoS-SYN", "*"]}, W)
    with pytest.raises(ValueError, match="never introduced"):
        intro_windows({"t0": ["Benign"]}, W)
    with pytest.raises(ValueError, match="matches no class"):
        intro_windows({"t0": ["Benign", "BLE-*", "*"]}, W)


def test_allocation_primitives():
    c = proportional_counts(10, np.array([0.5, 0.3, 0.2]))
    assert c.sum() == 10 and c.tolist() == [5, 3, 2]
    assert proportional_counts(7, np.array([1, 1, 1])).sum() == 7
    m = train_test_mask(np.random.default_rng(0), 100, 0.2)
    assert m.sum() == 20
    assert train_test_mask(np.random.default_rng(0), 2, 0.2).sum() == 1
    base = np.array([0.1, 0.2, 0.3, 0.4])
    d = drift_shares(np.random.default_rng(1), base, np.array([0, 3]), 0.5)
    assert np.isclose(d.sum(), 1) and np.allclose(d[[1, 2]], base[[1, 2]])
    assert jsd([1, 0], [1, 0]) == 0 and np.isclose(jsd([1, 0], [0, 1]), 1)


def test_device_group():
    assert device_group("Owltron_Camera_WAN_RECORDING") == "Owltron_Camera|WAN"
    assert device_group("Singcall_Power") == "Singcall|Power"
    assert device_group("Multifunctional_Pager_LAN_APP") == "Multifunctional_Pager|LAN"


def test_dpub_caps():
    y = np.array([0] * 1000 + [1] * 100000 + [5] * 40)
    cfg_d = {"benign": 500, "attack_frac": 0.01, "min_per_class": 50, "max_class_frac": 0.05}
    pos = select_dpub(y, cfg_d, seed=0)
    counts = np.bincount(y[pos], minlength=19)
    assert counts[0] == 50            # 500 requested, capped at 5% of 1000
    assert counts[1] == 1000          # 1% of a large attack class
    assert counts[5] == 2             # min 50 requested, capped at 5% of 40


def _toy(seed=0, per_class=400):
    r = np.random.default_rng(seed)
    y = np.repeat(np.arange(len(ATTACKS19)), per_class)
    csv = pd.DataFrame({"row_id": np.arange(len(y), dtype=np.int32), "y_attack19": y.astype(np.int8),
                        "is_dup": r.random(len(y)) < 0.01})
    names = (["Active"] * 300 + ["Owltron_Camera_LAN_MIC"] * 40 + ["Owltron_Camera_WAN_MIC"] * 30 +
             ["Ecobee_Camera_LAN_WATCH"] * 30 + ["Singcall_Power"] * 10)
    prof = pd.DataFrame({"row_id": np.arange(len(names), dtype=np.int32), "device_scenario": names,
                         "is_dup": False})
    ble = pd.DataFrame({"row_index": np.arange(600, dtype=np.int32),
                        "y_binary": np.r_[np.zeros(300), np.ones(300)].astype(np.int8),
                        "split": np.where(np.arange(600) % 5 == 0, "test", "train")})
    return csv, prof, ble


@pytest.fixture(scope="module")
def built():
    cfg = load_config(env="local", overrides={"federation": {"n_clients": 5},
                                              "dpub": {"benign": 40, "attack_frac": 0.1}})
    csv, prof, ble = _toy()
    return cfg, csv, build_partition(cfg, "category_intro", csv, prof, ble, "toy")


def test_partition_invariants(built):
    cfg, csv, part = built
    a = part.assignments
    key = a["source"].astype(int) * 10**7 + a["row_id"]
    assert key.is_unique                                         # every row used at most once
    dpub_ids = set(part.dpub["row_id"])
    assert not dpub_ids & set(a.loc[a.source == SRC_CSV, "row_id"])   # D_pub disjoint from clients
    dups = set(csv.loc[csv.is_dup, "row_id"])
    assert not dups & set(a.loc[a.source == SRC_CSV, "row_id"]) and not dups & dpub_ids
    used = len(a[a.source == SRC_CSV]) + len(part.dpub)
    assert used == int((~csv.is_dup).sum())                      # nothing else lost
    intro = part.definition["intro_window"]
    first = a[a.source == SRC_CSV].groupby("y_attack19")["window"].min()
    for c, w in first.items():
        assert w == intro[ATTACKS19[c]]                          # never before its intro window
    assert set(a.loc[a.y_attack19 == ATTACK19_CODE["DoS-SYN"], "window"]) == set(range(6))
    test_share = (a.split == TEST).mean()
    assert 0.18 < test_share < 0.22
    assert a["client"].between(0, 4).all()


def test_devices_owned_by_one_client(built):
    _, _, part = built
    a = part.assignments
    prof = a[a.source == SRC_PROFILING]
    names = np.array((["Active"] * 300 + ["Owltron_Camera_LAN_MIC"] * 40 + ["Owltron_Camera_WAN_MIC"] * 30 +
                      ["Ecobee_Camera_LAN_WATCH"] * 30 + ["Singcall_Power"] * 10))
    grp = pd.Series(names[prof["row_id"]]).map(lambda s: s if s == "Active" else device_group(s))
    owners = prof.assign(g=grp.values).groupby("g")["client"].nunique()
    assert (owners.drop("Active") == 1).all()                   # each device group -> one client
    assert owners["Active"] > 1                                  # background spread over clients
    held = part.meta["devices"]["held_back_until_drift"]
    assert len(held) == 1                                        # round(0.25 * 4 groups)
    held_rows = prof[grp.values == held[0]]
    assert set(held_rows["window"]) == {5}                       # onboarded only at drift window


def test_deterministic_and_ble(built):
    cfg, _, part = built
    csv, prof, ble = _toy()
    again = build_partition(cfg, "category_intro", csv, prof, ble, "toy")
    pd.testing.assert_frame_equal(part.assignments, again.assignments)
    pd.testing.assert_frame_equal(part.dpub, again.dpub)
    b = part.ble
    assert set(b["window"]) == {4, 5} and b["row_index"].is_unique and len(b) == 600
    assert ((b.split == TEST) == (ble.set_index("row_index").loc[b.row_index, "split"].values == "test")).all()
    counts, cstats = partition_stats(part)
    assert counts["n"].sum() == len(part.assignments)
    assert cstats["jsd_attack19_vs_window"].between(0, 1).all()
