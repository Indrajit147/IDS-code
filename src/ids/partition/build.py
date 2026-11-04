"""Build the experimental world: windows x clients x {train, test} + D_pub (+ BLE).

Every random choice uses a seed derived from (cfg.seed, purpose, keys...), so
the partition is identical across runs/machines. Output is row-level:
assignments say which processed row goes to which (window, client, split).

Sources: 0 = CICIoMT2024 CSV (processed parquet, row_id = position),
         1 = Zenodo Wi-Fi benign device profiling (row_id = position in its parquet).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..data.schema import (ATTACK19_TO_CATEGORY_CODE, ATTACKS19, CATEGORIES, FEATURES,
                           attack_category)
from ..data.sample import stratified_indices
from ..utils.seed import rng
from .allocate import (dirichlet_shares, drift_shares, jsd, proportional_counts,
                       split_by_counts, train_test_mask)
from .timeline import window_definition

SRC_CSV, SRC_PROFILING = 0, 1
TRAIN, TEST = 0, 1
NETWORK_TOKENS = ("LAN", "WAN", "Power")
CAT_OF = np.asarray(ATTACK19_TO_CATEGORY_CODE)  # attack19 code -> category code


# ---------------------------------------------------------------- helpers
def duplicate_mask(path, columns=FEATURES) -> np.ndarray:
    """True for rows whose feature values exactly repeat an earlier row in this file."""
    hashes = []
    for b in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=list(columns)):
        hashes.append(pd.util.hash_pandas_object(b.to_pandas(), index=False).to_numpy())
    return pd.Series(np.concatenate(hashes)).duplicated(keep="first").to_numpy()


def device_group(device_scenario: str) -> str:
    """'Owltron_Camera_WAN_RECORDING' -> 'Owltron_Camera|WAN'; 'Singcall_Power' -> 'Singcall|Power'."""
    toks = device_scenario.split("_")
    for i, t in enumerate(toks):
        if t in NETWORK_TOKENS:
            return "_".join(toks[:i]) + "|" + t
    return device_scenario


def select_dpub(y19: np.ndarray, cfg_dpub: dict, seed: int) -> np.ndarray:
    """Positions (into y19) of the public calibration set (proposal, D2): a small fixed
    benign subset plus `attack_frac` of every attack class, so each category contributes
    that share. Rare classes get `min_per_class`; no class gives up more than `max_class_frac`."""
    sizes = np.bincount(y19, minlength=len(ATTACKS19))
    target = np.round(sizes * cfg_dpub["attack_frac"]).astype(np.int64)
    target[ATTACKS19.index("Benign")] = cfg_dpub["benign"]
    target = np.where(sizes > 0, np.maximum(target, cfg_dpub["min_per_class"]), 0)
    target = np.minimum(target, np.floor(sizes * cfg_dpub["max_class_frac"]).astype(np.int64))
    picks = [rng(seed, "dpub", c).choice(np.flatnonzero(y19 == c), target[c], replace=False)
             for c in range(len(ATTACKS19)) if target[c] > 0]
    return np.sort(np.concatenate(picks))


@dataclass
class Partition:
    pool: str
    timeline: str
    definition: dict
    assignments: pd.DataFrame
    dpub: pd.DataFrame
    ble: pd.DataFrame | None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- main builder
def build_partition(cfg: dict, timeline: str, csv_rows: pd.DataFrame,
                    profiling: pd.DataFrame | None, ble: pd.DataFrame | None,
                    pool: str) -> Partition:
    """csv_rows: [row_id, y_attack19, is_dup]; profiling: [row_id, device_scenario, is_dup];
    ble: [row_index, y_binary, split] (official split)."""
    seed = cfg["seed"]
    tl, fed = cfg["timeline"], cfg["federation"]
    windows = tl["windows"]
    W, K, alpha = len(windows), fed["n_clients"], fed["dirichlet_alpha"]
    wdef = window_definition(cfg, timeline)
    intro = [wdef["intro_window"][a] for a in ATTACKS19]
    pd_cfg = fed["population_drift"]
    w_drift = windows.index(pd_cfg["window"])
    n_drift = int(round(pd_cfg["frac_clients"] * K))
    drifted = np.sort(rng(seed, "drifted_clients").choice(K, n_drift, replace=False))
    rows: list[pd.DataFrame] = []

    def emit(src, ids, w, client_of, split, y):
        rows.append(pd.DataFrame({"source": np.int8(src), "row_id": ids.astype(np.int32),
                                  "window": np.int8(w), "client": client_of.astype(np.int8),
                                  "split": np.int8(split), "y_attack19": np.int8(y)}))

    def allocate(ids, shares, gen_key):
        """Assign ids to clients by shares; returns client index per id."""
        counts = proportional_counts(len(ids), shares)
        return np.repeat(np.arange(K), counts)[rng(seed, *gen_key).permutation(len(ids))]

    # ---- CSV rows: dedup, D_pub, then class -> windows -> train/test -> clients
    keep = csv_rows[~csv_rows["is_dup"]] if tl["dedup"] else csv_rows
    ids_all, y_all = keep["row_id"].to_numpy(), keep["y_attack19"].to_numpy()
    dpos = select_dpub(y_all, cfg["dpub"], seed)
    dpub = pd.DataFrame({"source": np.int8(SRC_CSV), "row_id": ids_all[dpos].astype(np.int32),
                         "y_attack19": y_all[dpos].astype(np.int8)})
    dpub["intro_window"] = np.asarray(intro, np.int8)[dpub["y_attack19"]]
    rest = np.ones(len(ids_all), bool); rest[dpos] = False
    ids_all, y_all = ids_all[rest], y_all[rest]

    for c in range(len(ATTACKS19)):
        cid = ids_all[y_all == c]
        if not len(cid):
            continue
        cid = cid[rng(seed, "perm", c).permutation(len(cid))]
        base = dirichlet_shares(rng(seed, "dir", c), K, alpha)
        drifted_sh = drift_shares(rng(seed, "drift", c), base, drifted, alpha)
        active = range(intro[c], W)
        for w, chunk in zip(active, np.array_split(cid, len(active))):
            shares = drifted_sh if w >= w_drift else base
            test = train_test_mask(rng(seed, "tt", c, w), len(chunk), tl["test_frac"])
            for split, sel in ((TRAIN, ~test), (TEST, test)):
                part = chunk[sel]
                emit(SRC_CSV, part, w, allocate(part, shares, ("alloc", c, w, split)), split, c)

    # ---- Benign device profiling: background spread over everyone, devices owned by one client
    meta_dev: dict = {}
    if profiling is not None and cfg["profiling"]["use"]:
        prof = profiling[~profiling["is_dup"]] if tl["dedup"] else profiling
        bg_names = set(cfg["profiling"]["background"])
        groups = prof["device_scenario"].map(lambda s: s if s in bg_names else device_group(s))
        dev_groups = sorted(set(groups) - bg_names)
        gperm = rng(seed, "device_groups").permutation(len(dev_groups))
        n_hold = int(round(pd_cfg["held_back_device_frac"] * len(dev_groups)))
        held = {dev_groups[i] for i in gperm[:n_hold]}
        cperm = rng(seed, "device_clients").permutation(K)
        owner = {dev_groups[g]: int(cperm[j % K]) for j, g in enumerate(gperm)}
        for gi, g in enumerate(sorted(set(groups))):
            gid = prof.loc[groups == g, "row_id"].to_numpy()
            gid = gid[rng(seed, "perm_prof", gi).permutation(len(gid))]
            active = [w_drift] if g in held else list(range(W))
            for w, chunk in zip(active, np.array_split(gid, len(active))):
                test = train_test_mask(rng(seed, "tt_prof", gi, w), len(chunk), tl["test_frac"])
                for split, sel in ((TRAIN, ~test), (TEST, test)):
                    part = chunk[sel]
                    if g in bg_names:
                        cl = allocate(part, np.ones(K), ("alloc_bg", gi, w, split))
                    else:
                        cl = np.full(len(part), owner[g])
                    emit(SRC_PROFILING, part, w, cl, split, 0)
        meta_dev = {"device_groups": dev_groups, "held_back_until_drift": sorted(held),
                    "owner": owner, "background": sorted(bg_names)}

    assignments = pd.concat(rows, ignore_index=True)

    # ---- BLE sub-federation (official train/test split; windows start..last)
    ble_out = None
    if ble is not None:
        bc = cfg["ble"]
        Kb, w0 = bc["n_clients"], windows.index(bc["start_window"])
        parts = []
        for split_name, split in (("train", TRAIN), ("test", TEST)):
            for yb in (0, 1):
                bid = ble.loc[(ble["split"] == split_name) & (ble["y_binary"] == yb), "row_index"].to_numpy()
                bid = bid[rng(seed, "ble_perm", split, yb).permutation(len(bid))]
                shares = dirichlet_shares(rng(seed, "ble_dir", yb), Kb, alpha)
                active = range(w0, W)
                for w, chunk in zip(active, np.array_split(bid, len(active))):
                    counts = proportional_counts(len(chunk), shares)
                    cl = np.repeat(np.arange(Kb), counts)[rng(seed, "ble_alloc", split, yb, w).permutation(len(chunk))]
                    parts.append(pd.DataFrame({"row_index": chunk.astype(np.int32), "window": np.int8(w),
                                               "client": cl.astype(np.int8), "split": np.int8(split),
                                               "y_binary": np.int8(yb)}))
        ble_out = pd.concat(parts, ignore_index=True)

    meta = {
        "pool": pool, "timeline": timeline, "seed": seed, "n_clients": K, "dirichlet_alpha": alpha,
        "dedup": tl["dedup"], "test_frac": tl["test_frac"],
        "n_csv_duplicates_dropped": int(csv_rows["is_dup"].sum()) if tl["dedup"] else 0,
        "n_profiling_duplicates_dropped": int(profiling["is_dup"].sum()) if (profiling is not None and tl["dedup"]) else 0,
        "population_drift": {"window": pd_cfg["window"], "drifted_clients": drifted.tolist()},
        "devices": meta_dev,
        "n_dpub": int(len(dpub)),
        "dpub_by_class": {ATTACKS19[k]: int(v) for k, v in dpub["y_attack19"].value_counts().sort_index().items()},
    }
    return Partition(pool, timeline, wdef, assignments, dpub, ble_out, meta)


# ---------------------------------------------------------------- statistics
def partition_stats(part: Partition) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-form counts and per-(window, client) non-IID measures on the train split."""
    a = part.assignments
    counts = (a.groupby(["window", "client", "split", "source", "y_attack19"]).size()
              .rename("n").reset_index())
    tr = a[a["split"] == TRAIN]
    rows = []
    for w, g in tr.groupby("window"):
        glob = np.bincount(g["y_attack19"], minlength=len(ATTACKS19))
        for k, gk in g.groupby("client"):
            loc = np.bincount(gk["y_attack19"], minlength=len(ATTACKS19))
            cat = np.bincount(CAT_OF[gk["y_attack19"].to_numpy()], minlength=len(CATEGORIES))
            rows.append({"window": int(w), "client": int(k), "n_train": int(len(gk)),
                         "jsd_attack19_vs_window": jsd(loc, glob),
                         "n_classes_present": int((loc > 0).sum()),
                         "n_classes_in_window": int((glob > 0).sum()),
                         **{f"n_{c}": int(v) for c, v in zip(CATEGORIES, cat)}})
    return counts, pd.DataFrame(rows)


def ble_stats(ble: pd.DataFrame) -> pd.DataFrame:
    return ble.groupby(["window", "client", "split", "y_binary"]).size().rename("n").reset_index()


def dev_ble_subset(ble: pd.DataFrame, frac: float, seed: int, floor: int = 5000) -> pd.DataFrame:
    """Stratified (label x split) BLE subsample for the local dev pool."""
    key = (ble["y_binary"].astype(int) * 2 + (ble["split"] == "test").astype(int)).to_numpy()
    pos, _ = stratified_indices(key, frac, floor, seed)
    return ble.iloc[pos]
