"""M2: build synthetic drift timelines, non-IID clients, D_pub and BLE sub-federation.

Outputs per pool (dev | full) and timeline -> data/splits/<pool>/<timeline>/:
  assignments.parquet      source,row_id,window,client,split,y_attack19
  dpub.parquet             public calibration set (+ intro_window)
  ble_assignments.parquet  BLE row_index,window,client,split,y_binary
  window_definition.json, client_partitions.json, counts.csv, client_stats.csv
Report for the headline timeline -> reports/m2/ (tracked).

Usage: python scripts/m2_build_partitions.py [--pools dev full] [--env ...] [--set k=v ...]
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ids.data.io import FILES, processed_dir
from ids.data.schema import ATTACKS19, CATEGORIES, FEATURES
from ids.partition.build import (TEST, TRAIN, ble_stats, build_partition, dev_ble_subset,
                                 duplicate_mask, partition_stats)
from ids.partition.io import partition_dir, save_partition
from ids.utils import Run, load_config, set_global_seed
from ids.utils.config import REPO_ROOT
from ids.utils.jsonl import dumps
from ids.utils.seed import rng

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5",
                                                "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def inputs(cfg: dict, pool: str):
    proc = processed_dir(cfg)
    src = proc / FILES["full" if pool == "full" else "dev"]
    csv_rows = pd.read_parquet(src, columns=["row_id", "y_attack19"])
    csv_rows["is_dup"] = duplicate_mask(src, FEATURES)

    prof = pd.read_parquet(proc / FILES["wifi_profiling"], columns=["device_scenario"])
    prof["row_id"] = np.arange(len(prof), dtype=np.int32)
    prof["is_dup"] = duplicate_mask(proc / FILES["wifi_profiling"], FEATURES)
    if pool != "full":  # dev: subsample the big network-wide captures, keep devices whole
        bg = prof["device_scenario"].isin(cfg["profiling"]["background"]).to_numpy()
        keep = ~bg | (rng(cfg["seed"], "dev_profiling").random(len(prof)) < cfg["resources"]["dev_sample_frac"])
        prof = prof[keep]

    ble = pd.read_parquet(proc / FILES["ble"], columns=["y_binary", "split"])
    ble["row_index"] = np.arange(len(ble), dtype=np.int32)
    ble = ble[ble["split"].isin(["train", "test"])]
    if pool != "full":
        ble = dev_ble_subset(ble, cfg["ble"]["dev_frac"], cfg["seed"])
    return csv_rows, prof, ble


# ------------------------------------------------------------------ report
def _heat(ax, mat, xt, yt, title, cmap=SEQ, vmin=None, vmax=None):
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xt)), xt, rotation=90, fontsize=7)
    ax.set_yticks(range(len(yt)), yt, fontsize=7)
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def write_report(cfg, part, counts, cstats, ble_counts, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = cfg["timeline"]["windows"]
    K = cfg["federation"]["n_clients"]
    tr = counts[counts["split"] == TRAIN]

    # 1) windows x classes (train rows, log10)
    m = (tr.groupby(["window", "y_attack19"])["n"].sum().unstack(fill_value=0)
         .reindex(index=range(len(windows)), columns=range(len(ATTACKS19)), fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = _heat(ax, np.log10(m.values + 1), ATTACKS19, windows,
               f"Training rows per window and class ({part.pool} pool, {part.timeline}), log10")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("log10(rows+1)", color=INK2)
    cb.outline.set_visible(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m.values[i, j] == 0:
                ax.text(j, i, "–", ha="center", va="center", color=INK2, fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "window_class_counts.png", dpi=150); plt.close(fig)

    # 2) client x class share in three windows (row-normalized), small multiples
    show = [0, len(windows) // 2, len(windows) - 1]
    fig, axes = plt.subplots(1, len(show), figsize=(15, 4.4), sharey=True)
    for ax, w in zip(axes, show):
        g = (tr[tr.window == w].groupby(["client", "y_attack19"])["n"].sum().unstack(fill_value=0)
             .reindex(index=range(K), columns=range(len(ATTACKS19)), fill_value=0))
        share = g.div(g.sum(axis=1).clip(lower=1), axis=0)
        im = _heat(ax, share.values, ATTACKS19, [f"c{k:02d}" for k in range(K)],
                   f"{windows[w]}: class share per client", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01); cb.set_label("share of client's rows", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(out_dir / "client_class_share.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 3) JSD per client per window (strip + mean)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for w, g in cstats.groupby("window"):
        x = w + rng(0, "jit", w).uniform(-0.12, 0.12, len(g))
        ax.scatter(x, g["jsd_attack19_vs_window"], s=22, color="#2a78d6", alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.8)
        ax.plot([w - 0.25, w + 0.25], [g["jsd_attack19_vs_window"].mean()] * 2, color=INK, lw=2)
    ax.set_xticks(range(len(windows)), windows)
    ax.set_ylabel("JSD(client ‖ window), base 2")
    ax.set_ylim(0, 1); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Label-distribution divergence of each client (dots) and mean (bar)", loc="left")
    fig.tight_layout(); fig.savefig(out_dir / "client_jsd.png", dpi=150); plt.close(fig)

    # ---- PARTITION.md
    meta, wdef = part.meta, part.definition
    per_w = (counts.groupby(["window", "split"])["n"].sum().unstack(fill_value=0)
             .rename(columns={TRAIN: "train", TEST: "test"}))
    jsd_w = cstats.groupby("window")["jsd_attack19_vs_window"].agg(["mean", "min", "max"])
    lines = ["| window | new classes | train rows | test rows | JSD mean | JSD min | JSD max |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for i, w in enumerate(windows):
        new = ", ".join(wdef["new_classes_by_window"][w]) or "— (population drift)"
        lines.append(f"| {w} | {new} | {int(per_w.loc[i, 'train']):,} | {int(per_w.loc[i, 'test']):,} | "
                     f"{jsd_w.loc[i, 'mean']:.3f} | {jsd_w.loc[i, 'min']:.3f} | {jsd_w.loc[i, 'max']:.3f} |")
    dev = meta["devices"]
    owners: dict[int, list[str]] = {}
    for g, k in dev.get("owner", {}).items():
        owners.setdefault(k, []).append(g + (" (onboarded at drift)" if g in dev["held_back_until_drift"] else ""))
    dev_lines = "\n".join(f"- c{k:02d}: {', '.join(sorted(v))}" for k, v in sorted(owners.items()))
    size = cstats.pivot(index="client", columns="window", values="n_train").fillna(0).astype(int)
    size_lines = ["| client | " + " | ".join(windows) + " |", "|---|" + "---:|" * len(windows)]
    for k, r in size.iterrows():
        mark = " ⟲" if k in meta["population_drift"]["drifted_clients"] else ""
        size_lines.append(f"| c{k:02d}{mark} | " + " | ".join(f"{v:,}" for v in r.values) + " |")
    ble_t = ble_counts.groupby(["window", "split", "y_binary"])["n"].sum().unstack(fill_value=0)
    ble_lines = ["| window | split | benign | BLE-DoS |", "|---|---|---:|---:|"]
    for (w, s), r in ble_t.iterrows():
        ble_lines.append(f"| {windows[w]} | {'train' if s == TRAIN else 'test'} | {int(r.get(0, 0)):,} | {int(r.get(1, 0)):,} |")

    md = f"""# M2 partition report — `{part.pool}` pool, `{part.timeline}` timeline

Generated by `scripts/m2_build_partitions.py` (seed {meta['seed']}). Files: `data/splits/{part.pool}/{part.timeline}/`.

## Setup
- {meta['n_clients']} Wi-Fi/MQTT clients, Dirichlet α = {meta['dirichlet_alpha']} label skew per class; test share {meta['test_frac']:.0%} per (window, class), client-local.
- Exact duplicates dropped before splitting: {meta['n_csv_duplicates_dropped']:,} CSV rows, {meta['n_profiling_duplicates_dropped']:,} profiling rows. Train and test are therefore duplicate-disjoint.
- Each class enters at its introduction window and then recurs with **fresh, disjoint rows** in every later window.
- Population drift at {meta['population_drift']['window']}: clients {', '.join(f'c{k:02d}' for k in meta['population_drift']['drifted_clients'])} (⟲) get re-drawn attack mixes, and {len(dev.get('held_back_until_drift', []))} held-back device groups are onboarded.
- D_pub: {meta['n_dpub']:,} rows, removed from all client data. Window mode `{cfg['dpub']['window_mode']}`: at window t only classes introduced by t are used.

## Windows

{chr(10).join(lines)}

JSD is the Jensen–Shannon divergence between a client's 19-class training distribution and the whole window's, where 0 = identical and 1 = disjoint. It measures how non-IID the clients are.

![window x class](window_class_counts.png)

![client share](client_class_share.png)

![client jsd](client_jsd.png)

## Training rows per client and window

{chr(10).join(size_lines)}

## Benign device ownership (real devices from Zenodo profiling)
Network-wide captures ({', '.join(dev.get('background', []))}) are spread over all clients. Each device/network group belongs to exactly one client:

{dev_lines}

## D_pub by class
{', '.join(f'{k}: {v:,}' for k, v in meta['dpub_by_class'].items())}

## BLE sub-federation ({cfg['ble']['n_clients']} clients, official train/test split)

{chr(10).join(ble_lines)}
"""
    (out_dir / "PARTITION.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pools", nargs="+", default=["dev", "full"])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])

    with Run(cfg, "m2_build_partitions") as run:
        log, t0 = run.logger, time.time()
        summary = {}
        for pool in args.pools:
            csv_rows, prof, ble = inputs(cfg, pool)
            log.info("[%s] csv=%s (dups %s) profiling=%s ble=%s", pool, len(csv_rows),
                     int(csv_rows.is_dup.sum()), len(prof), len(ble))
            for tl in cfg["timeline"]["build"]:
                part = build_partition(cfg, tl, csv_rows, prof, ble, pool)
                out = partition_dir(cfg, pool, tl)
                save_partition(part, out)
                counts, cstats = partition_stats(part)
                bcounts = ble_stats(part.ble)
                counts.to_csv(out / "counts.csv", index=False)
                cstats.to_csv(out / "client_stats.csv", index=False)
                bcounts.to_csv(out / "ble_counts.csv", index=False)
                key = f"{pool}/{tl}"
                summary[key] = {"n_assigned": int(len(part.assignments)), "n_dpub": int(len(part.dpub)),
                                "n_ble": int(len(part.ble)),
                                "jsd_mean_by_window": cstats.groupby("window")["jsd_attack19_vs_window"].mean().round(4).tolist()}
                log.info("[%s] assigned=%s dpub=%s jsd=%s (%.0fs)", key, summary[key]["n_assigned"],
                         summary[key]["n_dpub"], summary[key]["jsd_mean_by_window"], time.time() - t0)
                if tl == cfg["timeline"]["headline"]:
                    rep = REPO_ROOT / "reports" / "m2" / pool
                    write_report(cfg, part, counts, cstats, bcounts, rep)
                    (rep / "client_partitions.json").write_text(dumps(part.meta, indent=2), encoding="utf-8")
                    (rep / "window_definition.json").write_text(dumps(part.definition, indent=2), encoding="utf-8")
        run.save_json("summary.json", summary)
        run.log("m2_done", **summary)


if __name__ == "__main__":
    main()
