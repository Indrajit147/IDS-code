"""M1 EDA report -> reports/m1/ (figures + EDA.md). Run after m1_prepare_data.py.

Uses the dev sample for feature-level analysis (fast, rare classes kept whole) and
the manifest for full-data counts. Includes a single-feature "shortcut" check:
features that alone separate classes almost perfectly are flagged as possible
capture artifacts (e.g. timing/ordering fields) for a decision before M3.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ids.data.io import feature_columns, load, load_manifest
from ids.data.schema import ATTACKS19, BLE_FEATURES, CATEGORIES, attack_category
from ids.utils import load_config
from ids.utils.config import REPO_ROOT

# Reference palette (dataviz skill, light mode), categorical slots in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
CAT_COLOR = dict(zip(CATEGORIES, SERIES))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def single_feature_scores(X: pd.DataFrame, y: np.ndarray, Xte=None, yte=None, seed=0) -> pd.Series:
    if Xte is None:
        X, Xte, y, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    out = {}
    for f in X.columns:
        clf = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=seed)
        clf.fit(X[[f]].fillna(-1), y)
        out[f] = f1_score(yte, clf.predict(Xte[[f]].fillna(-1)), average="macro")
    return pd.Series(out).sort_values(ascending=False)


def fig_class_counts(counts: dict, path):
    s = pd.Series(counts).reindex(ATTACKS19)[::-1]
    cats = [attack_category(a) for a in s.index]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.barh(s.index, s.values, color=[CAT_COLOR[c] for c in cats], height=0.72,
            edgecolor=SURFACE, linewidth=1)
    ax.set_xscale("log")
    for i, v in enumerate(s.values):
        ax.text(v * 1.08, i, f"{v:,}", va="center", color=INK2, fontsize=8)
    ax.set_xlim(right=s.max() * 6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("rows (log scale)")
    ax.set_title("CICIoMT2024 Wi-Fi/MQTT: rows per class (full data, 8.78M)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR[c]) for c in CATEGORIES]
    ax.legend(handles, CATEGORIES, frameon=False, loc="lower right", title="category")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_heatmap(mat: pd.DataFrame, path, title, vlim, cbar_label, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values, cmap=DIVERGING, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, color=INK2); cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_single_feature(scores: pd.Series, path, title, top=20):
    s = scores.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=SERIES[0], height=0.7)
    for i, v in enumerate(s.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", color=INK2, fontsize=7)
    ax.set_xlim(0, 1.08); ax.grid(axis="y", visible=False)
    ax.set_xlabel("macro-F1 using this feature alone (depth-4 tree)")
    ax.set_title(title, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_ble(ble: pd.DataFrame, path):
    feats = [f for f in BLE_FEATURES if f != "cumulative_packet_count"]
    fig, axes = plt.subplots(1, len(feats), figsize=(3 * len(feats), 2.8))
    for ax, f in zip(axes, feats):
        for j, lab in enumerate(["Benign", "BLE-DoS"]):  # fixed order: Benign = slot 1
            v = ble.loc[ble.y_attack == lab, f].dropna()
            hi = np.nanpercentile(ble[f], 99.5)
            ax.hist(v.clip(upper=hi), bins=40, density=True, histtype="step", linewidth=2,
                    color=SERIES[j], label=lab)
        ax.set_title(f, loc="left", fontsize=9); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("BLE packet features: Benign vs BLE-DoS (density, clipped at p99.5)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt="{:.4g}") -> str:
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for row in df.itertuples():  # itertuples keeps per-column dtypes (iterrows upcasts ints)
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    args = ap.parse_args()
    cfg = load_config(env=args.env)
    man = load_manifest(cfg)
    rep = REPO_ROOT / "reports" / "m1"
    rep.mkdir(parents=True, exist_ok=True)
    feats = feature_columns(man)

    fig_class_counts(man["class_counts_attack19"], rep / "class_counts.png")

    # EDA analyses every non-constant feature, including ones excluded from modelling (D6)
    dev = load(cfg, "dev", columns=[*feats, "y_category"])
    X, ycat = dev[feats], dev["y_category"].to_numpy()

    corr = X.corr(method="spearman")
    fig_heatmap(corr, rep / "feature_correlation.png",
                "Spearman correlation between features (dev sample)", 1, "Spearman ρ", (10, 9))
    iu = np.triu_indices_from(corr, 1)
    pairs = pd.DataFrame({"a": corr.index[iu[0]], "b": corr.columns[iu[1]], "rho": corr.values[iu]})
    high = pairs[pairs.rho.abs() >= 0.98].sort_values("rho", key=abs, ascending=False)

    z = (X - X.mean()) / X.std().replace(0, 1)
    prof = z.groupby(pd.Series(ycat, index=X.index).map(dict(enumerate(CATEGORIES)))).mean()
    prof = prof.reindex(CATEGORIES)
    fig_heatmap(prof.clip(-3, 3), rep / "class_profiles.png",
                "Per-category mean of z-scored features (dev sample)", 3, "mean z-score", (12, 3.6))

    sf = single_feature_scores(X, ycat, seed=cfg["seed"])
    fig_single_feature(sf, rep / "single_feature_wifi.png",
                       "Wi-Fi/MQTT: 6-class macro-F1, single feature (top 20)")

    ble = load(cfg, "ble")
    fig_ble(ble, rep / "ble_features.png")
    btr, bte = ble[ble.split == "train"], ble[ble.split == "test"]
    sf_ble = single_feature_scores(btr[BLE_FEATURES + ["timestamp"]], btr.y_binary.to_numpy(),
                                   bte[BLE_FEATURES + ["timestamp"]], bte.y_binary.to_numpy())
    fig_single_feature(sf_ble, rep / "single_feature_ble.png",
                       "BLE: binary macro-F1, single feature (train→test)")

    # ---------- EDA.md ----------
    dup, dv = man.get("duplicates", {}), man["dev_sample"]
    stats = pd.DataFrame(man["column_stats"]).T[["min", "max", "mean", "std"]]
    stats["dtype"] = pd.Series(man["dtypes"])
    stats["dev_nunique"] = X.nunique().reindex(stats.index)
    stats.index.name = "feature"
    zc = pd.DataFrame(man["zenodo_vs_csv"]["per_class"]).set_index("class")
    zc.index.name = "class"
    zc["csv_exact_duplicates"] = [int(dup.get("duplicate_rows_by_class", {}).get(c, 0)) for c in zc.index]
    dvt = pd.DataFrame(dv["per_class"]).T[["n_full", "n_sample", "weight"]]
    dvt[["n_full", "n_sample"]] = dvt[["n_full", "n_sample"]].astype(int)
    dvt.index.name = "class"
    ble_counts = pd.DataFrame(man["ble"]["counts_by_label_split"]).pivot(
        index="y_attack", columns="split", values="n").fillna(0).astype(int)
    ble_counts.index.name = "label"
    flagged = sf[sf >= 0.80]

    md = f"""# M1 EDA report

Generated by `scripts/m1_eda.py` from `data/processed/` (see `data_manifest.json`).

## Raw data
- File: `{man['raw']['path'].split('/')[-1].split(chr(92))[-1]}`, {man['raw']['bytes']:,} bytes, SHA-256 `{man['raw']['sha256']}`
- {man['raw']['n_rows']:,} rows × {man['raw']['n_cols']} columns. NaN: {man['cleaning']['n_nan_total']}, inf: {man['cleaning']['n_inf_total']}
- Constant columns (excluded by the loader): {', '.join(man['cleaning']['constant_columns']) or 'none'}
- Kept as float64 to avoid precision loss: {', '.join(k for k, v in man['dtypes'].items() if v == 'float64') or 'none'}; all others float32
- Processed full parquet: {man['full_parquet']['bytes'] / 1e6:,.0f} MB

![class counts](class_counts.png)

## Duplicates (exact match on all 45 features)
- Duplicate rows: **{dup.get('n_duplicate_rows', 0):,}** ({dup.get('n_duplicate_rows', 0) / man['raw']['n_rows']:.1%} of rows) in {dup.get('n_duplicate_groups', 0):,} groups; unique feature rows: {dup.get('n_unique_feature_rows', 0):,}
- Duplicate groups with **conflicting labels**: {dup.get('n_conflicting_label_groups', 0):,} ({dup.get('n_rows_in_conflicting_groups', 0):,} rows)
- Duplicate rows by class: {', '.join(f'{k}: {v:,}' for k, v in dup.get('duplicate_rows_by_class', {}).items())}

Duplicates are kept in the processed file. Removing them, and making sure train/test splits are duplicate-disjoint, is decided in M2.

## Dev sample
5% per class, but classes with ≤{dv['floor']:,} rows are kept whole and every other class gets at least {dv['floor']:,} rows → **{dv['n_rows']:,} rows**. `weight` = n_full/n_sample restores the full-data class priors.

{md_table(dvt, "{:.2f}")}

## Feature statistics (full data; nunique on dev sample)

{md_table(stats)}

## Correlation
Feature pairs with |Spearman ρ| ≥ 0.98 ({len(high)}): {', '.join(f'{a}~{b} ({r:+.3f})' for a, b, r in high.itertuples(index=False)) or 'none'}

![correlation](feature_correlation.png)

![class profiles](class_profiles.png)

## Single-feature shortcut check
Each feature alone is given to a depth-4 decision tree (balanced class weights) and scored by 6-class macro-F1 on a stratified 70/30 split of the dev sample. Features scoring ≥0.80 are listed as possible capture artifacts and are excluded from modelling by default (`features.exclude_wifi`, decision D6):
{', '.join(f'`{f}` ({v:.3f})' for f, v in flagged.items()) or 'none'}

![single feature wifi](single_feature_wifi.png)

## Zenodo Wi-Fi (train+test) vs our CSV

{md_table(zc)}

Totals: CSV {man['zenodo_vs_csv']['csv_total']:,} vs Zenodo {man['zenodo_vs_csv']['zenodo_total']:,}. For every class except Recon-Port-Scan, `diff` equals minus the CSV's exact-duplicate count, so the Zenodo Wi-Fi table is effectively a **de-duplicated** CSV. Port-Scan keeps 195 rows that the CSV marks as duplicates, probably because of float rounding in Zenodo's own export.

## Wi-Fi device profiling (Zenodo, benign only)
{man['wifi_profiling']['n_rows']:,} rows from {man['wifi_profiling']['n_device_scenarios']} device/scenario captures. Largest: {', '.join(f'{k} ({v:,})' for k, v in list(man['wifi_profiling']['rows_per_device_scenario'].items())[:6])}.

## BLE (Zenodo, packet-level)
Dropped all-NaN columns: {', '.join(man['ble']['dropped_all_nan_columns'])}. `UNKNOWN` was relabelled to `BLE-DoS` (all of those rows come from DoS capture files).

{md_table(ble_counts)}

![ble features](ble_features.png)

Single-feature binary macro-F1, official train→test split:
{', '.join(f'`{f}` ({v:.3f})' for f, v in sf_ble.items())}

![single feature ble](single_feature_ble.png)
"""
    (rep / "EDA.md").write_text(md, encoding="utf-8")
    sf.rename("macro_f1").to_csv(rep / "single_feature_wifi.csv")
    sf_ble.rename("macro_f1").to_csv(rep / "single_feature_ble.csv")
    print("wrote", rep / "EDA.md")
    print("flagged wifi:", flagged.round(3).to_dict())
    print("ble:", sf_ble.round(3).to_dict())


if __name__ == "__main__":
    main()
