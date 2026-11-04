"""M3: edge ensemble + centralized (E1) and static baselines, BLE edge model, latency.

For every window t the centralized model trains on ALL clients' train rows of
windows 0..t, then is scored on the test set of EVERY window w (a T x T matrix).
Row t=0 of that matrix is the static baseline (trained on t0 only, never updated).

Usage: python scripts/m3_baselines.py [--pool dev|full] [--target y_category] [--env ...]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from ids.data.io import ble_features, model_features
from ids.data.schema import ATTACKS19, CATEGORIES
from ids.eval.harness import Evaluator, load_ble_test_sets, load_test_sets, predict_global
from ids.eval.plots import fig_confusions, fig_curves, fig_matrix
from ids.models.edge import EdgeConfig, EdgeEnsemble, measure_latency
from ids.partition.io import load_ble_rows, load_rows
from ids.utils import Run, load_config, set_global_seed
from ids.utils.config import REPO_ROOT
from ids.utils.jsonl import dumps
from ids.utils.seed import rng

CLASS_NAMES = {"y_category": CATEGORIES, "y_attack19": ATTACKS19, "y_binary": ["benign", "attack"]}


def val_split(y: np.ndarray, frac: float, seed: int, key) -> np.ndarray:
    """Stratified boolean validation mask (>=1 row per class with >=2 rows)."""
    m = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(max(int(round(frac * len(idx))), 1), len(idx) - 1) if len(idx) >= 2 else 0
        if k:
            m[rng(seed, "val", *key, int(c)).choice(idx, k, replace=False)] = True
    return m


def edge_config(cfg: dict, n_classes: int) -> EdgeConfig:
    mc = cfg["models"]
    return EdgeConfig(n_classes=n_classes, xgb_params=mc["xgb"], lgb_params=mc["lgb"],
                      early_stopping=mc["early_stopping"], weight_cap=mc["weight_cap"],
                      n_jobs=cfg["resources"]["n_jobs"], seed=cfg["seed"])


def run_matrix(cfg, run, train_fn, evaluator, train_windows, n_classes, tag):
    """Train at each t in train_windows; the evaluator scores every component on every test window."""
    models = {}
    for t in train_windows:
        X, y = train_fn(t)
        vm = val_split(y, cfg["models"]["val_frac"], cfg["seed"], (tag, t))
        t0 = time.perf_counter()
        m = EdgeEnsemble(edge_config(cfg, n_classes)).fit(X[~vm], y[~vm], X[vm], y[vm])
        sizes = m.save(run.path("models", tag, f"t{t}"))
        run.logger.info("[%s] trained through t%d on %s rows in %.1fs (w_xgb=%.1f, T=%.2f, iters xgb=%s lgb=%s)",
                        tag, t, f"{len(y):,}", time.perf_counter() - t0, m.w_xgb_, m.T_,
                        m.fit_info_["xgb_best_iteration"], m.fit_info_["lgb_best_iteration"])
        models[t] = m
        evaluator.evaluate(tag, t, predict_global(m), extra={"fit": m.fit_info_, **sizes})
        del X, y
    return evaluator.frame("all").drop(columns=["slice", "per_class", "macro_auc_ovr"]), models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default=None, help="dev | full (default from resources.use_dev_sample)")
    ap.add_argument("--target", default=None)
    ap.add_argument("--timeline", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    pool = args.pool or ("dev" if cfg["resources"]["use_dev_sample"] else "full")
    target = args.target or cfg["labels"]["headline_target"]
    tl = args.timeline or cfg["timeline"]["headline"]
    windows = cfg["timeline"]["windows"]
    names = CLASS_NAMES[target]
    feats = model_features(cfg)

    with Run(cfg, f"m3_baselines_{pool}") as run:
        log = run.logger
        log.info("pool=%s target=%s timeline=%s features=%d", pool, target, tl, len(feats))

        test_sets = load_test_sets(cfg, pool, tl, target, feats)
        log.info("test rows per window: %s", {windows[w]: len(v.y) for w, v in test_sets.items()})
        evaluator = Evaluator(test_sets, names)

        def wifi_train(t):
            d = load_rows(cfg, pool, tl, windows=list(range(t + 1)), split="train", features=feats)
            return d[feats].to_numpy(np.float32), d[target].to_numpy()

        res, models = run_matrix(cfg, run, wifi_train, evaluator, range(len(windows)), len(names), "wifi")

        # ---- latency + size (smallest and largest model)
        lat = {}
        Xlat = test_sets[len(windows) - 1].X
        for t in (0, len(windows) - 1):
            lat[windows[t]] = measure_latency(models[t], Xlat, seed=cfg["seed"])
            lat[windows[t]].update(res[(res.train_through == t) & (res.variant == "calibrated")]
                                   [["xgb_bytes", "lgb_bytes"]].iloc[0].astype(int).to_dict())
        log.info("latency: %s", lat)

        # ---- BLE edge model (binary), centralized over BLE windows
        w0 = windows.index(cfg["ble"]["start_window"])
        bfeats = ble_features(cfg)
        ble_eval = Evaluator(load_ble_test_sets(cfg, pool, tl, bfeats, w0), ["benign", "BLE-DoS"])

        def ble_train(t):
            d = load_ble_rows(cfg, pool, tl, windows=list(range(w0, t + 1)), split="train")
            return d[bfeats].to_numpy(np.float32), d["y_binary"].to_numpy()

        ble_res, _ = run_matrix(cfg, run, ble_train, ble_eval, range(w0, len(windows)), 2, "ble")

        # ---- persist
        allres = pd.concat([res, ble_res], ignore_index=True)
        allres.to_json(run.path("results.jsonl"), orient="records", lines=True)
        run.save_json("latency.json", lat)

        rep = REPO_ROOT / "reports" / "m3" / pool
        rep.mkdir(parents=True, exist_ok=True)
        last = len(windows) - 1
        fig_matrix(evaluator.matrix("wifi"), windows, rep / "macro_f1_matrix.png",
                   f"Centralized ensemble: macro-F1 ({pool} pool)")
        M = evaluator.matrix("wifi")
        fig_curves({"Centralized, retrained each window (E1)": pd.Series({w: M.loc[w, w] for w in M.columns}),
                    "Static, trained on t0 only": M.loc[0]}, windows, rep / "static_vs_centralized.png",
                   "Drift degradation without updates vs. per-window retraining")
        fig_confusions([(evaluator.confusions[("wifi", 0, last, "all")], "Static t0 model, last window (row-norm.)"),
                        (evaluator.confusions[("wifi", last, last, "all")], "Centralized, last window (row-norm.)")],
                       names, rep / "confusion_last_window.png")
        write_report(cfg, res, ble_res, lat, models, pool, target, tl, feats, rep, run.dir.name)
        cols = ["model", "train_through", "eval_window", "variant", "n", "accuracy", "macro_f1",
                "weighted_f1", "ece", "auroc_attack_vs_benign"]
        allres[cols].to_csv(rep / "results.csv", index=False)
        log.info("report -> %s", rep)


def write_report(cfg, res, ble_res, lat, models, pool, target, tl, feats, rep, run_name):
    windows = cfg["timeline"]["windows"]
    c = res[res.variant == "calibrated"]
    diag = c[c.train_through == c.eval_window].set_index("eval_window")
    static = c[c.train_through == 0].set_index("eval_window")
    T = len(windows)
    cum_rows = []
    for t in range(T):
        sub = c[(c.train_through == t) & (c.eval_window <= t)]
        cum_rows.append((sub["macro_f1"] * sub["n"]).sum() / sub["n"].sum())
    lines = ["| window | new classes | E1 centralized macro-F1 | E1 accuracy | static macro-F1 | static accuracy | E1 n_train | w_xgb | T |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    wdefs = cfg["timeline"]["definitions"][tl]
    for w in range(T):
        m = models[w].fit_info_
        lines.append(f"| {windows[w]} | {', '.join(wdefs[windows[w]]) or '—'} | {diag.loc[w, 'macro_f1']:.4f} | "
                     f"{diag.loc[w, 'accuracy']:.4f} | {static.loc[w, 'macro_f1']:.4f} | {static.loc[w, 'accuracy']:.4f} | "
                     f"{m['n_train']:,} | {m['w_xgb']:.1f} | {m['temperature']:.2f} |")
    var = res[res.train_through == res.eval_window].groupby("variant")[["macro_f1", "accuracy", "ece"]].mean()
    var = var.reindex(["xgb", "lgb", "fused", "calibrated"])
    var_lines = ["| variant | mean macro-F1 | mean accuracy | mean ECE |", "|---|---:|---:|---:|"]
    var_lines += [f"| {v} | {r.macro_f1:.4f} | {r.accuracy:.4f} | {r.ece:.4f} |" for v, r in var.iterrows()]
    pc = diag["per_class_f1"]
    pcl = ["| window | " + " | ".join(CLASS_NAMES[target]) + " |", "|---|" + "---:|" * len(CLASS_NAMES[target])]
    for w in range(T):
        pcl.append(f"| {windows[w]} | " + " | ".join(f"{pc.loc[w].get(n, float('nan')):.3f}" if n in pc.loc[w] else "—"
                                                    for n in CLASS_NAMES[target]) + " |")
    lat_lines = ["| model | single-flow P50 (ms) | P95 (ms) | batched (µs/flow) | XGB size (KB) | LGBM size (KB) |",
                 "|---|---:|---:|---:|---:|---:|"]
    for k, v in lat.items():
        lat_lines.append(f"| trained through {k} | {v['single_ms_p50']:.2f} | {v['single_ms_p95']:.2f} | "
                         f"{v['batch_us_per_row']:.2f} | {v['xgb_bytes'] / 1e3:,.0f} | {v['lgb_bytes'] / 1e3:,.0f} |")
    b = ble_res[ble_res.variant == "calibrated"]
    ble_lines = ["| trained through | eval window | macro-F1 | accuracy | AUROC | ECE |", "|---|---|---:|---:|---:|---:|"]
    for _, r in b.iterrows():
        ble_lines.append(f"| {windows[r.train_through]} | {windows[r.eval_window]} | {r.macro_f1:.4f} | {r.accuracy:.4f} | "
                         f"{r.get('auroc_attack_vs_benign', float('nan')):.4f} | {r.ece:.4f} |")
    md = f"""# M3 results — edge ensemble baselines (`{pool}` pool)

Run `{run_name}`; target `{target}` ({len(CLASS_NAMES[target])} classes); timeline `{tl}`; {len(feats)} features (D6: IAT excluded).
Model: XGBoost + LightGBM with balanced class weights (capped at {cfg['models']['weight_cap']:g}), early stopping on a {cfg['models']['val_frac']:.0%} stratified validation split, fusion weight chosen by validation log-loss, temperature-calibrated.

## E1 centralized vs static, on each window's own test set

{chr(10).join(lines)}

- *E1 centralized* = retrained at every window on all clients' train data for windows 0..t (the upper bound).
- *Static* = trained on t0 only and never updated, which quantifies drift degradation.
- Window-averaged macro-F1: **E1 {diag['macro_f1'].mean():.4f}**, **static {static['macro_f1'].mean():.4f}**.
- E1 macro-F1 on the cumulative test set (windows 0..t, row-weighted) per t: {', '.join(f'{windows[t]} {v:.4f}' for t, v in enumerate(cum_rows))}.

![static vs centralized](static_vs_centralized.png)

![matrix](macro_f1_matrix.png)

Each matrix row is one model (trained through window t) and each column is a window's test set. Cells right of the diagonal are future windows the model hasn't seen.

## Ensemble components (averaged over the diagonal)

{chr(10).join(var_lines)}

## Per-class F1 of E1 on each window (— = class not yet present)

{chr(10).join(pcl)}

![confusion](confusion_last_window.png)

## Edge cost

{chr(10).join(lat_lines)}

Measured on a machine with {os.cpu_count()} logical CPUs, using all of them, with the full fused + calibrated ensemble. Single-flow latency includes Python call overhead.

## BLE edge model (binary, separate feature space: {', '.join(ble_features(cfg))})

{chr(10).join(ble_lines)}
"""
    (rep / "RESULTS.md").write_text(md, encoding="utf-8")
    (rep / "latency.json").write_text(dumps(lat, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
