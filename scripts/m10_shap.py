"""M10: TreeSHAP for the E4 carried forward (D11, `e4_added`): local store, global store, SHAP drift signal, cost.

Two steps, each can run in its own process:
  fit   reruns M9's e4_added pass with the K from M9's report, saves each window's client models, and checks the scores
        against M9's report;
  shap  explains the saved models. See README, M10, for the design.

Usage: python scripts/m10_shap.py [--pool dev|full] [--env ...] [--tag seed43] [--set ...] [--only fit|shap]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from m9_e4 import HEADLINE, run_pass   # noqa: E402

from ids.data.io import model_features   # noqa: E402
from ids.data.schema import CATEGORIES   # noqa: E402
from ids.drift.gate import gate_signals   # noqa: E402
from ids.eval.explain import ensemble_shap, ranking_change, top_k   # noqa: E402
from ids.eval.harness import Evaluator, client_clusters, load_test_sets   # noqa: E402
from ids.eval.plots import fig_curves   # noqa: E402
from ids.eval.system import payload_bytes   # noqa: E402
from ids.partition.build import SRC_CSV   # noqa: E402
from ids.partition.io import load_dpub, load_rows   # noqa: E402
from ids.utils import Run, load_config, rng, set_global_seed   # noqa: E402
from ids.utils.config import REPO_ROOT   # noqa: E402
from ids.utils.jsonl import dumps   # noqa: E402
from ids.utils.run import memory_note   # noqa: E402

PASS = "e4_added"   # D11


def fit(cfg, pool, tag, models_dir, rep, log, hook=None) -> dict:
    """models_dir None: keep no models. hook: passed to M9's gate (M11)."""
    tl, feats, target = cfg["timeline"]["headline"], model_features(cfg), cfg["labels"]["headline_target"]
    m9 = json.loads((REPO_ROOT / "reports" / "m9" / pool / (tag or "") / "summary.json").read_text(encoding="utf-8"))
    k = m9["k"]
    ev = Evaluator(load_test_sets(cfg, pool, tl, target, feats), CATEGORIES, client_clusters(cfg, pool, tl, cfg["eval"]["n_client_clusters"]))
    window_rows = {}
    for w in range(len(cfg["timeline"]["windows"])):
        d = load_rows(cfg, pool, tl, windows=[w], split="train", features=feats)
        X, y, cl = d[feats].to_numpy(np.float32), d[target].to_numpy().astype(np.int64), d["client"].to_numpy()
        del d
        window_rows[w] = {int(c): (X[cl == c], y[cl == c]) for c in np.unique(cl)}
    dp = load_dpub(cfg, pool, tl, features=feats)
    X_pub, y_pub, intro = dp[feats].to_numpy(np.float32), dp[target].to_numpy().astype(np.int64), dp["intro_window"].to_numpy()
    del dp
    log.info("rows loaded, K = %d from M9's report: %s", k, memory_note())
    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
    run_pass(cfg, PASS, window_rows, X_pub, y_pub, intro, ev, log, k=k, save=models_dir, hook=hook)
    now = ev.summary(PASS, HEADLINE, eps=cfg["eval"]["recover_eps"])["macro_f1_by_window"]
    then = next(s for s in m9["summaries"] if s["method"] == PASS)["macro_f1_by_window"]
    check = {"k": k, "by_window": {int(t): {"m10": float(now[int(t)]), "m9": float(v), "difference": float(now[int(t)] - v)} for t, v in then.items()}}
    check["largest_difference"] = max(abs(v["difference"]) for v in check["by_window"].values())
    (rep / "fit_check.json").write_text(dumps(check, indent=2), encoding="utf-8")
    log.info("scores against M9's %s: largest difference per window %.6f", PASS, check["largest_difference"])
    return check


def timed(f):
    tic = time.perf_counter()
    out = f()
    return out, time.perf_counter() - tic


def inference(model, X, threads):
    model.xgb_.set_param("nthread", threads)
    model.xgb_.predict(xgb.DMatrix(X, nthread=threads))
    model.lgb_.predict(X, raw_score=True, num_threads=threads)
    model.xgb_.set_param("nthread", model.cfg.n_jobs)


def explain(cfg, pool, models_dir, rep, log) -> None:
    ex, seed, n_jobs = cfg["explain"], cfg["seed"], cfg["resources"]["n_jobs"]
    tl, feats, target = cfg["timeline"]["headline"], model_features(cfg), cfg["labels"]["headline_target"]
    windows = cfg["timeline"]["windows"]
    tests = load_test_sets(cfg, pool, tl, target, feats)
    k = ex["top_k"]

    def halves(t, c):
        """Two disjoint uniform samples of a client's CSV test rows, drawn without labels."""
        ts = tests[t]
        idx = np.flatnonzero((ts.client == c) & (ts.source == SRC_CSV))
        p = rng(seed, "m10_rows", c, t).permutation(idx)
        m = min(ex["rows_per_client"], len(p) // 2)
        return np.sort(p[:m]), np.sort(p[m:2 * m])

    local, glob, drift, cost = [], [], [], []
    prev, ref = None, {}
    for t in range(len(windows)):
        models = pickle.loads((models_dir / f"t{t}.pkl").read_bytes())
        ts = tests[t]
        cur = {}
        for c, model in models.items():
            a, b = halves(t, c)
            if not len(a):
                continue
            X = ts.X[a]
            comps = model.predict_components(X)
            cls = comps["calibrated"].argmax(1)
            out, sec = timed(lambda: ensemble_shap(model, X, cls=cls, threads=n_jobs))
            _, inf_sec = timed(lambda: inference(model, X, n_jobs))
            few = X[: ex["timing_rows"]]
            _, sec1 = timed(lambda: ensemble_shap(model, few, cls=cls[: len(few)], threads=1))
            _, inf1 = timed(lambda: inference(model, few, 1))
            idx, val = top_k(out["phi"], k)
            dis = gate_signals(comps)["disagreement"]
            sizes = [payload_bytes({"top": [[feats[i], round(float(v), 4)] for i, v in zip(idx[r], val[r])],
                                    "probs": [round(float(v), 4) for v in comps["calibrated"][r]], "disagreement": round(float(dis[r]), 4)})
                     for r in range(len(a))]
            local.append(pd.DataFrame({"window": t, "client": c, "row": a, "y": ts.y[a], "pred": cls, "bias": out["bias"],
                                       **{f"f{j + 1}": [feats[i] for i in idx[:, j]] for j in range(k)},
                                       **{f"v{j + 1}": val[:, j] for j in range(k)}, "payload_bytes": sizes,
                                       "margin_agrees": out["margin_argmax"] == cls}))
            imp = np.abs(out["phi"]).mean(0)
            cur[c] = imp
            glob += [{"window": t, "client": c, "feature": f, "mean_abs_shap": float(v)} for f, v in zip(feats, imp)]
            cost.append({"window": t, "client": c, "rows": len(a), "shap_us_per_row": 1e6 * sec / len(a), "infer_us_per_row": 1e6 * inf_sec / len(a),
                         "shap_us_per_row_1thread": 1e6 * sec1 / len(few), "infer_us_per_row_1thread": 1e6 * inf1 / len(few),
                         "xgb_trees": model.tree_counts()["xgb"], "lgb_trees": model.tree_counts()["lgb"]})
            # Drift signal at t-1 -> t: the model through t-1, before any update, on window t's rows against its own window's.
            if prev is not None and c in prev and c in ref:
                old = prev[c]
                new = np.abs(ensemble_shap(old, X, threads=n_jobs)["phi"]).mean(0)
                _, b_prev = halves(t - 1, c)
                row = {"boundary": t, "client": c, "rows_new": len(a), "rows_ref": int(ref[c]["n"])}
                row.update({f"{n}_new": v for n, v in ranking_change(ref[c]["imp"], new, k).items()})
                if len(b_prev):
                    ctl = np.abs(ensemble_shap(old, tests[t - 1].X[b_prev], threads=n_jobs)["phi"]).mean(0)
                    row.update({f"{n}_control": v for n, v in ranking_change(ref[c]["imp"], ctl, k).items()})
                drift.append(row)
        log.info("t%d explained: %d clients, %s", t, len(cur), memory_note())
        ref = {c: {"imp": v, "n": len(halves(t, c)[0])} for c, v in cur.items()}
        prev = models

    local, glob, drift, cost = pd.concat(local, ignore_index=True), pd.DataFrame(glob), pd.DataFrame(drift), pd.DataFrame(cost)
    cloud = glob.groupby(["window", "feature"], as_index=False).mean_abs_shap.mean()
    cloud["rank"] = cloud.groupby("window").mean_abs_shap.rank(ascending=False, method="first").astype(int)
    local.to_parquet(rep / "local.parquet", index=False)
    glob.round(6).to_csv(rep / "global_by_client.csv", index=False)
    cloud.round(6).to_csv(rep / "global.csv", index=False)
    drift.round(6).to_csv(rep / "shap_drift.csv", index=False)
    cost.round(3).to_csv(rep / "cost.csv", index=False)
    by_b = drift.groupby("boundary").mean(numeric_only=True)
    above = drift.assign(above=drift.change_new > drift.change_control).groupby("boundary").above.mean()
    fig_curves({"new window's rows": by_b.change_new, "same window, other rows (noise floor)": by_b.change_control},
               windows, rep / "shap_drift.png", "SHAP ranking change of the model before each boundary, mean over clients", ylabel="1 - Spearman rho")
    summary = {"pool": pool, "seed": seed, "margin_agreement": float(local.margin_agrees.mean()),
               "payload_bytes_mean": float(local.payload_bytes.mean()), "rows_explained": len(local),
               "cost_by_window": cost.groupby("window").mean(numeric_only=True).drop(columns="client").round(2).to_dict("index"),
               "drift_by_boundary": by_b.drop(columns="client").round(4).to_dict("index"), "share_above_control": above.round(3).to_dict(),
               "top10_by_window": {int(t): g.sort_values("rank").feature.head(10).tolist() for t, g in cloud.groupby("window")}}
    (rep / "summary.json").write_text(dumps(summary, indent=2), encoding="utf-8")
    write_report(rep, summary, cloud, windows, k)


def write_report(rep, s, cloud, windows, k) -> None:
    fc = rep / "fit_check.json"
    check = json.loads(fc.read_text(encoding="utf-8")) if fc.exists() else None
    lines = [f"# M10: TreeSHAP explanations of E4 (`{PASS}`, D11), `{s['pool']}` pool", "",
             f"Seed {s['seed']}. Generated by `scripts/m10_shap.py`. Attributions are exact TreeSHAP of w·margin_XGB + (1 − w)·margin_LGBM for the class the edge predicts, on up to the configured number of uniformly sampled CSV test rows per client and window (README, M10).", ""]
    if check:
        lines += [f"**Models against M9:** the rerun `{PASS}` (K = {check['k']}) differs from M9's report by at most {check['largest_difference']:.6f} macro-F1 in any window.", ""]
    lines += [f"**Rows explained:** {s['rows_explained']:,}. **The class with the highest weighted margin is the edge's predicted class** for {s['margin_agreement']:.4f} of them.",
              f"**Escalation payload** (top {k} features with values, six probabilities, disagreement, as compact JSON): {s['payload_bytes_mean']:.0f} bytes on average.", "",
              "## Cost per row (µs), mean over clients", "", "| window | TreeSHAP, 1 thread | inference, 1 thread | TreeSHAP, 4 threads | inference, 4 threads | XGBoost trees | LightGBM trees |", "|---|---:|---:|---:|---:|---:|---:|"]
    for t, r in s["cost_by_window"].items():
        lines.append(f"| {windows[int(t)]} | {r['shap_us_per_row_1thread']:.0f} | {r['infer_us_per_row_1thread']:.1f} | {r['shap_us_per_row']:.0f} | {r['infer_us_per_row']:.1f} | {r['xgb_trees']:.0f} | {r['lgb_trees']:.0f} |")
    lines += ["", "TreeSHAP time includes the margins used to check that the attributions add up.", "",
              "## SHAP drift signal", "", "The model through t−1, before any update, explained on window t's rows and compared with its explanation on window t−1's rows (1 − Spearman ρ of mean |SHAP|, and the share of the top features kept). The control compares two disjoint samples of window t−1, so it is the noise floor. Mean over clients.", "",
              f"| boundary | change, new window | change, control | top-{k} kept, new window | top-{k} kept, control | share of clients above their control |", "|---|---:|---:|---:|---:|---:|"]
    for b, r in s["drift_by_boundary"].items():
        lines.append(f"| {windows[int(b) - 1]} → {windows[int(b)]} | {r['change_new']:.3f} | {r.get('change_control', float('nan')):.3f} | {r['top_overlap_new']:.2f} | {r.get('top_overlap_control', float('nan')):.2f} | {s['share_above_control'][b]:.2f} |")
    lines += ["", "![drift](shap_drift.png)", "", "## Global ranking (cloud: mean over clients of mean |SHAP|), top 10", "", "| rank | " + " | ".join(windows) + " |", "|---|" + "---|" * len(windows)]
    top = s["top10_by_window"]
    for i in range(10):
        lines.append(f"| {i + 1} | " + " | ".join(f"`{top[t][i]}`" if i < len(top[t]) else "" for t in sorted(top)) + " |")
    lines += ["", "Stores: `local.parquet` (top features per row), `global_by_client.csv`, `global.csv`, `shap_drift.csv`, `cost.csv`."]
    (rep / "SHAP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--only", choices=["fit", "shap"], default=None)
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    pool = args.pool or ("dev" if cfg["resources"]["use_dev_sample"] else "full")
    rep = REPO_ROOT / "reports" / "m10" / pool / (args.tag or "")
    rep.mkdir(parents=True, exist_ok=True)
    models_dir = REPO_ROOT / "results" / "m10_models" / pool / (args.tag or "default")
    with Run(cfg, f"m10_shap_{pool}") as run:
        if args.only != "shap":
            fit(cfg, pool, args.tag, models_dir, rep, run.logger)
        if args.only != "fit":
            explain(cfg, pool, models_dir, rep, run.logger)
        run.logger.info("report -> %s", rep)


if __name__ == "__main__":
    main()
