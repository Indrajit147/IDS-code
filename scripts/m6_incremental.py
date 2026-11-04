"""M6: per-client incremental learning, with and without rehearsal, before any federation.

Each client keeps its own model and sees only its own rows. Methods:
  naive_incremental   continued boosting on the new window only (the baseline H1 names)
  local_rehearsal     continued boosting on the new window + a bounded buffer (the local-only baseline E3 must beat)
  rehearsal_<n>       the same at other buffer sizes
  retrain_local       a fresh model on all of the client's rows so far (the cost/quality reference)

Usage: python scripts/m6_incremental.py [--pool dev|full] [--env ...] [--resume]
Finished methods are saved to reports/m6/<pool>/checkpoints/; --resume skips them if the settings match.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import time

import numpy as np
import pandas as pd

from ids.data.io import model_features
from ids.data.schema import CATEGORIES
from ids.eval.harness import Evaluator, client_clusters, load_test_sets, predict_by_client
from ids.eval.plots import fig_curves, fig_matrix
from ids.fl.rehearsal import RehearsalBuffer
from ids.models.edge import EdgeConfig, val_split
from ids.models.incremental import IncrementalEdge
from ids.partition.io import load_rows
from ids.utils import Run, load_config, rng, set_global_seed
from ids.utils.config import REPO_ROOT
from ids.utils.jsonl import dumps

HEADLINE = "csv_only"    # D7


def edge_cfg(cfg: dict, n_classes: int) -> EdgeConfig:
    mc, ic = cfg["models"], cfg["incremental"]
    cap = {"max_delta_step": ic["max_delta_step"]}
    return EdgeConfig(n_classes=n_classes, xgb_params={**mc["xgb"], **cap, "n_estimators": ic["first_trees"]},
                      lgb_params={**mc["lgb"], **cap, "n_estimators": ic["first_trees"]},
                      early_stopping=mc["early_stopping"], weight_cap=mc["weight_cap"],
                      n_jobs=cfg["resources"]["n_jobs"], seed=cfg["seed"])


def retrain_cfg(cfg: dict, n_classes: int) -> EdgeConfig:
    """Same model settings, with M3's full round limit: a fresh fit isn't limited by the update budget."""
    c, mc = edge_cfg(cfg, n_classes), cfg["models"]
    c.xgb_params["n_estimators"], c.lgb_params["n_estimators"] = mc["xgb"]["n_estimators"], mc["lgb"]["n_estimators"]
    return c


def run_method(cfg, name, capacity, retrain, window_rows, evaluator, log) -> dict:
    """capacity None = no buffer. window_rows: {window: {client: (X, y)}}."""
    seed, frac = cfg["seed"], cfg["models"]["val_frac"]
    n = len(CATEGORIES)
    clients = sorted({c for rows in window_rows.values() for c in rows})
    models = {c: IncrementalEdge(edge_cfg(cfg, n), cfg["incremental"]["new_trees"]) for c in clients}
    buffers = {c: RehearsalBuffer(capacity, rng(seed, "rehearsal", name, c)) for c in clients} if capacity else {}
    seen: dict[int, list] = {c: [] for c in clients}
    log_rows = []
    for t in sorted(window_rows):
        t0 = time.perf_counter()
        for c in clients:
            Xn, yn = window_rows[t][c]
            if retrain:
                seen[c].append((Xn, yn))
                X, y = np.concatenate([a for a, _ in seen[c]]), np.concatenate([b for _, b in seen[c]])
                models[c] = IncrementalEdge(retrain_cfg(cfg, n), cfg["incremental"]["new_trees"])
            elif capacity and len(buffers[c]):
                bx, by = buffers[c].get()
                X, y = np.concatenate([Xn, bx]), np.concatenate([yn, by])
            else:
                X, y = Xn, yn
            vm = val_split(y, frac, seed, ("m6", name, c, t))
            models[c].update(X[~vm], y[~vm], X[vm], y[vm])
            if capacity:
                buffers[c].add(Xn, yn)
        secs = time.perf_counter() - t0
        evaluator.evaluate(name, t, predict_by_client(models))
        trees = [models[c].tree_counts() for c in clients]
        sizes = [models[c].size_bytes() for c in clients]
        log_rows.append({"method": name, "train_through": t, "fit_seconds": round(secs, 2),
                         "mean_xgb_trees": float(np.mean([x["xgb"] for x in trees])), "mean_lgb_trees": float(np.mean([x["lgb"] for x in trees])),
                         "mean_model_bytes": float(np.mean(sizes)), "max_model_bytes": int(np.max(sizes)),
                         "max_buffer_rows": max((len(b) for b in buffers.values()), default=0),
                         "max_buffer_bytes": max((b.peak_bytes for b in buffers.values()), default=0)})
        log.info("[%s] through t%d: %.1fs, mean trees %.0f/%.0f", name, t, secs, log_rows[-1]["mean_xgb_trees"], log_rows[-1]["mean_lgb_trees"])
    return {"log": log_rows}


def write_report(rep, x) -> None:
    S, cost = x["summaries"], x["cost"]
    w = x["windows"]
    label = x["labels"]
    def row(s):
        return (f"| {label.get(s['method'], s['method'])} | {s['window_avg_macro_f1']:.4f} | {s['window_avg_macro_f1_excl_t0']:.4f} | {s['final_macro_f1']:.4f} | "
                + ("n/a" if s["mean_forgetting_by_class"] is None else f"{s['mean_forgetting_by_class']:.4f}") + " |\n")
    t1 = "| method | window avg | avg excl. t0 | final (t5) | class forgetting |\n|---|---:|---:|---:|---:|\n" + "".join(row(s) for s in S)
    if x["refs"]:
        t1 += "".join(row(s) for s in x["refs"])
    t2 = "| method | " + " | ".join(w) + " |\n|---|" + "---:|" * len(w) + "\n"
    for s in [*S, *x["refs"]]:
        t2 += f"| {label.get(s['method'], s['method'])} | " + " | ".join(f"{s['macro_f1_by_window'][i]:.3f}" for i in range(len(w))) + " |\n"
    classes = sorted({c for s in S for c in s["forgetting_by_class"]})
    t3 = "| method | " + " | ".join(classes) + " |\n|---|" + "---:|" * len(classes) + "\n"
    for s in S:
        t3 += f"| {label.get(s['method'], s['method'])} | " + " | ".join(f"{s['forgetting_by_class'].get(c, float('nan')):.3f}" for c in classes) + " |\n"
    last = cost[cost.train_through == cost.train_through.max()].set_index("method")
    tot = cost.groupby("method")["fit_seconds"].sum()
    t4 = "| method | total fit time (s) | mean trees XGB/LGBM at t5 | mean model size at t5 (MB) | largest buffer (rows) | largest buffer (kB) |\n|---|---:|---:|---:|---:|---:|\n"
    for m in x["order"]:
        r = last.loc[m]
        t4 += (f"| {label.get(m, m)} | {tot[m]:.0f} | {r.mean_xgb_trees:.0f} / {r.mean_lgb_trees:.0f} | {r.mean_model_bytes / 1e6:.2f} | "
               f"{int(cost[cost.method == m].max_buffer_rows.max()):,} | {cost[cost.method == m].max_buffer_bytes.max() / 1e3:.0f} |\n")
    md = f"""# M6: incremental local learning and rehearsal (`{x['pool']}` pool)

Run `{x['run']}`. Generated by `scripts/m6_incremental.py`. One seed. Scores use CSV-only rows (D7), and each row is scored by its own client's model.

Every client trains alone. Old classes recur with fresh rows in each later window of this timeline, so even a client without a buffer keeps seeing them. That limits how much forgetting this timeline can show.

Settings: {x['first']} rounds at most for the first fit, {x['new']} per update, leaf output capped at `max_delta_step` = {x['mds']}, buffer {x['cap']:,} rows per client, class-balanced. M3's centralized results, where available, come from `reports/m4/{x['pool']}/`.

## Results

{t1}
![curves](macro_f1_by_window.png)

## Macro-F1 per window

{t2}
## Forgetting per class

F_max minus F_final on the test set of the window that introduced the class.

{t3}
## Cost

{t4}
Per-class and confusion tables are in `per_class.csv` and `confusions.csv`.
"""
    (rep / "INCREMENTAL.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default=None, help="dev | full (default from resources.use_dev_sample)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true", help="reuse finished methods from reports/m6/<pool>/checkpoints")
    ap.add_argument("--tag", default=None, help="write to reports/m6/<pool>/<tag>, e.g. for a repeat with another seed")
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    pool = args.pool or ("dev" if cfg["resources"]["use_dev_sample"] else "full")
    tl, windows, feats = cfg["timeline"]["headline"], cfg["timeline"]["windows"], model_features(cfg)
    target = cfg["labels"]["headline_target"]
    ic, ec = cfg["incremental"], cfg["eval"]

    with Run(cfg, f"m6_incremental_{pool}") as run:
        log = run.logger
        clusters = client_clusters(cfg, pool, tl, ec["n_client_clusters"])
        ev = Evaluator(load_test_sets(cfg, pool, tl, target, feats), CATEGORIES, clusters)
        window_rows = {}
        for w in range(len(windows)):
            d = load_rows(cfg, pool, tl, windows=[w], split="train", features=feats)
            X, y, cl = d[feats].to_numpy(np.float32), d[target].to_numpy().astype(np.int64), d["client"].to_numpy()
            del d
            window_rows[w] = {int(c): (X[cl == c], y[cl == c]) for c in np.unique(cl)}

        cap = ic["rehearsal_capacity"]
        methods = [("naive_incremental", None, False), ("local_rehearsal", cap, False),
                   *[(f"rehearsal_{n}", n, False) for n in ic["rehearsal_sweep"]], ("retrain_local", None, True)]
        rep = REPO_ROOT / "reports" / "m6" / pool / (args.tag or "")
        ckpt = rep / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        key = {"seed": cfg["seed"], "n_jobs": cfg["resources"]["n_jobs"], "models": cfg["models"], "incremental": ic, "features": feats}
        cost = []
        for name, capacity, retrain in methods:
            f = ckpt / f"{name}.pkl"
            saved = pickle.loads(f.read_bytes()) if args.resume and f.exists() else None
            if saved and saved["key"] == key:
                ev.records += saved["records"]
                ev.confusions.update(saved["confusions"])
                cost += saved["log"]
                log.info("[%s] restored from checkpoint", name)
                continue
            rows = run_method(cfg, name, capacity, retrain, window_rows, ev, log)["log"]
            cost += rows
            f.write_bytes(pickle.dumps({"key": key, "log": rows, "records": [r for r in ev.records if r["model"] == name],
                                        "confusions": {k: v for k, v in ev.confusions.items() if k[0] == name}}))
            gc.collect()
        cost = pd.DataFrame(cost)
        order = [m[0] for m in methods]
        summaries = [ev.summary(m, HEADLINE, eps=ec["recover_eps"]) for m in order]
        all_rows = [ev.summary(m, "all", eps=ec["recover_eps"]) for m in order]

        refs = []
        m4 = REPO_ROOT / "reports" / "m4" / pool / "summary.json"
        if m4.exists():
            refs = [s for s in json.loads(m4.read_text(encoding="utf-8"))["summaries"] if s["slice"] == HEADLINE]
            for s in refs:
                s["macro_f1_by_window"] = {int(k): v for k, v in s["macro_f1_by_window"].items()}
        labels = {"naive_incremental": "Naive incremental", "local_rehearsal": f"Local + rehearsal ({cap:,})", "retrain_local": "Retrain per client",
                  **{f"rehearsal_{n}": f"Local + rehearsal ({n:,})" for n in ic["rehearsal_sweep"]},
                  "E1": "E1 centralized (M3)", "static": "Static centralized (M3)"}

        ev.long_table().round(6).to_csv(rep / "results.csv", index=False)
        ev.per_class_table().round(6).to_csv(rep / "per_class.csv", index=False)
        ev.confusion_table().to_csv(rep / "confusions.csv", index=False)
        cost.to_csv(rep / "cost.csv", index=False)
        (rep / "summary.json").write_text(dumps({"summaries": summaries, "summaries_all_rows": all_rows, "references": refs, "clusters": clusters}, indent=2), encoding="utf-8")
        shown = {labels[s["method"]]: pd.Series(s["macro_f1_by_window"]) for s in
                 [*[v for v in summaries if v["method"] in ("naive_incremental", "local_rehearsal", "retrain_local")], *[r for r in refs if r["method"] == "E1"]]}
        fig_curves(shown, windows, rep / "macro_f1_by_window.png", "Macro-F1 per window, CSV rows only")
        fig_matrix(ev.matrix("local_rehearsal", HEADLINE), windows, rep / "macro_f1_matrix.png", f"Train x eval macro-F1, local + rehearsal ({pool} pool)")
        write_report(rep, {"pool": pool, "run": run.dir.name, "windows": windows, "summaries": summaries, "refs": refs, "cost": cost, "labels": labels,
                           "order": order, "first": ic["first_trees"], "new": ic["new_trees"], "mds": ic["max_delta_step"], "cap": cap})
        shutil.rmtree(ckpt)
        log.info("report -> %s", rep)


if __name__ == "__main__":
    main()
