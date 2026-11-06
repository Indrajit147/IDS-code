"""M8: FedAvg and FedProx on an MLP (E2), the classic weight-averaging references.

Same partitions, timeline, test rows, rehearsal buffers and validation rows as E3, so the comparison differs
in how clients collaborate and in the model. One global model scores every row.

The rules below were fixed before any result:
  fedavg               the E2 headline: incremental, each window's rounds on new rows + the rehearsal buffer
  fedavg_static        the same model after t0, never updated ("none" in the proposal's matrix)
  fedprox_mu<m>        the same with FedProx at each mu of fedavg.mu_sweep
  fedprox              the mu with the best mean validation macro-F1 over windows, chosen without test rows; ties go to the lower mu
  fedavg_no_rehearsal  an addition: incremental on the new rows only, since E2's "incremental" leaves this open

Usage: python scripts/m8_fedavg.py [--pool dev|full] [--env ...] [--resume] [--tag seed43] [--only NAME] [--report]
--only runs one pass in this process and saves its checkpoint; --report writes the report from the checkpoints there are.
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
import torch

from ids.data.io import model_features
from ids.data.schema import CATEGORIES
from ids.eval.harness import Evaluator, client_clusters, load_test_sets, predict_global
from ids.eval.plots import fig_curves, fig_matrix
from ids.fl.fedavg import ClientData, Scaler, TabularMLP, run_rounds
from ids.fl.rehearsal import RehearsalBuffer
from ids.models.edge import balanced_weights, val_split
from ids.partition.io import load_rows
from ids.utils import Run, derive_seed, load_config, rng, set_global_seed
from ids.utils.config import REPO_ROOT
from ids.utils.jsonl import dumps
from ids.utils.run import memory_note

HEADLINE = "csv_only"    # D7
LOCAL = "local_rehearsal"   # E3's keys, so buffers and validation rows are the same as E3's


def pass_names(cfg: dict) -> list[str]:
    return ["fedavg", "fedavg_no_rehearsal", *[f"fedprox_mu{m:g}" for m in cfg["fedavg"]["mu_sweep"]]]


def pick_mu(val_by_pass: dict[str, list[float]]) -> str:
    """The FedProx pass with the best mean validation score; ties go to the lower mu."""
    return max(sorted(val_by_pass, key=lambda n: float(n.removeprefix("fedprox_mu"))), key=lambda n: np.mean(val_by_pass[n]))


def run_pass(cfg, name, window_rows, ev, log) -> dict:
    fc, seed, cap = cfg["fedavg"], cfg["seed"], cfg["models"]["weight_cap"]
    mu = float(name.removeprefix("fedprox_mu")) if name.startswith("fedprox_mu") else 0.0
    rehearse = name != "fedavg_no_rehearsal"
    ids = sorted({c for rows in window_rows.values() for c in rows})
    n_features = next(iter(window_rows[0].values()))[0].shape[1]
    model = TabularMLP(n_features, len(CATEGORIES), tuple(fc["hidden"]), derive_seed(seed, "m8_init"))
    buffers = {c: RehearsalBuffer(cfg["incremental"]["rehearsal_capacity"], rng(seed, "rehearsal", LOCAL, c)) for c in ids}
    log_rows = []
    for t in sorted(window_rows):
        tic = time.perf_counter()
        clients = []
        for c in ids:
            if c not in window_rows[t]:
                continue
            Xn, yn = window_rows[t][c]
            if rehearse and len(buffers[c]):
                bx, by = buffers[c].get()
                X, y = np.concatenate([Xn, bx]), np.concatenate([yn, by])
            else:
                X, y = Xn, yn
            vm = val_split(y, cfg["models"]["val_frac"], seed, ("m6", LOCAL, c, t))
            clients.append(ClientData(X[~vm], y[~vm], balanced_weights(y[~vm], cap).astype(np.float32), X[vm], y[vm]))
            if rehearse:
                buffers[c].add(Xn, yn)
        res = run_rounds(model, clients, rounds=fc["rounds_per_window"], patience=fc["patience"], mu=mu, lr=fc["lr"], momentum=fc["momentum"],
                         batch=fc["batch"], epochs=fc["local_epochs"], rng_key=(seed, "m8_round", t))
        secs = time.perf_counter() - tic
        del clients
        ev.evaluate(name, t, predict_global(model))
        log_rows.append({"method": name, "train_through": t, "fit_seconds": round(secs, 2), "rounds_run": res["rounds_run"], "best_round": res["best_round"],
                         "best_val": res["best_val"], "temperature": res["temperature"], "clients": len(ids), "model_bytes": model.size_bytes()})
        log.info("[%s] through t%d: %.1fs, %d rounds, best val %.4f; %s", name, t, secs, res["rounds_run"], res["best_val"], memory_note())
    return {"log": log_rows}


def table_scores(rows, label) -> str:
    t = "| method | window avg | avg excl. t0 | final (t5) | class forgetting | mean ECE |\n|---|---:|---:|---:|---:|---:|\n"
    for s in rows:
        forget = "n/a" if s["mean_forgetting_by_class"] is None else f"{s['mean_forgetting_by_class']:.4f}"
        ece = f"{s['mean_ece']:.4f}" if s.get("mean_ece") is not None else ""
        t += f"| {label[s['method']]} | {s['window_avg_macro_f1']:.4f} | {s['window_avg_macro_f1_excl_t0']:.4f} | {s['final_macro_f1']:.4f} | {forget} | {ece} |\n"
    return t


def write_report(rep, x) -> None:
    S, R, W, L = x["summaries"], x["refs"], x["windows"], x["labels"]
    t2 = "| method | " + " | ".join(W) + " |\n|---|" + "---:|" * len(W) + "\n"
    for s in [*S, *R]:
        t2 += f"| {L[s['method']]} | " + " | ".join(f"{s['macro_f1_by_window'].get(i, float('nan')):.3f}" for i in range(len(W))) + " |\n"
    by = {s["method"]: s for s in [*S, *R]}
    t3 = "| FedAvg minus | window avg | final (t5) |\n|---|---:|---:|\n"
    for m in ["e3", LOCAL, "fedmd", "E1", "fedprox"]:
        if m in by:
            t3 += f"| {L[m]} | {by['fedavg']['window_avg_macro_f1'] - by[m]['window_avg_macro_f1']:+.4f} | {by['fedavg']['final_macro_f1'] - by[m]['final_macro_f1']:+.4f} |\n"
    classes = list(CATEGORIES)
    t4 = "| method | " + " | ".join(classes) + " |\n|---|" + "---:|" * len(classes) + "\n"
    for m, row in x["per_class_t5"].items():
        t4 += f"| {L[m]} | " + " | ".join(f"{row.get(c, float('nan')):.3f}" for c in classes) + " |\n"
    t5 = "| FedProx mu | mean best validation macro-F1 | window avg (test) | final t5 (test) | picked |\n|---:|---:|---:|---:|---|\n"
    for n, v in x["mu_val"].items():
        t5 += f"| {n.removeprefix('fedprox_mu')} | {np.mean(v):.4f} | {by[n]['window_avg_macro_f1']:.4f} | {by[n]['final_macro_f1']:.4f} | {'yes' if n == x['picked'] else ''} |\n"
    C = x["cost"]
    t6 = "| method | through | rounds run | best round | best validation | seconds | MB up + down, per client |\n|---|---|---:|---:|---:|---:|---:|\n"
    for r in C[C.method.isin(["fedavg", x["picked"]])].itertuples():
        t6 += f"| {L[r.method]} | {W[r.train_through]} | {r.rounds_run} | {r.best_round} | {r.best_val:.4f} | {r.fit_seconds:.0f} | {2 * r.rounds_run * r.model_bytes / 1e6:.2f} |\n"
    fa = C[C.method == "fedavg"]
    n_cl, size = int(fa.clients.iloc[0]), int(fa.model_bytes.iloc[0])
    total = 2 * fa.rounds_run.sum() * size * n_cl / 1e6
    md = f"""# M8: FedAvg and FedProx on an MLP (`{x['pool']}` pool)

Run `{x['run']}`. Generated by `scripts/m8_fedavg.py`. Seed {x['seed']}. Scores use CSV-only rows (D7), and one global model scores every row. Same partitions, test rows, rehearsal buffers and validation rows as E3.
{x['missing']}
MLP: hidden layers {x['hidden']}, {size // 4:,} parameters ({size / 1e3:.0f} kB as float32). At most {x['rounds']} rounds per window with patience {x['patience']}, {x['epochs']} local epoch(s), SGD at lr {x['lr']} with momentum {x['momentum']}, batch {x['batch']}, class-balanced loss per client, a temperature fitted on validation rows. All of these are assumptions: the proposal names FedAvg and FedProx with an MLP and sets none of them.

`fedprox` is the mu with the best validation score, chosen without test rows. `fedavg_no_rehearsal` is an addition.

## Results

{table_scores([*S, *R], L)}
![curves](macro_f1_by_window.png)

{t3}
## Macro-F1 per window

{t2}
## F1 per class at t5

{t4}
## FedProx mu sweep

{t5}
## Rounds and communication

FedAvg, whole run: {total:.1f} MB up and down over {n_cl} clients and {int(fa.rounds_run.sum())} rounds ({size / 1e3:.0f} kB each way per client per round), plus {x['scaler_bytes']} bytes per client once for the scaling statistics. Inference: {x['us_per_flow']:.1f} µs per flow, batched, at {x['threads']} threads.

{t6}"""
    (rep / "FEDAVG.md").write_text(md, encoding="utf-8")


def load_refs(pool: str, tag: str | None) -> tuple[list[dict], dict]:
    """E3 and the earlier baselines, from M7's report for this seed."""
    d = REPO_ROOT / "reports" / "m7" / pool / (tag or "")
    refs, pc = [], {}
    if (d / "summary.json").exists():
        m7 = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        refs += [s for s in m7["summaries"] if s["method"] in ("e3", "e3_guard", LOCAL, "fedmd")]
        refs += [s for s in m7["references"]]
        if (d / "per_class.csv").exists():   # seed folders keep only the summary
            p = pd.read_csv(d / "per_class.csv")
            p = p[(p.train_through == 5) & (p.eval_window == 5) & (p["slice"] == HEADLINE) & (p.model == "e3")]
            pc = {"e3": p.set_index("class")["f1"].to_dict()}
    for s in refs:
        s["macro_f1_by_window"] = {int(k): v for k, v in s["macro_f1_by_window"].items()}
    return refs, pc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default=None, help="dev | full (default from resources.use_dev_sample)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true", help="reuse finished passes from reports/m8/<pool>/checkpoints")
    ap.add_argument("--tag", default=None, help="write to reports/m8/<pool>/<tag>, e.g. for a repeat with another seed")
    ap.add_argument("--only", default=None, help="run one pass, save its checkpoint and stop")
    ap.add_argument("--report", action="store_true", help="train nothing: write the report from the saved checkpoints")
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    n_jobs = cfg["resources"]["n_jobs"]
    if n_jobs > 0:
        torch.set_num_threads(n_jobs)
    pool = args.pool or ("dev" if cfg["resources"]["use_dev_sample"] else "full")
    tl, windows, feats = cfg["timeline"]["headline"], cfg["timeline"]["windows"], model_features(cfg)
    target, ec, fc = cfg["labels"]["headline_target"], cfg["eval"], cfg["fedavg"]
    names = pass_names(cfg)
    if args.only and args.only not in names:
        raise SystemExit(f"--only takes one of {names}")

    with Run(cfg, f"m8_fedavg_{pool}") as run:
        log = run.logger
        ev = Evaluator(load_test_sets(cfg, pool, tl, target, feats), CATEGORIES, client_clusters(cfg, pool, tl, ec["n_client_clusters"]))
        log.info("test sets loaded: %s", memory_note())
        window_rows = {}
        for w in range(0 if args.report else len(windows)):
            d = load_rows(cfg, pool, tl, windows=[w], split="train", features=feats)
            X, y, cl = d[feats].to_numpy(np.float32), d[target].to_numpy().astype(np.int64), d["client"].to_numpy()
            del d
            window_rows[w] = {int(c): (X[cl == c], y[cl == c]) for c in np.unique(cl)}
        if not args.report:
            # The statistics come from t0's rows only, so no later window leaks in, and stay fixed so that the model's inputs don't move under it.
            scaler = Scaler.from_clients([x for x, _ in window_rows[0].values()])
            window_rows = {t: {c: (scaler.transform(x), y) for c, (x, y) in rows.items()} for t, rows in window_rows.items()}
            for ts in ev.test_sets.values():
                ts.X[:] = scaler.transform(ts.X)
            log.info("rows loaded and scaled: %s", memory_note())

        rep = REPO_ROOT / "reports" / "m8" / pool / (args.tag or "")
        ckpt = rep / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        key = {"seed": cfg["seed"], "n_jobs": n_jobs, "fedavg": fc, "incremental": cfg["incremental"], "val_frac": cfg["models"]["val_frac"],
               "weight_cap": cfg["models"]["weight_cap"], "features": feats}
        cost = []
        for name in names:
            if args.only and args.only != name:
                continue
            f = ckpt / f"{name}.pkl"
            saved = pickle.loads(f.read_bytes()) if (args.resume or args.report) and f.exists() else None
            if saved and saved["key"] == key:
                ev.records += saved["records"]
                ev.confusions.update(saved["confusions"])
                cost += saved["log"]
                log.info("[%s] restored from checkpoint", name)
                continue
            if args.report:
                log.warning("[%s] no usable checkpoint, the report goes without it", name)
                continue
            rows = run_pass(cfg, name, window_rows, ev, log)["log"]
            cost += rows
            f.write_bytes(pickle.dumps({"key": key, "log": rows, "records": [r for r in ev.records if r["model"] == name],
                                        "confusions": {k: v for k, v in ev.confusions.items() if k[0] == name}}))
            gc.collect()
        if args.only:
            log.info("[%s] checkpoint saved, no report in this process", args.only)
            return

        cost = pd.DataFrame(cost)
        done = list(cost.method.unique())
        if "fedavg" not in done:
            raise SystemExit(f"no FedAvg checkpoint in {ckpt}")
        mu_val = {n: cost[cost.method == n].best_val.tolist() for n in done if n.startswith("fedprox_mu")}
        picked = pick_mu(mu_val) if mu_val else None
        ev.alias("fedavg_static", "fedavg", 0)
        labels = {"fedavg": "E2 FedAvg, incremental", "fedavg_static": "E2 FedAvg, none (t0 only)", "fedavg_no_rehearsal": "FedAvg, no rehearsal (addition)",
                  "fedprox": f"FedProx, incremental (mu {picked.removeprefix('fedprox_mu')})" if picked else "FedProx", "fedprox_static": "FedProx, none (t0 only)",
                  "e3": "E3 DA-FedDistill (M7)", "e3_guard": "E3 + client guard (M7)", LOCAL: "Local + rehearsal (M6)", "fedmd": "FedMD-style digest (M7)",
                  "naive_incremental": "Naive incremental (M6)", "E1": "E1 centralized (M3)", "static": "Static centralized (M3)",
                  **{n: f"FedProx, mu {n.removeprefix('fedprox_mu')}" for n in names if n.startswith("fedprox_mu")}}
        order = ["fedavg", "fedavg_static", *[n for n in done if n == "fedavg_no_rehearsal"]]
        if picked:
            for r in [r for r in ev.records if r["model"] == picked]:
                ev.records.append({**r, "model": "fedprox"})
            ev.confusions.update({("fedprox", *k[1:]): v for k, v in list(ev.confusions.items()) if k[0] == picked})
            ev.alias("fedprox_static", "fedprox", 0)
            order += ["fedprox", "fedprox_static", *mu_val]
        summaries = []
        for m in order:
            s = ev.summary(m, HEADLINE, eps=ec["recover_eps"])
            d = ev.primary(m, HEADLINE)
            s["mean_ece"] = float(d[d.train_through == d.eval_window]["ece"].mean()) if not m.endswith("static") else None
            summaries.append(s)
        all_rows = [ev.summary(m, "all", eps=ec["recover_eps"]) for m in order]
        refs, pc_ref = load_refs(pool, args.tag)

        per_class = ev.per_class_table().round(6)
        last = per_class[(per_class.train_through == len(windows) - 1) & (per_class.eval_window == len(windows) - 1) & (per_class.slice == HEADLINE)]
        pc_t5 = {m: last[last.model == m].set_index("class")["f1"].to_dict() for m in order if m in ("fedavg", "fedavg_no_rehearsal", "fedprox")}
        pc_t5.update(pc_ref)

        ev.long_table().round(6).to_csv(rep / "results.csv", index=False)
        per_class.to_csv(rep / "per_class.csv", index=False)
        ev.confusion_table().to_csv(rep / "confusions.csv", index=False)
        cost.to_csv(rep / "cost.csv", index=False)
        (rep / "summary.json").write_text(dumps({"summaries": summaries, "summaries_all_rows": all_rows, "references": refs, "picked_mu": picked, "mean_val_by_mu": {k: float(np.mean(v)) for k, v in mu_val.items()}}, indent=2), encoding="utf-8")
        shown = {labels[s["method"]]: pd.Series(s["macro_f1_by_window"]) for s in [*summaries, *refs] if s["method"] in ("fedavg", "fedprox", "e3", "E1")}
        fig_curves(shown, windows, rep / "macro_f1_by_window.png", "Macro-F1 per window, CSV rows only")
        fig_matrix(ev.matrix("fedavg", HEADLINE), windows, rep / "macro_f1_matrix.png", f"Train x eval macro-F1, FedAvg ({pool} pool)")

        probe = TabularMLP(len(feats), len(CATEGORIES), tuple(fc["hidden"]), 0)
        Xp = np.zeros((200_000, len(feats)), np.float32)
        tic = time.perf_counter()
        probe.predict_proba(Xp)
        us = (time.perf_counter() - tic) / len(Xp) * 1e6
        write_report(rep, {"pool": pool, "run": run.dir.name, "seed": cfg["seed"], "windows": windows, "summaries": summaries, "refs": refs, "labels": labels,
                           "per_class_t5": pc_t5, "mu_val": mu_val, "picked": picked, "cost": cost, "hidden": fc["hidden"], "rounds": fc["rounds_per_window"], "patience": fc["patience"],
                           "epochs": fc["local_epochs"], "lr": fc["lr"], "momentum": fc["momentum"], "batch": fc["batch"], "scaler_bytes": 8 * (2 * len(feats) + 1),
                           "us_per_flow": us, "threads": torch.get_num_threads(),
                           "missing": "" if set(names) <= set(done) else f"\n**Passes missing from this report: they did not finish: {sorted(set(names) - set(done))}.**\n"})
        if set(names) <= set(done):
            shutil.rmtree(ckpt)
        log.info("report -> %s", rep)


if __name__ == "__main__":
    main()
