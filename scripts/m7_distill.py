"""M7: DA-FedDistill (E3) and the plain FedMD-style baseline.

At each window, following the proposal:
  1. each client updates locally on new rows + its rehearsal buffer (exactly M6's local_rehearsal)
  2. each client sends p_xgb and p_lgb on the D_pub rows of classes introduced so far
  3. the cloud aggregates them into pseudo-labels (mean or reliability-weighted mean)
  4. the cloud fits g_t(p_xgb, p_lgb, drift features) -> p_global on those pseudo-labels
  5. the cloud sends g_t and gate thresholds back; clients apply them at inference

The variants below share one set of local models, so every comparison with local-only is paired.
The headline `e3` was fixed before any result: mean, class-balanced, with drift features. D_pub's true
labels are read only for the pseudo-label diagnostics and never reach a fit.
FedMD trains separate models: it adds trees on the consensus (digest), then continues locally.

Two additions the proposal doesn't name. Each client fits a temperature on its own validation rows after g_t,
because g_t's soft target leaves it underconfident; this can't change a predicted class, so macro-F1 is unaffected.
And e3_guard, a variant and not the headline: a client keeps g_t only if it beats its own model on those rows.

Usage: python scripts/m7_distill.py [--pool dev|full] [--env ...] [--resume] [--tag seed43] [--only e3|fedmd] [--report]
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
from sklearn.metrics import f1_score

from ids.data.io import model_features
from ids.data.schema import CATEGORIES
from ids.drift.gate import escalate, frontier, gate_outcome, tune
from ids.drift.shift import psi_share
from ids.eval.harness import Evaluator, client_clusters, load_test_sets, predict_by_client
from ids.eval.plots import fig_curves, fig_matrix
from ids.fl.distill import (GlobalCalibrator, accept_calibrator, aggregate_soft, apply_temperatures, dequantize,
                            drift_signals, fit_client_temperature, pseudo_label_weights, quantize, reliability_from_val,
                            uplink_bytes)
from ids.fl.rehearsal import RehearsalBuffer
from ids.models.edge import EdgeConfig, val_split
from ids.models.incremental import IncrementalEdge
from ids.partition.build import SRC_CSV
from ids.partition.io import load_dpub, load_rows
from ids.utils import Run, load_config, rng, set_global_seed
from ids.utils.config import REPO_ROOT
from ids.utils.run import memory_note
from ids.utils.jsonl import dumps

HEADLINE = "csv_only"    # D7
LOCAL = "local_rehearsal"
VARIANTS = {             # aggregation, class-balanced on pseudo-labels, drift features, 8-bit uplink
    "e3": ("mean", True, True, False),
    "e3_reliability": ("reliability", True, True, False),
    "e3_unweighted": ("mean", False, True, False),
    "e3_no_drift": ("mean", True, False, False),
    "e3_q8": ("mean", True, True, True),
}
LABELS = {"e3": "E3 DA-FedDistill (mean, balanced, drift)", "e3_reliability": "E3, reliability-weighted", "e3_unweighted": "E3, no class balance",
          "e3_no_drift": "E3, no drift features", "e3_q8": "E3, 8-bit uplink", "e3_guard": "E3 + client guard (addition)", "fedmd": "FedMD-style digest", LOCAL: "Local + rehearsal (M6)",
          "naive_incremental": "Naive incremental (M6)", "E1": "E1 centralized (M3)", "static": "Static centralized (M3)"}


def edge_cfg(cfg: dict, n_classes: int) -> EdgeConfig:
    mc, ic = cfg["models"], cfg["incremental"]
    cap = {"max_delta_step": ic["max_delta_step"]}
    return EdgeConfig(n_classes=n_classes, xgb_params={**mc["xgb"], **cap, "n_estimators": ic["first_trees"]},
                      lgb_params={**mc["lgb"], **cap, "n_estimators": ic["first_trees"]},
                      early_stopping=mc["early_stopping"], weight_cap=mc["weight_cap"],
                      n_jobs=cfg["resources"]["n_jobs"], seed=cfg["seed"])


class Clients:
    """The client side, keyed so that the models reproduce M6's local_rehearsal."""

    def __init__(self, cfg, window_rows):
        self.cfg, self.window_rows, self.seed = cfg, window_rows, cfg["seed"]
        self.ids = sorted({c for rows in window_rows.values() for c in rows})
        n = len(CATEGORIES)
        self.models = {c: IncrementalEdge(edge_cfg(cfg, n), cfg["incremental"]["new_trees"]) for c in self.ids}
        self.buffers = {c: RehearsalBuffer(cfg["incremental"]["rehearsal_capacity"], rng(self.seed, "rehearsal", LOCAL, c)) for c in self.ids}
        self.baseline: dict[int, list[np.ndarray]] = {c: [] for c in self.ids}
        self.reliability = np.zeros((len(self.ids), n))
        self.shift = np.zeros(max(self.ids) + 1)
        self.val: dict[int, dict] = {}

    def _sample(self, X, key):
        cap = self.cfg["drift"]["shift_max_rows"]
        return X if len(X) <= cap else X[rng(self.seed, "m7_shift", *key).choice(len(X), cap, replace=False)]

    def allowed_features(self, c, t, X, y):
        """All features here; M9 narrows this to the top-K."""
        return None

    def step(self, t: int) -> None:
        dc = self.cfg["drift"]
        for i, c in enumerate(self.ids):
            Xn, yn = self.window_rows[t][c]
            if len(self.buffers[c]):
                bx, by = self.buffers[c].get()
                X, y = np.concatenate([Xn, bx]), np.concatenate([yn, by])
            else:
                X, y = Xn, yn
            vm = val_split(y, self.cfg["models"]["val_frac"], self.seed, ("m6", LOCAL, c, t))
            self.models[c].update(X[~vm], y[~vm], X[vm], y[vm], allowed=self.allowed_features(c, t, X[~vm], y[~vm]))
            self.buffers[c].add(Xn, yn)
            comps = self.models[c].predict_components(X[vm])
            self.reliability[i] = reliability_from_val(y[vm], comps["calibrated"].argmax(1), len(CATEGORIES))
            self.val[c] = {"y": y[vm], **{k: comps[k].astype(np.float32) for k in ("xgb", "lgb", "calibrated")}}
            cur = self._sample(Xn, ("cur", c, t))
            if self.baseline[c]:
                self.shift[c] = psi_share(np.concatenate(self.baseline[c]), cur, dc["psi_alert"], dc["shift_bins"])
            self.baseline[c].append(self._sample(Xn, ("base", c, t)))

    def public_predictions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """(clients, rows, classes) arrays of what each client computes on the public rows."""
        comps = [self.models[c].predict_components(X) for c in self.ids]
        return {k: np.stack([p[k] for p in comps]).astype(np.float32) for k in ("xgb", "lgb", "calibrated")}


def macro_f1(y, p) -> float:
    return float(f1_score(y, p.argmax(1), labels=np.unique(y), average="macro", zero_division=0))


def _apply_g(cal, name, px, pl, client_shift) -> np.ndarray:
    use_drift, q8 = VARIANTS[name][2], VARIANTS[name][3]
    if q8:
        px, pl = dequantize(quantize(px)), dequantize(quantize(pl))
    return cal.predict(px, pl, drift_signals(px, pl, client_shift) if use_drift else None)


def fit_variants(cfg, t, clients, pub, y_pub, only=None) -> tuple:
    """Steps 3 and 4 for every variant (or those named in only), on one shared sample of (client, public row) pairs."""
    dc = cfg["distill"]
    k, ids = len(CATEGORIES), clients.ids
    n_rows, n_cl = pub["xgb"].shape[1], len(ids)
    pairs = n_rows * n_cl
    take = rng(cfg["seed"], "m7_calib", t).choice(pairs, min(pairs, dc["calibrator_max_rows"]), replace=False)
    ci, ri = take // n_rows, take % n_rows
    shift = clients.shift[np.asarray(ids)][ci]
    cons = {"mean": aggregate_soft(pub["calibrated"]), "reliability": aggregate_soft(pub["calibrated"], clients.reliability)}
    cons8 = aggregate_soft(dequantize(quantize(pub["calibrated"])))
    px, pl = pub["xgb"][ci, ri], pub["lgb"][ci, ri]
    px8, pl8 = dequantize(quantize(px)), dequantize(quantize(pl))
    cals = {}
    for name, (agg, balanced, use_drift, q8) in VARIANTS.items():
        if only and name not in only:
            continue
        q = cons8 if q8 else cons[agg]
        a, b = (px8, pl8) if q8 else (px, pl)
        extra = drift_signals(a, b, shift) if use_drift else None
        w = pseudo_label_weights(q, cfg["models"]["weight_cap"])[ri] if balanced else None
        cal = GlobalCalibrator(k, 3 if use_drift else 0, dc["calibrator_l2"], dc["calibrator_max_iter"])
        cals[name] = cal.fit(a, b, extra, q[ri], w)

    # Back on the clients: each fits its temperature for each variant on its own validation rows.
    size = max(ids) + 1
    temps = {name: np.ones(size) for name in cals}
    accept = np.zeros(size, bool)
    for c in ids:
        v = clients.val[c]
        for name, cal in cals.items():
            pv = _apply_g(cal, name, v["xgb"], v["lgb"], np.full(len(v["y"]), clients.shift[c]))
            temps[name][c] = fit_client_temperature(pv, v["y"])
            if name == "e3":
                accept[c] = accept_calibrator(v["y"], v["calibrated"], pv)

    # The thresholds the cloud sends: most disagreement with the consensus caught, within the escalation budget.
    g = cfg["gate"]
    pg = apply_temperatures(cals["e3"].predict(px, pl, drift_signals(px, pl, shift)), temps["e3"][np.asarray(ids)][ci])
    front = frontier(pg.max(1), 0.5 * np.abs(px - pl).sum(1), pg.argmax(1) != cons["mean"][ri].argmax(1),
                     np.round(np.arange(0.40, 1.00, 0.02), 2), [float(x) for x in g["tune_tau_d"]])
    tau = tune(front, g["max_escalation_rate"])

    # Diagnostics only: this is the one place D_pub's true labels are read.
    diag = {"train_through": t, "n_pub": n_rows,
            "pseudo_mean": macro_f1(y_pub, cons["mean"]), "pseudo_reliability": macro_f1(y_pub, cons["reliability"]),
            "local_on_pub": float(np.mean([macro_f1(y_pub, pub["calibrated"][i]) for i in range(n_cl)]))}
    e3_pub = []
    for i, c in enumerate(ids):
        e = drift_signals(pub["xgb"][i], pub["lgb"][i], np.full(n_rows, clients.shift[c]))
        e3_pub.append(macro_f1(y_pub, cals["e3"].predict(pub["xgb"][i], pub["lgb"][i], e)))
    diag["e3_on_pub"] = float(np.mean(e3_pub))
    diag["mean_temperature"] = float(temps["e3"][np.asarray(ids)].mean())
    diag["guard_accepted"] = int(accept[np.asarray(ids)].sum())
    return cals, temps, accept, tau, diag


def variant_predictor(name, cal, memo, shift_lut, temps, accept=None):
    """accept given = the guard: clients that rejected g_t keep their own calibrated output."""
    def predict(X, client):
        mem = memo[id(X)]
        p = apply_temperatures(_apply_g(cal, name, mem["xgb"], mem["lgb"], shift_lut[client]), temps[client])
        if accept is not None:
            p = np.where(accept[client][:, None], p, mem["calibrated"])
        return {"global": p}
    return predict


def communication(t, pub, cals, ids) -> dict:
    k = len(CATEGORIES)
    n = pub["xgb"].shape[1]
    up = [uplink_bytes([pub["xgb"][i], pub["lgb"][i]], metadata_bytes=4 * (k + 4)) for i in range(len(ids))]   # F1 per class, w, T, PSI share, T after g_t
    fm = [uplink_bytes([pub["calibrated"][i]]) for i in range(len(ids))]
    cons = uplink_bytes([aggregate_soft(pub["calibrated"])])
    return {"train_through": t, "n_pub": n, "clients": len(ids),
            **{f"e3_up_{m}": float(np.mean([u[f"{m}_bytes"] for u in up])) for m in ("raw", "q8", "q8_zlib")},
            "e3_down": len(cals["e3"].to_bytes()) + 8,          # + tau_c, tau_d
            **{f"fedmd_up_{m}": float(np.mean([u[f"{m}_bytes"] for u in fm])) for m in ("raw", "q8", "q8_zlib")},
            **{f"fedmd_down_{m}": cons[f"{m}_bytes"] for m in ("raw", "q8", "q8_zlib")}}


def run_e3(cfg, window_rows, X_pub, y_pub, intro, ev, log) -> dict:
    clients = Clients(cfg, window_rows)
    out = {"comm": [], "diag": [], "per_client": [], "gate": [], "calibrators": [], "fit": []}
    test_sets = ev.test_sets
    for t in sorted(window_rows):
        t0 = time.perf_counter()
        clients.step(t)
        t1 = time.perf_counter()
        log.info("[e3] t%d local updates done: %s", t, memory_note())
        m = intro <= t
        pub = clients.public_predictions(X_pub[m])
        cals, temps, accept, tau, diag = fit_variants(cfg, t, clients, pub, y_pub[m])
        t2 = time.perf_counter()
        out["comm"].append(communication(t, pub, cals, clients.ids))
        out["diag"].append(diag)
        out["calibrators"].append({"train_through": t, "tau_c": tau["tau_c"], "tau_d": tau["tau_d"], "client_psi_share": clients.shift.tolist(),
                                   "client_temperature_e3": temps["e3"].round(4).tolist(), "guard_accepted": accept.tolist(),
                                   "W": {n: c.W.round(5).tolist() for n, c in cals.items()}})
        out["fit"].append({"train_through": t, "local_seconds": round(t1 - t0, 2), "cloud_seconds": round(t2 - t1, 2)})
        del pub
        log.info("[e3] t%d cloud fit done: %s", t, memory_note())

        memo: dict[int, dict] = {}
        base = predict_by_client(clients.models)

        def remember(X, client):
            comps = base(X, client)
            memo[id(X)] = {k: comps[k].astype(np.float32) for k in ("xgb", "lgb", "calibrated")}
            return comps

        ev.evaluate(LOCAL, t, remember)
        for name, cal in cals.items():
            ev.evaluate(name, t, variant_predictor(name, cal, memo, clients.shift, temps[name]))
        ev.evaluate("e3_guard", t, variant_predictor("e3", cals["e3"], memo, clients.shift, temps["e3"], accept))

        predict_e3 = variant_predictor("e3", cals["e3"], memo, clients.shift, temps["e3"])
        for w in [t, t + 1]:
            if w not in test_sets:
                continue
            ts = test_sets[w]
            s = ts.source == SRC_CSV
            pg, pc, mem = predict_e3(ts.X, ts.client)["global"][s], memo[id(ts.X)]["calibrated"][s], memo[id(ts.X)]
            dis = 0.5 * np.abs(mem["xgb"][s] - mem["lgb"][s]).sum(1)
            for who, p in (("e3", pg), (LOCAL, pc)):
                out["gate"].append({"train_through": t, "eval_window": w, "method": who, "tau_c": tau["tau_c"], "tau_d": tau["tau_d"], "error_rate": float((p.argmax(1) != ts.y[s]).mean()),
                                    **gate_outcome(escalate(p.max(1), dis, tau["tau_c"], tau["tau_d"]), p.argmax(1) != ts.y[s])})
            if w == t:
                cl = ts.client[s]
                for c in clients.ids:
                    r = cl == c
                    if r.any():
                        out["per_client"].append({"train_through": t, "client": c, "n": int(r.sum()), "local": macro_f1(ts.y[s][r], pc[r]), "e3": macro_f1(ts.y[s][r], pg[r])})
        ts = test_sets[t]
        tic = time.perf_counter()
        predict_e3(ts.X, ts.client)
        out.setdefault("calib_us_per_flow", []).append((time.perf_counter() - tic) / len(ts.X) * 1e6)
        memo.clear()
        log.info("[e3] through t%d: local %.1fs, cloud %.1fs, tau_c %.2f tau_d %g, pseudo %.3f, on D_pub local %.3f -> e3 %.3f; %s",
                 t, t1 - t0, t2 - t1, tau["tau_c"], tau["tau_d"], diag["pseudo_mean"], diag["local_on_pub"], diag["e3_on_pub"], memory_note())
    return out


def run_fedmd(cfg, window_rows, X_pub, intro, ev, log) -> dict:
    clients = Clients(cfg, window_rows)
    rounds, cap = cfg["distill"]["digest_trees"], cfg["models"]["weight_cap"]
    fit = []
    for t in sorted(window_rows):
        t0 = time.perf_counter()
        clients.step(t)
        log.info("[fedmd] t%d local updates done: %s", t, memory_note())
        Xp = X_pub[intro <= t]
        q = aggregate_soft(clients.public_predictions(Xp)["calibrated"])
        w = pseudo_label_weights(q, cap)
        for c in clients.ids:
            clients.models[c].digest(Xp, q, w, rounds)
        secs = time.perf_counter() - t0
        log.info("[fedmd] t%d digest done: %s", t, memory_note())
        ev.evaluate("fedmd", t, predict_by_client(clients.models))
        fit.append({"train_through": t, "seconds": round(secs, 2), "mean_model_bytes": float(np.mean([m.size_bytes() for m in clients.models.values()]))})
        log.info("[fedmd] through t%d: %.1fs; %s", t, secs, memory_note())
    return {"fit": fit}


def table_scores(rows, label) -> str:
    t = "| method | window avg | avg excl. t0 | final (t5) | class forgetting | mean ECE |\n|---|---:|---:|---:|---:|---:|\n"
    for s in rows:
        forget = "n/a" if s["mean_forgetting_by_class"] is None else f"{s['mean_forgetting_by_class']:.4f}"
        ece = f"{s['mean_ece']:.4f}" if s.get("mean_ece") is not None else ""
        t += f"| {label[s['method']]} | {s['window_avg_macro_f1']:.4f} | {s['window_avg_macro_f1_excl_t0']:.4f} | {s['final_macro_f1']:.4f} | {forget} | {ece} |\n"
    return t


def write_report(rep, x) -> None:
    S, R, W, L = x["summaries"], x["refs"], x["windows"], LABELS
    by = {s["method"]: s for s in [*S, *R]}
    t2 = "| method | " + " | ".join(W) + " |\n|---|" + "---:|" * len(W) + "\n"
    for s in [*S, *R]:
        t2 += f"| {L[s['method']]} | " + " | ".join(f"{s['macro_f1_by_window'][i]:.3f}" for i in range(len(W))) + " |\n"
    e3 = by["e3"]["window_avg_macro_f1"]
    t3 = "| E3 minus | window avg | final (t5) |\n|---|---:|---:|\n"
    for m in [LOCAL, "naive_incremental", "E1", "static", "fedmd"]:
        if m in by:
            t3 += f"| {L[m]} | {e3 - by[m]['window_avg_macro_f1']:+.4f} | {by['e3']['final_macro_f1'] - by[m]['final_macro_f1']:+.4f} |\n"
    pc = x["per_class_t5"]
    classes = list(CATEGORIES)
    t4 = "| method | " + " | ".join(classes) + " |\n|---|" + "---:|" * len(classes) + "\n"
    for m, row in pc.items():
        t4 += f"| {L[m]} | " + " | ".join(f"{row.get(c, float('nan')):.3f}" for c in classes) + " |\n"
    D = x["diag"]
    t5 = "| through | public rows | pseudo-labels (mean) | pseudo-labels (reliability) | mean client, local | mean client, g_t |\n|---|---:|---:|---:|---:|---:|\n"
    for d in D:
        t5 += f"| {W[d['train_through']]} | {d['n_pub']:,} | {d['pseudo_mean']:.3f} | {d['pseudo_reliability']:.3f} | {d['local_on_pub']:.3f} | {d['e3_on_pub']:.3f} |\n"
    pcl = x["per_client"]
    t6 = "| window | clients better with g_t | worse | mean change | worst | best | mean client temperature | guard accepted g_t |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for t, g in pcl.groupby("train_through"):
        d = g.e3 - g.local
        t6 += (f"| {W[t]} | {(d > 0).sum()}/{len(d)} | {(d < 0).sum()} | {d.mean():+.4f} | {d.min():+.4f} | {d.max():+.4f} | "
               f"{D[t]['mean_temperature']:.3f} | {D[t]['guard_accepted']}/{len(d)} |\n")
    C = x["comm"]
    t7 = "| through | public rows | E3 up, float32 (MB) | E3 up, 8-bit (MB) | E3 up, 8-bit + zlib (MB) | E3 down (bytes) | FedMD up, 8-bit + zlib (MB) | FedMD down, 8-bit + zlib (MB) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for _, r in C.iterrows():
        t7 += (f"| {W[int(r.train_through)]} | {int(r.n_pub):,} | {r.e3_up_raw / 1e6:.3f} | {r.e3_up_q8 / 1e6:.3f} | {r.e3_up_q8_zlib / 1e6:.3f} | {int(r.e3_down)} | "
               f"{r.fedmd_up_q8_zlib / 1e6:.3f} | {r.fedmd_down_q8_zlib / 1e6:.3f} |\n")
    n_cl = int(C.clients.iloc[0])
    G = x["gate"]
    t8 = "| model through | eval window | τc | τd | method | error rate | escalated | errors caught | accuracy on the rest |\n|---|---|---:|---:|---|---:|---:|---:|---:|\n"
    for r in G.itertuples():
        kept = "n/a" if pd.isna(r.accuracy_not_escalated) else f"{r.accuracy_not_escalated:.3f}"
        caught = "n/a" if pd.isna(r.error_capture) else f"{r.error_capture:.3f}"
        t8 += f"| {W[r.train_through]} | {W[r.eval_window]} | {r.tau_c:.2f} | {r.tau_d:g} | {L[r.method]} | {r.error_rate:.3f} | {r.escalation_rate:.3f} | {caught} | {kept} |\n"
    reg = x["regression"]
    regtxt = ("M6's report for this seed was not found, so the models were not checked against it." if reg is None else
              f"Check against M6: the local models here score {reg['here']:.4f} on window average and {reg['m6']:.4f} in M6's report (difference {reg['here'] - reg['m6']:+.5f}).")
    md = f"""# M7: DA-FedDistill (`{x['pool']}` pool)

Run `{x['run']}`. Generated by `scripts/m7_distill.py`. Seed {x['seed']}. Scores use CSV-only rows (D7), and each row is scored by its own client's model. {regtxt}

All E3 variants and local + rehearsal share the same local models, so differences between them come from g_t alone. FedMD trains its own models, adding {x['digest']} rounds per client per window on the consensus. The headline `e3` was fixed before any result: mean aggregation, class-balanced on pseudo-labels, with drift features.

Two additions the proposal doesn't name. Each client fits a temperature on its own validation rows after g_t, which can't change a predicted class. And `e3_guard`, a variant and not the headline: a client keeps g_t only if it beats the client's own model on those rows, which are also used for its early stopping, so that choice is mildly optimistic.
{x['missing']}
## Results

{table_scores([*S, *R], L)}
![curves](macro_f1_by_window.png)

{t3}
## Macro-F1 per window

{t2}
## F1 per class at t5

{t4}
## Clients: g_t against the client's own model

Macro-F1 on each client's rows of the current window, over the classes present in them.

{t6}
## Pseudo-label quality on D_pub

Macro-F1 against D_pub's true labels. These are read for this table only and never reach a fit.

{t5}
## Communication per client per round

{n_cl} clients, {len(C)} rounds. Whole run: E3 sends {C.e3_up_raw.sum() * n_cl / 1e6:.1f} MB up as float32 and {C.e3_up_q8_zlib.sum() * n_cl / 1e6:.1f} MB as 8-bit + zlib, and {C.e3_down.sum() * n_cl / 1e3:.1f} kB down. FedMD sends {C.fedmd_up_q8_zlib.sum() * n_cl / 1e6:.1f} MB up and {C.fedmd_down_q8_zlib.sum() * n_cl / 1e6:.1f} MB down (8-bit + zlib). The 8-bit uplink's accuracy is the `E3, 8-bit uplink` row above. g_t adds about {x['calib_us']:.1f} µs per flow at inference, batched.

{t7}
## Gate with the cloud's thresholds

The cloud picks τc and τd on D_pub for the most disagreement with the consensus caught, within the {x['budget']:.0%} budget (an assumption, see M5). D_pub's class mix is not a client's, so test rates can leave the budget. Local + rehearsal is shown with the same thresholds, for reference. Next-window rows show the old model facing new traffic.

{t8}
Parameters of every g_t are in `calibrators.json`; per-client and per-class tables are in `per_client.csv` and `per_class.csv`.
"""
    (rep / "DISTILL.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default=None, help="dev | full (default from resources.use_dev_sample)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true", help="reuse finished passes from reports/m7/<pool>/checkpoints")
    ap.add_argument("--tag", default=None, help="write to reports/m7/<pool>/<tag>, e.g. for a repeat with another seed")
    ap.add_argument("--only", choices=["e3", "fedmd"], default=None, help="run one pass, save its checkpoint and stop")
    ap.add_argument("--report", action="store_true", help="train nothing: write the report from the saved checkpoints")
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    pool = args.pool or ("dev" if cfg["resources"]["use_dev_sample"] else "full")
    tl, windows, feats = cfg["timeline"]["headline"], cfg["timeline"]["windows"], model_features(cfg)
    target = cfg["labels"]["headline_target"]
    ec = cfg["eval"]

    with Run(cfg, f"m7_distill_{pool}") as run:
        log = run.logger
        clusters = client_clusters(cfg, pool, tl, ec["n_client_clusters"])
        ev = Evaluator(load_test_sets(cfg, pool, tl, target, feats), CATEGORIES, clusters)
        log.info("test sets loaded: %s", memory_note())
        window_rows = {}
        for w in range(0 if args.report else len(windows)):
            d = load_rows(cfg, pool, tl, windows=[w], split="train", features=feats)
            X, y, cl = d[feats].to_numpy(np.float32), d[target].to_numpy().astype(np.int64), d["client"].to_numpy()
            del d
            window_rows[w] = {int(c): (X[cl == c], y[cl == c]) for c in np.unique(cl)}
        X_pub, y_pub, intro = None, None, None
        if not args.report:
            dp = load_dpub(cfg, pool, tl, features=feats)
            X_pub, y_pub, intro = dp[feats].to_numpy(np.float32), dp[target].to_numpy().astype(np.int64), dp["intro_window"].to_numpy()
            del dp
            log.info("D_pub: %d rows; %s", len(X_pub), memory_note())

        rep = REPO_ROOT / "reports" / "m7" / pool / (args.tag or "")
        ckpt = rep / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        key = {"seed": cfg["seed"], "n_jobs": cfg["resources"]["n_jobs"], "models": cfg["models"], "incremental": cfg["incremental"],
               "distill": cfg["distill"], "gate": cfg["gate"], "drift": cfg["drift"], "features": feats}
        results = {}
        for name, fn in (("e3", lambda: run_e3(cfg, window_rows, X_pub, y_pub, intro, ev, log)),
                         ("fedmd", lambda: run_fedmd(cfg, window_rows, X_pub, intro, ev, log))):
            if args.only and args.only != name:
                continue
            f = ckpt / f"{name}.pkl"
            saved = pickle.loads(f.read_bytes()) if (args.resume or args.report) and f.exists() else None
            if saved and saved["key"] == key:
                ev.records += saved["records"]
                ev.confusions.update(saved["confusions"])
                results[name] = saved["out"]
                log.info("[%s] restored from checkpoint", name)
                continue
            if args.report:
                log.warning("[%s] no usable checkpoint, the report goes without it", name)
                continue
            have = {r["model"] for r in ev.records}
            results[name] = fn()
            f.write_bytes(pickle.dumps({"key": key, "out": results[name], "records": [r for r in ev.records if r["model"] not in have],
                                        "confusions": {k: v for k, v in ev.confusions.items() if k[0] not in have}}))
            gc.collect()
        if args.only:
            log.info("[%s] checkpoint saved, no report in this process", args.only)
            return
        if "e3" not in results:
            raise SystemExit(f"no E3 checkpoint in {ckpt}")

        has_fedmd = "fedmd" in results
        order = [LOCAL, *VARIANTS, "e3_guard", *(["fedmd"] if has_fedmd else [])]
        summaries = []
        for m in order:
            s = ev.summary(m, HEADLINE, eps=ec["recover_eps"])
            d = ev.primary(m, HEADLINE)
            s["mean_ece"] = float(d[d.train_through == d.eval_window]["ece"].mean())
            summaries.append(s)
        all_rows = [ev.summary(m, "all", eps=ec["recover_eps"]) for m in order]

        refs, regression = [], None
        tag_dir = REPO_ROOT / "reports" / "m6" / pool / (args.tag or "")
        if (tag_dir / "summary.json").exists():
            m6 = {s["method"]: s for s in json.loads((tag_dir / "summary.json").read_text(encoding="utf-8"))["summaries"]}
            refs.append(m6["naive_incremental"])
            here = next(s for s in summaries if s["method"] == LOCAL)["window_avg_macro_f1"]
            regression = {"here": here, "m6": m6[LOCAL]["window_avg_macro_f1"]}
            log.info("local models against M6: %.5f vs %.5f", here, regression["m6"])
        m4 = REPO_ROOT / "reports" / "m4" / pool / "summary.json"
        if m4.exists():
            refs += [s for s in json.loads(m4.read_text(encoding="utf-8"))["summaries"] if s["slice"] == HEADLINE]
        for s in refs:
            s["macro_f1_by_window"] = {int(k): v for k, v in s["macro_f1_by_window"].items()}

        e3 = results["e3"]
        per_class = ev.per_class_table().round(6)
        last = per_class[(per_class.train_through == len(windows) - 1) & (per_class.eval_window == len(windows) - 1) & (per_class.slice == HEADLINE)]
        pc_t5 = {m: last[last.model == m].set_index("class")["f1"].to_dict() for m in order if m in (LOCAL, "e3", "e3_guard", "fedmd")}
        comm, gate, per_client = pd.DataFrame(e3["comm"]), pd.DataFrame(e3["gate"]), pd.DataFrame(e3["per_client"])

        ev.long_table().round(6).to_csv(rep / "results.csv", index=False)
        per_class.to_csv(rep / "per_class.csv", index=False)
        ev.confusion_table().to_csv(rep / "confusions.csv", index=False)
        comm.to_csv(rep / "communication.csv", index=False)
        gate.round(6).to_csv(rep / "gate_test.csv", index=False)
        per_client.round(6).to_csv(rep / "per_client.csv", index=False)
        (rep / "calibrators.json").write_text(dumps(e3["calibrators"]), encoding="utf-8")
        (rep / "summary.json").write_text(dumps({"summaries": summaries, "summaries_all_rows": all_rows, "references": refs, "regression": regression,
                                                 "pseudo_label_diagnostics": e3["diag"], "fit_e3": e3["fit"], "fit_fedmd": results["fedmd"]["fit"] if has_fedmd else None, "clusters": clusters}, indent=2), encoding="utf-8")
        shown = {LABELS[s["method"]]: pd.Series(s["macro_f1_by_window"]) for s in [*summaries, *refs] if s["method"] in (LOCAL, "e3", "fedmd", "E1")}
        fig_curves(shown, windows, rep / "macro_f1_by_window.png", "Macro-F1 per window, CSV rows only")
        fig_matrix(ev.matrix("e3", HEADLINE), windows, rep / "macro_f1_matrix.png", f"Train x eval macro-F1, E3 ({pool} pool)")
        write_report(rep, {"pool": pool, "run": run.dir.name, "seed": cfg["seed"], "windows": windows, "summaries": summaries, "refs": refs, "regression": regression,
                           "per_class_t5": pc_t5, "diag": e3["diag"], "per_client": per_client, "comm": comm, "gate": gate, "digest": cfg["distill"]["digest_trees"],
                           "calib_us": float(np.mean(e3["calib_us_per_flow"])), "budget": cfg["gate"]["max_escalation_rate"],
                           "missing": "" if has_fedmd else "\n**The FedMD pass is missing from this report: it did not finish.** The communication table still gives its computed sizes.\n"})
        if has_fedmd:
            shutil.rmtree(ckpt)
        log.info("report -> %s", rep)


if __name__ == "__main__":
    main()
