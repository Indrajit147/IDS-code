"""M9: E4, drift-triggered updates and top-K features on DA-FedDistill, plus the adaptive gate.

A window's rows reach each client in rounds. E3's fixed schedule updates once, at the end of the window. E4 scores
each round with the current model and feeds ADWIN. A rise triggers an update at once, on the rows held so far, and
another at the end of that window on the rest. A client whose ADWIN stays quiet doesn't update and keeps its rows for
later. Between its update and the next g_t, a client uses its own calibrated output, because the old g_t predates the
new classes. The stream order is random: the dataset has no timestamps. See README, M9, for what is an addition.

Passes: fixed (reproduces E3), fixed_k<K> (the K sweep), e4_schedule (all features), e4_label_free (trigger on
disagreement), e4 (trigger + the K with the best validation score), e4_added (e4, but every client also updates at
the end of every window: the trigger adds to the fixed schedule instead of replacing it).

Usage: python scripts/m9_e4.py [--pool dev|full] [--env ...] [--tag seed43] [--resume] [--only NAME] [--report]
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from m7_distill import LOCAL, Clients, _apply_g, communication, fit_variants, macro_f1, variant_predictor   # noqa: E402

from ids.data.io import model_features   # noqa: E402
from ids.data.schema import CATEGORIES   # noqa: E402
from ids.drift.adwin import Adwin   # noqa: E402
from ids.drift.gate import adaptive_taus, drift_score, drift_state, escalate, gate_outcome, gate_signals, same_cost_mask   # noqa: E402
from ids.drift.shift import psi_share   # noqa: E402
from ids.eval.harness import Evaluator, client_clusters, load_test_sets, predict_by_client   # noqa: E402
from ids.eval.plots import fig_curves   # noqa: E402
from ids.fl.distill import apply_temperatures, reliability_from_val   # noqa: E402
from ids.models.edge import val_split   # noqa: E402
from ids.partition.build import SRC_CSV   # noqa: E402
from ids.partition.io import load_dpub, load_rows   # noqa: E402
from ids.utils import Run, load_config, rng, set_global_seed   # noqa: E402
from ids.utils.config import REPO_ROOT   # noqa: E402
from ids.utils.jsonl import dumps   # noqa: E402
from ids.utils.run import memory_note   # noqa: E402
from ids.utils.seed import derive_seed   # noqa: E402

HEADLINE = "csv_only"    # D7
GATE_PASSES = ("fixed", "e4_schedule", "e4_label_free", "e4", "e4_added")


def pass_names(cfg: dict) -> list[str]:
    return ["fixed", *[f"fixed_k{k}" for k in cfg["schedule"]["k_sweep"]], "e4_schedule", "e4_label_free", "e4", "e4_added"]


def pick_k(val_by_k: dict[int, float]) -> int:
    """The K with the best mean validation macro-F1; ties go to the smaller K."""
    return max(sorted(val_by_k), key=lambda k: val_by_k[k])


def pick_strength(rows: pd.DataFrame, target: float) -> float:
    """Most errors caught, on validation rows of the next window, with mean escalation within the target. Ties go to the weaker;
    0, the fixed gate, is the fallback when nothing fits."""
    g = rows.groupby("strength")[["error_capture", "escalation_rate"]].mean().sort_index()
    ok = g[g.escalation_rate <= target]
    return 0.0 if ok.empty else float(ok.error_capture.idxmax())


def recovery_curve(y, cold, versions: list[tuple[np.ndarray, int, np.ndarray]], final, chunks: int) -> list[float]:
    """Macro-F1 on one window's test rows after j = 0..chunks rounds. A version is (mask of the client's rows, round it takes effect, predictions)."""
    labels, out = np.unique(y), []
    for j in range(chunks):
        p = cold.copy()
        for mask, start, pred in versions:
            if start <= j:
                p[mask] = pred
        out.append(float(f1_score(y, p, labels=labels, average="macro", zero_division=0)))
    out.append(float(f1_score(y, final, labels=labels, average="macro", zero_division=0)))
    return out


def recovery_share(curve: list[float], pre: float, eps: float) -> float | None:
    """Share of the window's rounds until the score is back within eps of pre; None if it isn't by the end of the window."""
    n = len(curve) - 1
    return next((j / n for j, v in enumerate(curve) if v >= pre - eps), None)


class StreamClients(Clients):
    def __init__(self, cfg, window_rows, k=None, trigger=None, end_all=False):
        super().__init__(cfg, window_rows)
        self.k, self.trigger, self.end_all = k, trigger, end_all
        self.sc = cfg["schedule"]
        self.top: list[dict] = []
        self.pending = {c: [] for c in self.ids}
        self.detector = {c: None for c in self.ids}
        self.updates: list[dict] = []
        self.updated_now: set[int] = set()

    def allowed_features(self, c, t, X, y):
        if self.k is None:
            return None
        take = []
        r = rng(self.seed, "m9_mi", c, t, self.n_updates(c))
        for cl in np.unique(y):
            idx = np.flatnonzero(y == cl)
            take.append(idx if len(idx) <= self.sc["mi_rows_per_class"] else r.choice(idx, self.sc["mi_rows_per_class"], replace=False))
        take = np.concatenate(take)
        mi = mutual_info_classif(X[take], y[take], random_state=derive_seed(self.seed, "m9_mi_knn", c, t))
        top = np.sort(np.argsort(-mi, kind="stable")[: self.k])
        self.top.append({"train_through": t, "client": c, "features": top.tolist()})
        return top

    def n_updates(self, c) -> int:
        return self.models[c].n_updates

    def fit_on(self, c, t, i, Xn, yn, key) -> None:
        """One local update on rows Xn + the buffer, then ADWIN restarts from this model's own level on new validation rows."""
        if len(self.buffers[c]):
            bx, by = self.buffers[c].get()
            X, y = np.concatenate([Xn, bx]), np.concatenate([yn, by])
        else:
            X, y = Xn, yn
        vm = val_split(y, self.cfg["models"]["val_frac"], self.seed, key)
        self.models[c].update(X[~vm], y[~vm], X[vm], y[vm], allowed=self.allowed_features(c, t, X[~vm], y[~vm]))
        self.buffers[c].add(Xn, yn)
        comps = self.models[c].predict_components(X[vm])
        self.reliability[i] = reliability_from_val(y[vm], comps["calibrated"].argmax(1), len(CATEGORIES))
        self.val[c] = {"y": y[vm], **{n: comps[n].astype(np.float32) for n in ("xgb", "lgb", "calibrated")}}
        self.pending[c] = []
        self.updated_now.add(c)
        new = int(vm[:len(Xn)].sum())
        ref = {n: comps[n][:new] for n in ("xgb", "lgb", "calibrated")}
        self.detector[c] = Adwin(self.cfg["drift"]["adwin_delta"])
        order = rng(self.seed, "m9_ref", c, t, self.n_updates(c)).permutation(new)[: self.sc["reference_rows"]]
        for v in self.signal(ref, y[vm][:new])[order].tolist():
            self.detector[c].update(v)

    def signal(self, comps, y) -> np.ndarray:
        """Error rate needs the labels these rows come with; disagreement doesn't."""
        return (comps["calibrated"].argmax(1) != y).astype(float) if self.trigger == "error" else gate_signals(comps)["disagreement"]

    def rise_in(self, c, comps, y) -> bool:
        d = self.detector[c]
        return any(d.update(v) and d.last_change > 0 for v in self.signal(comps, y).tolist())

    def run_window(self, t, on_trigger) -> None:
        """Rows arrive round by round. on_trigger(c, round) reports each early update."""
        self.updated_now = set()
        for i, c in enumerate(self.ids):
            Xn, yn = self.window_rows[t][c]
            if t == 0:      # the first fit has nothing to trigger it
                self.fit_on(c, t, i, Xn, yn, ("m6", LOCAL, c, t))
                self.updates.append({"train_through": t, "client": c, "round": None, "rows": len(Xn)})
                continue
            order = rng(self.seed, "m9_order", c, t).permutation(len(Xn))
            fired = False
            for j, idx in enumerate(np.array_split(order, self.sc["chunks"])):
                self.pending[c].append((Xn[idx], yn[idx]))
                if fired:
                    continue
                if self.rise_in(c, self.models[c].predict_components(Xn[idx]), yn[idx]):
                    fired = True
                    self._fit_pending(c, t, i, j)
                    on_trigger(c, j)
            if (fired or self.end_all) and self.pending[c]:
                self._fit_pending(c, t, i, None)

    def _fit_pending(self, c, t, i, j) -> None:
        X = np.concatenate([x for x, _ in self.pending[c]])
        y = np.concatenate([v for _, v in self.pending[c]])
        self.fit_on(c, t, i, X, y, ("m9", c, t, "end" if j is None else j))
        self.updates.append({"train_through": t, "client": c, "round": j, "rows": len(X)})

    def fixed_window(self, t) -> None:
        self.updated_now = set(self.ids)
        self.step(t)
        self.updates += [{"train_through": t, "client": c, "round": None, "rows": len(self.window_rows[t][c][0])} for c in self.ids]

    def update_shift(self, t) -> None:
        dc = self.cfg["drift"]
        for c in self.ids:
            cur = self._sample(self.window_rows[t][c][0], ("cur", c, t))
            if self.baseline[c]:
                self.shift[c] = psi_share(np.concatenate(self.baseline[c]), cur, dc["psi_alert"], dc["shift_bins"])
            self.baseline[c].append(self._sample(self.window_rows[t][c][0], ("base", c, t)))


def client_streams(cfg, models, ids, rows_pre, rows_post, carry, key) -> dict:
    """Per client, in a random order: the label-free score and the hold flag for each post-boundary row.

    rows_* are {client: X}. Returns {client: (order, score, hold)} and the score each client carries forward."""
    delta, hold = cfg["drift"]["adwin_delta"], cfg["gate"]["drift_hold_rows"]
    order_sig = cfg["gate"]["drift_signal"]
    out, new_carry = {}, {}
    for c in ids:
        Xb, Xa = rows_pre.get(c), rows_post.get(c)
        if Xa is None or not len(Xa):
            new_carry[c] = carry.get(c, 0.0)
            continue
        ib = rng(cfg["seed"], "m9_stream", key, c, "pre").permutation(len(Xb)) if Xb is not None and len(Xb) else np.array([], int)
        ia = rng(cfg["seed"], "m9_stream", key, c, "post").permutation(len(Xa))
        sa = gate_signals(models[c].predict_components(Xa[ia]))
        sb = gate_signals(models[c].predict_components(Xb[ib])) if len(ib) else {n: np.array([]) for n in sa}
        names = [order_sig, *[n for n in ("disagreement", "entropy") if n != order_sig]]
        run, fires = drift_score([np.concatenate([sb[n], sa[n]]) for n in names], len(ib), delta)
        out[c] = (ia, np.maximum(run, carry.get(c, 0.0)), drift_state(len(ia), fires, hold))
        new_carry[c] = float(run[-1])
    return {"streams": out, "carry": new_carry}


def run_pass(cfg, name, window_rows, X_pub, y_pub, intro, ev, log, k=None) -> dict:
    sc, trigger = cfg["schedule"], {"e4": "error", "e4_added": "error", "e4_schedule": "error", "e4_label_free": "disagreement"}.get(name)
    clients = StreamClients(cfg, window_rows, k=k, trigger=trigger, end_all=name == "e4_added")
    out = {"comm": [], "gate": [], "gate_val": [], "curves": [], "fit": [], "val": [], "k": k}
    tests = ev.test_sets
    carry_test, carry_val = {}, {}
    prior_pub = 0
    cold_prev = None
    for t in sorted(window_rows):
        tic = time.perf_counter()
        # Choose the strength on rows of this window that the current models haven't trained on yet.
        if t and name in GATE_PASSES:
            carry_val = gate_on_validation(cfg, clients, t, tau, cals["e3"], temps["e3"], carry_val, out, log)
        versions, ts = [], tests[t]
        csv = ts.source == SRC_CSV

        def on_trigger(c, j, ts=ts, csv=csv):
            mask = ts.client[csv] == c
            if mask.any():
                versions.append((mask, j + 1, clients.models[c].predict_components(ts.X[csv][mask])["calibrated"].argmax(1).astype(np.int8)))

        if trigger is None:
            clients.fixed_window(t)
        else:
            clients.run_window(t, on_trigger)
            clients.update_shift(t)
        t1 = time.perf_counter()
        log.info("[%s] t%d local done, %d of %d clients updated: %s", name, t, len(clients.updated_now), len(clients.ids), memory_note())

        m = intro <= t
        pub = clients.public_predictions(X_pub[m])
        cals, temps, accept, tau, _ = fit_variants(cfg, t, clients, pub, y_pub[m], only=("e3",))
        n_pub = int(m.sum())
        comm = communication(t, pub, cals, clients.ids)
        del pub
        # A client that didn't update has already sent its predictions for the older public rows.
        # shortcut: 8-bit + zlib sizes are scaled by row share, not recompressed; the float32 size is exact.
        share = [1.0 if c in clients.updated_now else (n_pub - prior_pub) / n_pub for c in clients.ids]
        out["comm"].append({"train_through": t, "n_pub": n_pub, "clients": len(clients.ids), "clients_updated": len(clients.updated_now),
                            "up_raw": comm["e3_up_raw"] * sum(share), "up_q8_zlib": comm["e3_up_q8_zlib"] * sum(share),
                            "down": (comm["e3_down"] + 4) * len(clients.ids)})   # + the drift score
        prior_pub = n_pub
        t2 = time.perf_counter()

        memo: dict[int, dict] = {}
        base = predict_by_client(clients.models)

        def remember(X, client):
            comps = base(X, client)
            memo[id(X)] = {n: comps[n].astype(np.float32) for n in ("xgb", "lgb", "calibrated")}
            return comps

        ev.evaluate(f"{name}_local", t, remember)
        predict = variant_predictor("e3", cals["e3"], memo, clients.shift, temps["e3"])
        ev.evaluate(name, t, predict)
        ev.evaluate(f"{name}_guard", t, variant_predictor("e3", cals["e3"], memo, clients.shift, temps["e3"], accept))

        final = predict(ts.X, ts.client)["global"][csv].argmax(1).astype(np.int8)
        if cold_prev is not None:
            out["curves"].append({"window": t, "curve": recovery_curve(ts.y[csv], cold_prev, versions, final, sc["chunks"]),
                                  "triggered": sorted(start for _, start, _ in versions)})
        if t + 1 in tests:
            nxt = tests[t + 1]
            s = nxt.source == SRC_CSV
            cold_prev = predict(nxt.X, nxt.client)["global"][s].argmax(1).astype(np.int8)
        if name in GATE_PASSES:
            carry_test = gate_on_test(cfg, clients, t, tau, predict, memo, tests, carry_test, out)
        memo.clear()
        v = [macro_f1(clients.val[c]["y"], clients.val[c]["calibrated"]) for c in clients.ids]
        out["val"].append(float(np.mean(v)))
        out["fit"].append({"train_through": t, "local_seconds": round(t1 - tic, 2), "cloud_seconds": round(t2 - t1, 2), "clients_updated": len(clients.updated_now)})
        log.info("[%s] through t%d: local %.1fs, cloud %.1fs, mean val %.4f, tau_c %.2f tau_d %g; %s", name, t, t1 - tic, t2 - t1, out["val"][-1], tau["tau_c"], tau["tau_d"], memory_note())
    out["updates"], out["top"] = clients.updates, clients.top
    out["xgb_retries"] = sum(m.xgb_retries for m in clients.models.values())
    return out


def gate_on_validation(cfg, clients, t, tau, cal, temps, carry, out, log) -> dict:
    """Model through t-1 on held-out rows of t and t-1: what the choice of strength sees. No test row is read."""
    sc, frac = cfg["schedule"], cfg["models"]["val_frac"]

    def sample(w, c):
        X, y = clients.window_rows[w][c]
        vm = np.flatnonzero(val_split(y, frac, clients.seed, ("m9_gate", c, w)))
        vm = vm if len(vm) <= sc["gate_val_rows"] else rng(clients.seed, "m9_gate_rows", c, w).choice(vm, sc["gate_val_rows"], replace=False)
        return X[vm], y[vm]

    post = {c: sample(t, c) for c in clients.ids}
    pre = {c: sample(t - 1, c) for c in clients.ids}
    st = client_streams(cfg, clients.models, clients.ids, {c: v[0] for c, v in pre.items()}, {c: v[0] for c, v in post.items()}, carry, ("val", t))
    conf, dis, err, score, hold = [], [], [], [], []
    for c, (order, s, h) in st["streams"].items():
        X, y = post[c][0][order], post[c][1][order]
        comps = clients.models[c].predict_components(X)
        p = apply_temperatures(_apply_g(cal, "e3", comps["xgb"], comps["lgb"], np.full(len(X), clients.shift[c])), np.full(len(X), temps[c]))
        conf.append(p.max(1)), dis.append(gate_signals(comps)["disagreement"]), err.append(p.argmax(1) != y), score.append(s), hold.append(h)
    conf, dis, err, score, hold = map(np.concatenate, (conf, dis, err, score, hold))
    for strength in sc["score_strength"]:
        tc, td = adaptive_taus(tau["tau_c"], tau["tau_d"], score, strength)
        mask = escalate(conf, dis, tc, td, hold)
        same = same_cost_mask(conf, (dis > tau["tau_d"]) | hold, float(mask.mean()))
        out["gate_val"].append({"train_through": t - 1, "eval_window": t, "strength": strength, **gate_outcome(mask, err), "fixed_same_cost_capture": gate_outcome(same, err)["error_capture"]})
    return st["carry"]


def gate_on_test(cfg, clients, t, tau, predict, memo, tests, carry, out) -> dict:
    """The gate with one model, on its own window's rows and on the next window's. Thresholds come from the cloud (M7)."""
    sc, dc, new_carry = cfg["schedule"], cfg["drift"], dict(carry)
    for w in [t, t + 1]:
        if w not in tests:
            continue
        ts = tests[w]
        s = ts.source == SRC_CSV
        mem, p = memo[id(ts.X)], predict(ts.X, ts.client)["global"]
        by_client = {c: np.flatnonzero(s & (ts.client == c)) for c in clients.ids}
        by_client = {c: i for c, i in by_client.items() if len(i)}
        if w == t:      # the score from this boundary, held flat: a new model on its own window has no boundary to detect
            order = {c: (rng(cfg["seed"], "m9_stream", "cur", c, w).permutation(len(i)), np.full(len(i), carry.get(c, 0.0)), np.zeros(len(i), bool)) for c, i in by_client.items()}
        else:
            prev = tests[t]
            prev_s = prev.source == SRC_CSV
            st = client_streams(cfg, clients.models, list(by_client), {c: prev.X[prev_s & (prev.client == c)] for c in by_client}, {c: ts.X[i] for c, i in by_client.items()}, carry, ("test", w))
            order, new_carry = st["streams"], {**new_carry, **st["carry"]}
        idx = np.concatenate([by_client[c][order[c][0]] for c in by_client])
        score = np.concatenate([order[c][1] for c in by_client])
        hold = np.concatenate([order[c][2] for c in by_client])
        # Against all of the client's rows, as the baseline holds profiling rows too: csv rows alone would look shifted.
        psi = np.concatenate([np.full(len(by_client[c]), psi_share(np.concatenate(clients.baseline[c]), clients._sample(ts.X[ts.client == c], ("gate", c, w)), dc["psi_alert"], dc["shift_bins"]) >= sc["psi_flag_share"]) for c in by_client])
        conf, pred = p[idx].max(1), p[idx].argmax(1)
        dis, err = 0.5 * np.abs(mem["xgb"][idx] - mem["lgb"][idx]).sum(1), pred != ts.y[idx]
        row = {"train_through": t, "eval_window": w, "tau_c": tau["tau_c"], "tau_d": tau["tau_d"], "error_rate": float(err.mean()), "mean_score": float(score.mean())}
        out["gate"].append({**row, "mode": "thresholds_only", "strength": None, **gate_outcome(escalate(conf, dis, tau["tau_c"], tau["tau_d"]), err)})
        out["gate"].append({**row, "mode": "psi_flag", "strength": None, **gate_outcome(escalate(conf, dis, tau["tau_c"], tau["tau_d"], psi), err)})
        for strength in sc["score_strength"]:
            tc, td = adaptive_taus(tau["tau_c"], tau["tau_d"], score, strength)
            mask = escalate(conf, dis, tc, td, hold)
            same = same_cost_mask(conf, (dis > tau["tau_d"]) | hold, float(mask.mean()))
            out["gate"].append({**row, "mode": "fixed" if strength == 0 else "adaptive", "strength": strength, **gate_outcome(mask, err), "fixed_same_cost_capture": gate_outcome(same, err)["error_capture"]})
    return new_carry


LABELS = {"fixed": "Fixed schedule, all features (reproduces E3)", "e4": "E4: ADWIN trigger + top-K", "e4_added": "ADWIN trigger added to the fixed schedule, top-K", "e4_schedule": "ADWIN trigger, all features", "e4_label_free": "Trigger on disagreement, all features"}


def label(m: str) -> str:
    if m.endswith("_guard"):
        return label(m.removesuffix("_guard")) + ", client guard"
    if m.endswith("_local"):
        return label(m.removesuffix("_local")) + ", no g_t"
    if m.startswith("fixed_k"):
        return f"Fixed schedule, top-{m.removeprefix('fixed_k')}"
    return {**LABELS, "e3": "E3 (M7)", "e3_guard": "E3 + client guard (M7)", LOCAL: "Local + rehearsal (M6)", "fedavg": "FedAvg (M8)"}.get(m, m)


def write_report(rep, x) -> None:
    W, S, R = x["windows"], x["summaries"], x["refs"]
    t1 = "| method | window avg | avg excl. t0 | final (t5) | SD over t1 to t5 | class forgetting |\n|---|---:|---:|---:|---:|---:|\n"
    for s in [*S, *R]:
        vals = [s["macro_f1_by_window"][str(i)] if str(i) in s["macro_f1_by_window"] else s["macro_f1_by_window"][i] for i in range(1, len(W))]
        f = "n/a" if s["mean_forgetting_by_class"] is None else f"{s['mean_forgetting_by_class']:.4f}"
        t1 += f"| {label(s['method'])} | {s['window_avg_macro_f1']:.4f} | {s['window_avg_macro_f1_excl_t0']:.4f} | {s['final_macro_f1']:.4f} | {np.std(vals, ddof=1):.4f} | {f} |\n"
    t2 = "| method | window | own score before the boundary | old model on new rows | after window | mean over rounds | rounds to reach E3's earlier score (share of window) | rounds to reach its own (share of window) | first trigger round |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    show = lambda v: "not in this window" if v is None else f"{v:.1f}"
    for r in x["recovery"]:
        if r["method"] in ("fixed", "e4", "e4_schedule", "e4_added", "e3", "e4_label_free"):
            t2 += f"| {label(r['method'])} | {W[r['window']]} | {r['pre']:.3f} | {r['cold']:.3f} | {r['after']:.3f} | {r['mean']:.3f} | {show(r['recovered_common'])} | {show(r['recovered'])} | {r['first'] if r['first'] is not None else 'none'} |\n"
    t3 = "| method | local updates | clients updated, by window | uplink float32 (MB) | uplink 8-bit + zlib (MB) | downlink (kB) | local minutes |\n|---|---:|---|---:|---:|---:|---:|\n"
    for m, c in x["cost"].items():
        t3 += f"| {label(m)} | {c['updates']} | {c['by_window']} | {c['up_raw'] / 1e6:.1f} | {c['up_zlib'] / 1e6:.1f} | {c['down'] / 1e3:.1f} | {c['minutes']:.1f} |\n"
    t4 = "| K | mean validation macro-F1 | window avg (test) | final t5 (test) | picked |\n|---:|---:|---:|---:|---|\n"
    for k, v in x["k_table"]:
        t4 += f"| {k} | {v['val']:.4f} | {v['avg']:.4f} | {v['t5']:.4f} | {'yes' if k == x['k'] else ''} |\n"
    t5 = "| method | model through | eval window | error rate | mode | strength | escalated | errors caught | fixed gate at the same cost | accuracy on the rest |\n|---|---|---|---:|---|---:|---:|---:|---:|---:|\n"
    for r in x["gate"][x["gate"].method.isin(["fixed", "e4"])].itertuples():
        same = "" if pd.isna(r.fixed_same_cost_capture) else f"{r.fixed_same_cost_capture:.3f}"
        kept = "n/a" if pd.isna(r.accuracy_not_escalated) else f"{r.accuracy_not_escalated:.3f}"
        caught = "n/a" if pd.isna(r.error_capture) else f"{r.error_capture:.3f}"
        st = "" if pd.isna(r.strength) else f"{r.strength:g}"
        t5 += f"| {label(r.method)} | {W[r.train_through]} | {W[r.eval_window]} | {r.error_rate:.3f} | {r.mode} | {st} | {r.escalation_rate:.3f} | {caught} | {same} | {kept} |\n"
    tf = "| method | strength | mean errors caught, next window | fixed gate at the same cost | mean escalated, next window | picked on validation rows |\n|---|---:|---:|---:|---:|---|\n"
    for r in x["strengths"]:
        if r["method"] in ("fixed", "e4"):
            tf += f"| {label(r['method'])} | {r['strength']:g} | {r['error_capture']:.3f} | {r['fixed_same_cost_capture']:.3f} | {r['escalation_rate']:.3f} | {'yes' if r['picked'] else ''} |\n"
    md = f"""# M9: E4, drift-triggered updates, top-K features, adaptive gate (`{x['pool']}` pool)

Run `{x['run']}`. Generated by `scripts/m9_e4.py`. Seed {x['seed']}. Scores use CSV-only rows (D7). Rows reach a client in {x['chunks']} rounds per window, in random order, because the dataset has no timestamps. Between its update and the next g_t, a client uses its own calibrated output. Top-K limits new trees; older trees keep their splits, so inference still needs every feature.
{x['missing']}
## Results

{t1}
![curves](macro_f1_by_window.png)

{x['regression']}

## Recovery inside each window

For E3, the old model and its g_t serve until the window ends. `mean over rounds` is macro-F1 averaged over the rounds of the window, with the window-end model as the last of them. `first trigger round` is the earliest over clients. Read the time to E3's earlier score: a method that already fell before the boundary reaches its own lower score sooner, which isn't a better recovery.

{t2}
## Cost

{t3}
## K sweep, fixed schedule

{t4}
## Adaptive gate: choosing the strength

{tf}
## Gate on test rows

`thresholds_only` is M7's gate. `fixed` adds the proposal's ADWIN flag and equals the adaptive gate at a score of 0. `psi_flag` is the D8 sensitivity run, not a headline. Target 10% (D10). Shown for E3's schedule and E4, at the picked strength; if that is 0, adaptive rows don't appear. All strengths and methods are in `gate_test.csv`. `fixed gate at the same cost` is the errors a fixed gate catches when loosened to the same escalation rate, on the same rows, so it shows whether the drift score adds anything over a looser threshold.

{t5}
"""
    (rep / "E4.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    pool = args.pool or ("dev" if cfg["resources"]["use_dev_sample"] else "full")
    tl, windows, feats = cfg["timeline"]["headline"], cfg["timeline"]["windows"], model_features(cfg)
    target, ec, sc = cfg["labels"]["headline_target"], cfg["eval"], cfg["schedule"]
    names = pass_names(cfg)
    if args.only and args.only not in names:
        raise SystemExit(f"--only takes one of {names}")

    with Run(cfg, f"m9_e4_{pool}") as run:
        log = run.logger
        clusters = client_clusters(cfg, pool, tl, ec["n_client_clusters"])
        ev = Evaluator(load_test_sets(cfg, pool, tl, target, feats), CATEGORIES, clusters)
        window_rows, X_pub, y_pub, intro = {}, None, None, None
        if not args.report:
            for w in range(len(windows)):
                d = load_rows(cfg, pool, tl, windows=[w], split="train", features=feats)
                X, y, cl = d[feats].to_numpy(np.float32), d[target].to_numpy().astype(np.int64), d["client"].to_numpy()
                del d
                window_rows[w] = {int(c): (X[cl == c], y[cl == c]) for c in np.unique(cl)}
            dp = load_dpub(cfg, pool, tl, features=feats)
            X_pub, y_pub, intro = dp[feats].to_numpy(np.float32), dp[target].to_numpy().astype(np.int64), dp["intro_window"].to_numpy()
            del dp
            log.info("rows loaded: %s", memory_note())

        rep = REPO_ROOT / "reports" / "m9" / pool / (args.tag or "")
        ckpt = rep / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        key = {"seed": cfg["seed"], "n_jobs": cfg["resources"]["n_jobs"], "models": cfg["models"], "incremental": cfg["incremental"], "distill": cfg["distill"],
               "gate": cfg["gate"], "drift": cfg["drift"], "schedule": sc, "features": feats}
        results: dict[str, dict] = {}
        for name in names:
            if args.only and args.only != name:
                f = ckpt / f"{name}.pkl"
                if name.startswith("fixed_k") and f.exists():   # e4 needs the sweep's scores to pick K
                    saved = pickle.loads(f.read_bytes())
                    if saved["key"] == key:
                        results[name] = {"val": saved["out"]["val"]}
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
            k = None
            if name.startswith("fixed_k"):
                k = int(name.removeprefix("fixed_k"))
            elif name in ("e4", "e4_added"):
                val = {int(n.removeprefix("fixed_k")): float(np.mean(r["val"])) for n, r in results.items() if n.startswith("fixed_k")}
                if len(val) != len(sc["k_sweep"]):
                    raise SystemExit(f"{name} needs the whole K sweep first; found K = {sorted(val)}")
                k = pick_k(val)
                log.info("[%s] K = %d from %s", name, k, val)
            have = {r["model"] for r in ev.records}
            results[name] = run_pass(cfg, name, window_rows, X_pub, y_pub, intro, ev, log, k=k)
            f.write_bytes(pickle.dumps({"key": key, "out": results[name], "records": [r for r in ev.records if r["model"] not in have],
                                        "confusions": {k_: v for k_, v in ev.confusions.items() if k_[0] not in have}}))
            gc.collect()
        if args.only:
            log.info("[%s] checkpoint saved, no report in this process", args.only)
            return

        done = [n for n in names if n in results]
        order = [m for n in done for m in (n, f"{n}_guard", f"{n}_local")]
        summaries = [ev.summary(m, HEADLINE, eps=ec["recover_eps"]) for m in order]
        by = {s["method"]: s for s in summaries}
        m7_dir = REPO_ROOT / "reports" / "m7" / pool / (args.tag or "")
        m7 = json.loads((m7_dir / "summary.json").read_text(encoding="utf-8"))
        refs = [s for s in m7["summaries"] if s["method"] in ("e3", "e3_guard", LOCAL)]
        regression = "No `fixed` pass in this report, so nothing was checked against M7."
        if "fixed" in by:
            e3 = next(s for s in refs if s["method"] == "e3")
            regression = f"Check against M7: `fixed` scores {by['fixed']['window_avg_macro_f1']:.5f} on window average and E3 in M7's report {e3['window_avg_macro_f1']:.5f} (difference {by['fixed']['window_avg_macro_f1'] - e3['window_avg_macro_f1']:+.5f})."

        recovery = []
        e3 = next(s for s in refs if s["method"] == "e3")
        for d, r in e3["time_to_recover"].items():
            d = int(d)
            after = e3["macro_f1_by_window"][str(d)]
            curve = [r["cold"]] * sc["chunks"] + [after]
            rec = recovery_share(curve, r["pre"], ec["recover_eps"])
            recovery.append({"method": "e3", "window": d, "pre": r["pre"], "cold": r["cold"], "after": after, "mean": float(np.mean(curve)), "recovered": rec, "recovered_common": rec, "first": None})
        for n in done:
            if not results[n]["curves"]:
                continue
            ser = by[n]["macro_f1_by_window"]
            for c in results[n]["curves"]:
                d, curve = c["window"], c["curve"]
                # Same numbers as the harness, or the recovery table isn't on the same scale as the results table.
                assert abs(curve[-1] - ser[d]) < 1e-6, (n, d, curve[-1], ser[d])
                recovery.append({"method": n, "window": d, "pre": ser[d - 1], "cold": curve[0], "after": curve[-1], "mean": float(np.mean(curve)), "recovered": recovery_share(curve, ser[d - 1], ec["recover_eps"]),
                                 "recovered_common": recovery_share(curve, e3["time_to_recover"][str(d)]["pre"], ec["recover_eps"]), "first": c["triggered"][0] if c["triggered"] else None})

        strengths, gates, picked = [], [], {}
        for n in done:
            if not results[n]["gate_val"]:
                continue
            gv = pd.DataFrame(results[n]["gate_val"])
            picked[n] = pick_strength(gv, cfg["gate"]["max_escalation_rate"])
            for st, g in gv.groupby("strength"):
                strengths.append({"method": n, "strength": float(st), "error_capture": float(g.error_capture.mean()), "fixed_same_cost_capture": float(g.fixed_same_cost_capture.mean()), "escalation_rate": float(g.escalation_rate.mean()), "picked": float(st) == picked[n]})
            gates.append(pd.DataFrame(results[n]["gate"]).assign(method=n, picked_strength=picked[n]))
        gate = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame()

        cost, k_table = {}, []
        for n in done:
            u, cm, fit = pd.DataFrame(results[n]["updates"]), pd.DataFrame(results[n]["comm"]), pd.DataFrame(results[n]["fit"])
            cost[n] = {"updates": len(u), "by_window": " ".join(str(int(v)) for v in cm.clients_updated), "up_raw": float(cm.up_raw.sum()), "up_zlib": float(cm.up_q8_zlib.sum()), "down": float(cm.down.sum()), "minutes": float(fit.local_seconds.sum() / 60), "xgb_retries": results[n]["xgb_retries"]}
            if n.startswith("fixed_k"):
                k_table.append((int(n.removeprefix("fixed_k")), {"val": float(np.mean(results[n]["val"])), "avg": by[n]["window_avg_macro_f1"], "t5": by[n]["final_macro_f1"]}))

        ev.long_table().round(6).to_csv(rep / "results.csv", index=False)
        ev.per_class_table().round(6).to_csv(rep / "per_class.csv", index=False)
        ev.confusion_table().to_csv(rep / "confusions.csv", index=False)
        if len(gate):
            gate.round(6).to_csv(rep / "gate_test.csv", index=False)
        pd.DataFrame(recovery).round(6).to_csv(rep / "recovery.csv", index=False)
        (rep / "summary.json").write_text(dumps({"summaries": summaries, "references": refs, "cost": cost, "k": results["e4"]["k"] if "e4" in results else None,
                                                 "strength": picked, "strengths": strengths, "recovery": recovery, "val": {n: results[n]["val"] for n in done},
                                                 "top_features": {n: results[n]["top"] for n in done if results[n]["top"]}, "curves": {n: results[n]["curves"] for n in done}}, indent=2), encoding="utf-8")
        for s in refs:
            s["macro_f1_by_window"] = {int(a): v for a, v in s["macro_f1_by_window"].items()}
        shown = {label(s["method"]): pd.Series(s["macro_f1_by_window"]) for s in [*summaries, *refs] if s["method"] in ("e3", "e4", "e4_schedule", LOCAL)}
        fig_curves(shown, windows, rep / "macro_f1_by_window.png", "Macro-F1 per window, CSV rows only")
        shown_gate = gate[(gate["mode"] != "adaptive") & (gate.strength.isna() | (gate.strength == 0)) | (gate.strength == gate.picked_strength)].drop_duplicates(["method", "train_through", "eval_window", "mode", "strength"]) if len(gate) else gate
        write_report(rep, {"pool": pool, "run": run.dir.name, "seed": cfg["seed"], "windows": windows, "chunks": sc["chunks"], "summaries": [s for s in summaries if not s["method"].endswith("_local")], "refs": refs,
                           "recovery": recovery, "cost": cost, "k_table": k_table, "k": results["e4"]["k"] if "e4" in results else None, "gate": shown_gate, "strengths": strengths, "regression": regression,
                           "missing": "" if set(done) == set(names) else f"\n**Passes missing from this report: {sorted(set(names) - set(done))}. They did not finish.**\n"})
        if set(done) == set(names):
            shutil.rmtree(ckpt)
        log.info("report -> %s", rep)


if __name__ == "__main__":
    main()
