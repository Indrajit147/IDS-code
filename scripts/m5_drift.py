"""M5: drift signals and the uncertainty gate, on the saved M3 models.

Boundary experiment, for each window d >= 1 and each client (plus one pooled stream): the model
trained through d-1 scores that client's test rows of window d-1, then those of window d. Rows are
shuffled inside each window, because the stored order is grouped by class and would look like drift.
A detection before the boundary is a false alarm; the first rise after it is the detection.

Usage: python scripts/m5_drift.py [--pool dev|full] [--run <m3 run dir>] [--env ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ids.data.io import model_features
from ids.data.schema import CATEGORIES
from ids.drift.adwin import detect
from ids.drift.gate import drift_state, escalate, frontier, gate_outcome, gate_signals, tune
from ids.drift.shift import feature_shift
from ids.eval.harness import load_test_sets
from ids.eval.plots import fig_escalation, fig_psi_by_window, fig_signal_traces
from ids.models.edge import EdgeEnsemble, val_split
from ids.partition.build import SRC_CSV
from ids.partition.io import load_rows
from ids.utils import JsonlWriter, Run, load_config, rng, set_global_seed
from ids.utils.config import REPO_ROOT
from ids.utils.jsonl import dumps
from ids.utils.run import latest_run

SIGNALS = ["error", "entropy", "disagreement"]
LABEL_FREE = ["entropy", "disagreement"]
POOLED = -1
BUDGETS = [0.05, 0.10, 0.20, 0.30]
ALERT_SHARE = 0.10      # a client counts as shifted if this share of its features is over the PSI alert
AFFECTED_RISE = 0.02    # a stream counts as affected if its error rate rises at least this much


def score(model, ts) -> dict[str, np.ndarray]:
    comps = model.predict_components(ts.X)
    out = gate_signals(comps)
    out["error"] = (comps["calibrated"].argmax(1) != ts.y).astype(float)
    return out


def stream_order(seed, ts, client, window) -> np.ndarray:
    idx = np.arange(len(ts.y)) if client == POOLED else np.flatnonzero(ts.client == client)
    return idx[rng(seed, "stream", client, window).permutation(len(idx))]


def boundary_streams(cfg, test_sets, models, registry, log):
    """ADWIN on every stream. Returns per-stream results and what the gate and the traces need."""
    delta, seed = cfg["drift"]["adwin_delta"], cfg["seed"]
    windows = cfg["timeline"]["windows"]
    clients = sorted(int(c) for c in np.unique(test_sets[0].client))
    rows, scored, traces = [], {}, {}
    for d in range(1, len(windows)):
        pre_ts, post_ts = test_sets[d - 1], test_sets[d]
        pre, post = score(models[d - 1], pre_ts), score(models[d - 1], post_ts)
        scored[d] = post
        for client in [*clients, POOLED]:
            i0, i1 = stream_order(seed, pre_ts, client, d - 1), stream_order(seed, post_ts, client, d)
            n0, n1 = len(i0), len(i1)
            rec = {"boundary": d, "client": client, "n_pre": n0, "n_post": n1,
                   "error_pre": float(pre["error"][i0].mean()) if n0 else None,
                   "error_post": float(post["error"][i1].mean()) if n1 else None}
            fires_by_signal = {}
            for sig in SIGNALS:
                stream = np.concatenate([pre[sig][i0], post[sig][i1]])
                fires = detect(stream, delta)
                fires_by_signal[sig] = fires
                rises = [p for p, change in fires if p >= n0 and change > 0]
                rec[f"{sig}_false_alarms"] = sum(p < n0 for p, _ in fires)
                rec[f"{sig}_detected"] = bool(rises)
                rec[f"{sig}_delay"] = rises[0] - n0 if rises else None
                for p, change in fires:
                    registry.write({"kind": "adwin", "boundary": windows[d], "client": client, "signal": sig,
                                    "stream_position": p, "rows_after_boundary": p - n0, "change_in_mean": change})
            rows.append(rec)
            if client == POOLED:
                traces[d] = {"n_pre": n0, "signals": {s: np.concatenate([pre[s][i0], post[s][i1]]) for s in SIGNALS},
                             "fires": {s: [p for p, _ in fires_by_signal[s]] for s in SIGNALS}}
            else:
                scored.setdefault((d, "post_order"), {})[client] = (i1, n0, fires_by_signal)
        log.info("boundary %s done", windows[d])
    return pd.DataFrame(rows), scored, traces


def adwin_table(streams: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    rows = []
    clients = streams[streams["client"] != POOLED]
    pooled = streams[streams["client"] == POOLED].set_index("boundary")
    for d, g in clients.groupby("boundary"):
        affected = g[(g["error_post"] - g["error_pre"]) >= AFFECTED_RISE]
        for sig in SIGNALS:
            delays = g[f"{sig}_delay"].dropna()
            p = pooled.loc[d]
            rows.append({"boundary": windows[d], "signal": sig, "clients_detected": int(g[f"{sig}_detected"].sum()),
                         "clients": len(g), "affected_detected": int(affected[f"{sig}_detected"].sum()),
                         "affected": len(affected), "median_delay_rows": float(delays.median()) if len(delays) else None,
                         "client_false_alarms": int(g[f"{sig}_false_alarms"].sum()),
                         "pre_rows": int(g["n_pre"].sum()), "pooled_detected": bool(p[f"{sig}_detected"]),
                         "pooled_delay_rows": p[f"{sig}_delay"], "pooled_false_alarms": int(p[f"{sig}_false_alarms"])})
    return pd.DataFrame(rows)


def tune_gate(cfg, models, feats, pool, tl, target="y_category"):
    """Thresholds from the validation rows M3 held out: same selection, order and seed as its training.
    They set early stopping and calibration but no tree splits, so they can flatter the model slightly."""
    g = cfg["gate"]
    taus_c = np.round(np.arange(0.40, 1.00, 0.02), 2)
    taus_d = [float(x) for x in g["tune_tau_d"]]
    fronts = []
    for t, model in models.items():
        if t == 0:      # Benign vs DoS only: no errors to tune on
            continue
        d = load_rows(cfg, pool, tl, windows=list(range(t + 1)), split="train", features=feats)
        y = d[target].to_numpy()
        vm = val_split(y, cfg["models"]["val_frac"], cfg["seed"], ("wifi", t))
        comps = model.predict_components(d[feats].to_numpy(np.float32)[vm])
        del d
        s = gate_signals(comps)
        err = (comps["calibrated"].argmax(1) != y[vm]).astype(float)
        fronts.append(frontier(s["confidence"], s["disagreement"], err, taus_c, taus_d).assign(model=t))
    front = pd.concat(fronts, ignore_index=True)
    mean = front.groupby(["tau_c", "tau_d"], as_index=False)[
        ["escalation_rate", "error_capture", "error_rate_in_escalated", "accuracy_not_escalated"]].mean()
    budgets = pd.DataFrame([{"budget": b, **tune(mean, b)} for b in BUDGETS])
    return mean, tune(mean, g["max_escalation_rate"]), budgets


def gate_on_test(cfg, test_sets, models, scored, chosen, windows) -> pd.DataFrame:
    """Headline rows are CSV-only (D7); escalation_rate_all counts every flow, which is what the LLM would see."""
    g, rows = cfg["gate"], []
    tc, td = chosen["tau_c"], chosen["tau_d"]
    for d in range(1, len(windows)):
        ts = test_sets[d]
        csv = ts.source == SRC_CSV
        unseen = ~np.isin(ts.y, models[d - 1].classes_)
        for state, s in (("settled", score(models[d], ts)), ("before update", scored[d])):
            base = escalate(s["confidence"], s["disagreement"], tc, td)
            drift = np.zeros(len(ts.y), bool)
            if state == "before update":     # settled models have had their update, so nothing fires here
                for client, (order, n0, fires) in scored[(d, "post_order")].items():
                    pos = [p - n0 for p, change in fires[g["drift_signal"]] if p >= n0 and change > 0]
                    drift[order] = drift_state(len(order), pos, g["drift_hold_rows"])
            for flag, mask in (("no", base), ("yes", base | drift)):
                if flag == "yes" and state == "settled":
                    continue
                rows.append({"window": windows[d], "model": state, "drift_flag": flag,
                             "escalation_rate_all": float(mask.mean()), **gate_outcome(mask[csv], s["error"][csv]),
                             "unseen_class_capture": float(mask[unseen].mean()) if state != "settled" and unseen.any() else None})
    return pd.DataFrame(rows)


def distribution_shift(cfg, test_sets, feats, pool, tl, registry, log):
    """Each client's test traffic at window w against its own running baseline (train rows of windows < w).
    Control: the same baseline against test traffic of window w-1, which the baseline already covers."""
    dc, seed, windows = cfg["drift"], cfg["seed"], cfg["timeline"]["windows"]
    cap = dc["shift_max_rows"]
    clients = sorted(int(c) for c in np.unique(test_sets[0].client))
    baseline: dict[int, list[np.ndarray]] = {c: [] for c in clients}
    rows = []

    def sample(X, key):
        return X if len(X) <= cap else X[rng(seed, "shift", *key).choice(len(X), cap, replace=False)]

    for w in range(len(windows)):
        if w > 0:
            for c in clients:
                ref = np.concatenate(baseline[c]) if baseline[c] else None
                for kind, ts in (("current", test_sets[w]), ("control", test_sets[w - 1])):
                    cur = ts.X[ts.client == c]
                    if ref is None or len(ref) < 50 or len(cur) < 50:
                        continue
                    df = feature_shift(ref, sample(cur, (kind, c, w)), feats, dc["shift_bins"])
                    rows.append(df.assign(window=w, client=c, kind=kind))
                    if kind == "current":
                        top = df.nlargest(3, "psi")
                        registry.write({"kind": "distribution", "window": windows[w], "client": c,
                                        "mean_psi": df["psi"].mean(), "mean_kl": df["kl"].mean(),
                                        "mean_wasserstein": df["wasserstein"].mean(),
                                        "n_features_over_alert": int((df["psi"] > dc["psi_alert"]).sum()),
                                        "top_features": dict(zip(top["feature"], top["psi"]))})
        d = load_rows(cfg, pool, tl, windows=[w], split="train", features=feats)
        for c, part in d.groupby("client"):
            baseline[int(c)].append(sample(part[feats].to_numpy(np.float32), ("base", int(c), w)))
        log.info("distribution shift through %s done", windows[w])
    return pd.concat(rows, ignore_index=True)


def write_report(rep, x) -> None:
    w = x["windows"]
    a = x["adwin"]
    def num(v, f="{:,.0f}"):
        return "n/a" if v is None or pd.isna(v) else f.format(v)
    t1 = "| boundary | signal | clients detected | affected clients detected | median delay (rows) | client false alarms | pre-boundary rows | pooled delay (rows) | pooled false alarms |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in a.itertuples():
        t1 += (f"| {r.boundary} | {r.signal} | {r.clients_detected}/{r.clients} | {r.affected_detected}/{r.affected} | {num(r.median_delay_rows)} | "
               f"{r.client_false_alarms} | {r.pre_rows:,} | {num(r.pooled_delay_rows) if r.pooled_detected else 'missed'} | {r.pooled_false_alarms} |\n")
    tot = a.groupby("signal").agg(fa=("client_false_alarms", "sum"), rows=("pre_rows", "sum"), det=("clients_detected", "sum"),
                                  n=("clients", "sum"), adet=("affected_detected", "sum"), an=("affected", "sum"),
                                  pooled=("pooled_detected", "sum"), pfa=("pooled_false_alarms", "sum")).reindex(SIGNALS)
    t0 = "| signal | clients detected | affected clients detected | boundaries caught, pooled | client false alarms | per 100k rows | pooled false alarms |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for s, r in tot.iterrows():
        t0 += f"| {s} | {r.det}/{r.n} | {r.adet}/{r.an} | {r.pooled}/{len(w) - 1} | {r.fa} | {1e5 * r.fa / r.rows:.1f} | {r.pfa} |\n"
    sh = x["shift"]
    t2 = f"| window | mean PSI | control | features over PSI alert | control | clients with at least {ALERT_SHARE:.0%} of features over it | control | mean KL | mean Wasserstein (std units) | top features by PSI |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    def shifted(g):
        return f"{(g.assign(over=g.psi > x['psi_alert']).groupby('client')['over'].mean() >= ALERT_SHARE).sum()}/{g.client.nunique()}"
    for d in range(1, len(w)):
        c, k = sh[(sh.window == d) & (sh.kind == "current")], sh[(sh.window == d) & (sh.kind == "control")]
        if c.empty:
            continue
        top = c.groupby("feature")["psi"].mean().nlargest(3)
        t2 += (f"| {w[d]} | {c.psi.mean():.3f} | {k.psi.mean():.3f} | {(c.psi > x['psi_alert']).mean():.1%} | {(k.psi > x['psi_alert']).mean():.1%} | {shifted(c)} | {shifted(k)} | "
               f"{c.kl.mean():.3f} | {c.wasserstein.mean():.3f} | {', '.join(f'{f} ({v:.2f})' for f, v in top.items())} |\n")
    g = x["gate_test"]
    t3 = "| window | model | drift flag | LLM calls per 1k flows | errors sent to cloud | error rate among escalated | accuracy on the rest | unseen-class rows escalated |\n|---|---|---|---:|---:|---:|---:|---:|\n"
    for r in g.itertuples():
        t3 += (f"| {r.window} | {r.model} | {r.drift_flag} | {1000 * r.escalation_rate_all:.0f} | {num(r.error_capture, '{:.3f}')} | "
               f"{num(r.error_rate_in_escalated, '{:.3f}')} | {num(r.accuracy_not_escalated, '{:.3f}')} | {num(r.unseen_class_capture, '{:.3f}')} |\n")
    c = x["chosen"]
    t4 = "| escalation budget | τc | τd | escalation rate | errors captured |\n|---:|---:|---:|---:|---:|\n"
    for r in x["budgets"].itertuples():
        t4 += f"| {r.budget:.0%} | {r.tau_c:.2f} | {r.tau_d:g} | {r.escalation_rate:.3f} | {r.error_capture:.3f} |\n"
    md = f"""# M5: drift signals and the uncertainty gate (`{x['pool']}` pool)

Run `{x['run']}`, models from `{x['m3_run']}`. Generated by `scripts/m5_drift.py`.

**Experiment.** At each boundary, the model trained through the earlier window scores a client's test rows from that window, then its rows from the new window. Rows are shuffled inside each window. A detection before the boundary is a false alarm. A *detection* is the first rise in the signal after the boundary. The data has no timestamps, so real bursts are absent and the false-alarm counts apply only to shuffled traffic. An *affected* client is one whose error rate rises by at least {AFFECTED_RISE}. Only `entropy` and `disagreement` are available at deployment, because `error` needs labels. ADWIN δ = {x['delta']}.

## ADWIN summary

{t0}
## ADWIN per boundary

{t1}
![traces](signal_traces.png)

## Distribution shift per client, against its running baseline

Baseline: the client's training rows from earlier windows. Control: the same baseline against test rows of the previous window, which it covers, though diluted by the older windows. PSI alert = {x['psi_alert']}. As in the proposal, these are reported and don't trigger anything.

{t2}
![psi](psi_by_window.png)

## Gate

Thresholds were tuned on the validation rows M3 held out, averaged over the models for t1 to t5, for the most errors captured within an escalation budget. The proposal sets no budget, so {x['budget']:.0%} is an assumption. **Used below: τc = {c['tau_c']:.2f}, τd = {c['tau_d']:g}.** Those rows set early stopping and calibration, so compare with the test rows below.

{t4}
![frontier](escalation_vs_tau.png)

On the test sets. LLM calls count every flow; the other columns use CSV-only rows (D7). *Before update* is the old model on the new window; *settled* is the model after its update. The drift flag uses `{x['drift_signal']}` and holds for {x['hold']} flows after each rise.

{t3}
Every ADWIN and distribution event is in the run's `drift_registry.jsonl`; `adwin_streams.csv` and `distribution_shift.csv` hold the per-stream and per-feature numbers.
"""
    (rep / "DRIFT.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default=None, help="dev | full (default from resources.use_dev_sample)")
    ap.add_argument("--run", default=None, help="M3 run directory (default: latest for the pool)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    pool = args.pool or ("dev" if cfg["resources"]["use_dev_sample"] else "full")
    tl, windows, feats = cfg["timeline"]["headline"], cfg["timeline"]["windows"], model_features(cfg)
    m3_run = Path(args.run) if args.run else latest_run(cfg, f"*_m3_baselines_{pool}")

    with Run(cfg, f"m5_drift_{pool}") as run:
        log = run.logger
        registry = JsonlWriter(run.path("drift_registry.jsonl"))
        test_sets = load_test_sets(cfg, pool, tl, cfg["labels"]["headline_target"], feats)
        models = {t: EdgeEnsemble.load(m3_run / "models" / "wifi" / f"t{t}", n_jobs=cfg["resources"]["n_jobs"]) for t in range(len(windows))}
        streams, scored, traces = boundary_streams(cfg, test_sets, models, registry, log)
        adwin = adwin_table(streams, windows)
        shift = distribution_shift(cfg, test_sets, feats, pool, tl, registry, log)
        front, chosen, budgets = tune_gate(cfg, models, feats, pool, tl)
        log.info("gate thresholds: %s", chosen)
        gate_test = gate_on_test(cfg, test_sets, models, scored, chosen, windows)

        rep = REPO_ROOT / "reports" / "m5" / pool
        rep.mkdir(parents=True, exist_ok=True)
        streams.to_csv(rep / "adwin_streams.csv", index=False)
        adwin.to_csv(rep / "adwin_summary.csv", index=False)
        shift.round(5).to_csv(rep / "distribution_shift.csv", index=False)
        front.round(5).to_csv(rep / "gate_frontier.csv", index=False)
        gate_test.round(5).to_csv(rep / "gate_test.csv", index=False)
        (rep / "gate_thresholds.json").write_text(dumps(chosen, indent=2), encoding="utf-8")
        fig_signal_traces(traces, windows, SIGNALS, rep / "signal_traces.png")
        fig_escalation(front, chosen, rep / "escalation_vs_tau.png")
        fig_psi_by_window(shift, windows, cfg["drift"]["psi_alert"], rep / "psi_by_window.png")
        write_report(rep, {"pool": pool, "run": run.dir.name, "m3_run": m3_run.name, "windows": windows, "adwin": adwin, "shift": shift,
                           "gate_test": gate_test, "chosen": chosen, "budgets": budgets, "psi_alert": cfg["drift"]["psi_alert"], "delta": cfg["drift"]["adwin_delta"],
                           "budget": cfg["gate"]["max_escalation_rate"], "drift_signal": cfg["gate"]["drift_signal"], "hold": cfg["gate"]["drift_hold_rows"]})
        log.info("report -> %s", rep)


if __name__ == "__main__":
    main()
