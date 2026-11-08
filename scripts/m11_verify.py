"""M11: gated LLM verification of E4 (`e4_added`, D11): E5 (gated) and E6 (always-on against gated).

Two steps, each can run in its own process:
  fit  reruns M9's e4_added pass (through M10's fit, which checks the scores against M9's report) and, from a hook in
       M9's gate, keeps the evidence for a uniform sample of test rows per client and evaluation window;
  llm  sends the sampled rows to the verifier, applies the D3 policy and scores. See README, M11, for the design.

Usage: python scripts/m11_verify.py [--pool dev|full] [--env ...] [--tag seed43] [--set llm.backend=hf ...] [--only fit|llm]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from m10_shap import fit   # noqa: E402

from ids.data.io import model_features   # noqa: E402
from ids.data.schema import CATEGORIES   # noqa: E402
from ids.drift.gate import adaptive_taus, escalate   # noqa: E402
from ids.eval.explain import ensemble_shap, top_k   # noqa: E402
from ids.eval.harness import load_test_sets   # noqa: E402
from ids.eval.system import latency_summary, payload_bytes   # noqa: E402
from ids.llm.verifier import d3_allows, grounded, make_backend, verify   # noqa: E402
from ids.partition.io import load_dpub   # noqa: E402
from ids.utils import Run, load_config, rng, set_global_seed   # noqa: E402
from ids.utils.config import REPO_ROOT   # noqa: E402
from ids.utils.jsonl import JsonlWriter, dumps   # noqa: E402
from ids.utils.run import memory_note   # noqa: E402

K = len(CATEGORIES)
BENIGN = CATEGORIES.index("benign")


def sig(v: float) -> float:
    return float(f"{float(v):.4g}")


class Capture:
    """Called by M9's gate with each window's stream; keeps the evidence of the sampled rows."""

    def __init__(self, cfg, feats, tests, X_pub, intro, strength, log):
        self.cfg, self.feats, self.tests, self.X_pub, self.intro, self.strength, self.log = cfg, feats, tests, X_pub, intro, strength, log
        self.rows, self.stream, self.centroids = [], [], {}

    def centroid(self, t, models) -> None:
        """Median of each feature per class on D_pub, by the clients' mean calibrated output (pseudo-labels, no D_pub label read)."""
        m = self.intro <= t
        X = self.X_pub[m]
        q = np.mean([mod.predict_components(X)["calibrated"] for mod in models.values()], axis=0)
        lab = q.argmax(1)
        self.centroids[t] = {CATEGORIES[c]: np.median(X[lab == c], axis=0).tolist() for c in range(K) if (lab == c).any()}

    def __call__(self, t, w, idx, p, xgb, lgb, score, hold, tau, models) -> None:
        lc, seed = self.cfg["llm"], self.cfg["seed"]
        if t not in self.centroids:
            self.centroid(t, models)
        ts = self.tests[w]
        client, pred, conf = ts.client[idx], p.argmax(1), p.max(1)
        dis = 0.5 * np.abs(xgb - lgb).sum(1)
        tc, td = adaptive_taus(tau["tau_c"], tau["tau_d"], score, self.strength)
        esc = escalate(conf, dis, tc, td, hold)
        prior, sent = np.zeros((len(idx), K)), 0
        for c in np.unique(client):
            m = np.flatnonzero(client == c)          # this client's rows, in stream order
            oh = np.eye(K)[pred[m]]
            seen = np.cumsum(oh, 0) - oh
            prior[m] = seen / np.maximum(np.arange(len(m)), 1)[:, None]
            self.stream.append({"train_through": t, "eval_window": w, "client": int(c), "rows": len(m), "escalated": int(esc[m].sum())})
            pick = rng(seed, "m11_rows", int(c), t, w).permutation(len(m))[: lc["rows_per_client"]]
            pos, always = m[pick], np.arange(len(pick)) < lc["always_on_rows"]
            need = esc[pos] | always
            fr = pd.DataFrame({"train_through": t, "eval_window": w, "client": int(c), "row": idx[pos], "y": ts.y[idx[pos]].astype(int), "pred": pred[pos],
                               **{f"p{j}": p[pos, j] for j in range(K)}, "xgb_pred": xgb[pos].argmax(1), "lgb_pred": lgb[pos].argmax(1),
                               "dis": dis[pos], "conf": conf[pos], "score": score[pos], "hold": hold[pos], "escalated": esc[pos], "always_on": always,
                               **{f"prior{j}": prior[pos, j] for j in range(K)}})
            k = self.cfg["explain"]["top_k"]
            sent += int(need.sum())
            for j in range(k):
                fr[f"f{j + 1}"], fr[f"x{j + 1}"], fr[f"s{j + 1}"] = "", np.nan, np.nan
            if need.any():
                X = ts.X[idx[pos[need]]]
                ti, tv = top_k(ensemble_shap(models[int(c)], X, cls=pred[pos[need]])["phi"], k)
                r = np.arange(len(X))
                for j in range(k):
                    fr.loc[need, f"f{j + 1}"] = [self.feats[i] for i in ti[:, j]]
                    fr.loc[need, f"x{j + 1}"] = X[r, ti[:, j]]
                    fr.loc[need, f"s{j + 1}"] = tv[:, j]
            self.rows.append(fr)
        self.log.info("t%d on window %d: %d rows, %.3f escalated, %d sampled rows for the verifier; %s", t, w, len(idx), esc.mean(), sent, memory_note())


def run_fit(cfg, pool, tag, rep, log) -> None:
    tl, feats, target = cfg["timeline"]["headline"], model_features(cfg), cfg["labels"]["headline_target"]
    m9 = json.loads((REPO_ROOT / "reports" / "m9" / pool / (tag or "") / "summary.json").read_text(encoding="utf-8"))
    strength = float(m9["strength"]["e4_added"])
    dp = load_dpub(cfg, pool, tl, features=feats)
    cap = Capture(cfg, feats, load_test_sets(cfg, pool, tl, target, feats), dp[feats].to_numpy(np.float32), dp["intro_window"].to_numpy(), strength, log)
    del dp
    log.info("gate strength %g from M9's report", strength)
    fit(cfg, pool, tag, None, rep, log, hook=cap)
    pd.concat(cap.rows, ignore_index=True).to_parquet(rep / "evidence.parquet", index=False)
    pd.DataFrame(cap.stream).to_csv(rep / "stream.csv", index=False)
    (rep / "centroids.json").write_text(dumps({"strength": strength, "features": feats, "by_window": cap.centroids}), encoding="utf-8")


def payload(r, cen: dict, k: int) -> dict:
    """What the cloud sends the LLM for one row: no label, no window name."""
    top = [{"feature": r[f"f{j}"], "value": sig(r[f"x{j}"]), "shap": sig(r[f"s{j}"])} for j in range(1, k + 1)]
    known = list(cen["by_window"][str(int(r.train_through))])
    med = cen["by_window"][str(int(r.train_through))]
    fi = {f: i for i, f in enumerate(cen["features"])}
    return {"edge_label": CATEGORIES[int(r.pred)], "probs": {c: sig(r[f"p{j}"]) for j, c in enumerate(CATEGORIES)},
            "booster_labels": {"xgboost": CATEGORIES[int(r.xgb_pred)], "lightgbm": CATEGORIES[int(r.lgb_pred)]}, "disagreement": sig(r.dis),
            "top_features": top, "class_medians": {t["feature"]: {c: sig(med[c][fi[t["feature"]]]) for c in known} for t in top},
            "recent_class_share": {c: round(float(r[f"prior{j}"]), 3) for j, c in enumerate(CATEGORIES)},
            "drift": {"flag": bool(r.hold), "score": sig(r.score)}, "known_classes": known}


def scores(y, yhat) -> dict:
    ben = y == BENIGN
    return {"rows": len(y), "macro_f1": float(f1_score(y, yhat, labels=np.unique(y), average="macro", zero_division=0)),
            "fp_rate": float((yhat[ben] != BENIGN).mean()) if ben.any() else None, "benign_rows": int(ben.sum()),
            "miss_rate": float((yhat[~ben] == BENIGN).mean()) if (~ben).any() else None}


def override_counts(y, edge, new) -> dict:
    ch = edge != new
    return {"overrides": int(ch.sum()), "fixed": int((ch & (edge != y) & (new == y)).sum()), "broke": int((ch & (edge == y) & (new != y)).sum())}


def run_llm(cfg, pool, rep, log) -> None:
    lc = cfg["llm"]
    ev = pd.read_parquet(rep / "evidence.parquet")
    cen = json.loads((rep / "centroids.json").read_text(encoding="utf-8"))
    need = ev[ev.escalated | ev.always_on]
    items = [{"key": int(i), "payload": payload(r, cen, cfg["explain"]["top_k"])} for i, r in need.iterrows()]
    log.info("%d rows to verify (%d escalated, %d always-on only), backend %s", len(items), int(ev.escalated.sum()), int((ev.always_on & ~ev.escalated).sum()), lc["backend"])
    backend = make_backend(lc)
    recs, batches = verify(items, backend, CATEGORIES, cen["features"], lc["batch_size"], lc["max_retries"], JsonlWriter(rep / "llm_log.jsonl"))
    log.info("verified: %d of %d valid; %s", sum(r["valid"] for r in recs), len(recs), memory_note())
    code = {c: i for i, c in enumerate(CATEGORIES)}
    keys = [r["key"] for r in recs]
    nd = need.loc[keys]
    flag = (nd.xgb_pred != nd.lgb_pred).to_numpy(), (nd.hold | (nd.score > 0)).to_numpy()
    out = pd.DataFrame({"llm": [code[r["reply"]["final_label"]] for r in recs], "valid": [r["valid"] for r in recs],
                        "first_try_valid": [r["first_try_valid"] for r in recs], "attempts": [r["attempts"] for r in recs],
                        "d3": [r["valid"] and d3_allows(r["reply"], a, b) for r, a, b in zip(recs, *flag)],
                        "grounded": [r["valid"] and grounded(r["reply"], it["payload"]) for r, it in zip(recs, items)],
                        "payload_bytes": [payload_bytes(it["payload"]) for it in items]}, index=keys)
    ev = ev.join(out)
    ev["llm"] = ev.llm.fillna(-1).astype(int)
    for c in ("valid", "first_try_valid", "d3", "grounded"):
        ev[c] = ev[c].fillna(False).astype(bool)
    ev["e5"] = np.where(ev.escalated & ev.d3, ev.llm, ev.pred)
    ev["e5_free"] = np.where(ev.escalated & ev.valid, ev.llm, ev.pred)
    ev["e6"] = np.where(ev.d3, ev.llm, ev.pred)
    ev["e6_free"] = np.where(ev.valid, ev.llm, ev.pred)
    ev.drop(columns=[c for c in ev if c[:1] in "fxs" and c[1:].isdigit()]).to_parquet(rep / "verified.parquet", index=False)

    ev["setting"] = np.where(ev.eval_window == ev.train_through, "own window", "boundary")
    tables, ovr = [], []
    for (st, w), g in ev.groupby(["setting", "eval_window"]):
        y = g.y.to_numpy()
        for m in ("pred", "e5", "e5_free"):
            tables.append({"rows_set": "sample", "setting": st, "eval_window": w, "method": m, **scores(y, g[m].to_numpy())})
        a = g[g.always_on]
        for m in ("pred", "e5", "e6", "e6_free"):
            tables.append({"rows_set": "always_on", "setting": st, "eval_window": w, "method": m, **scores(a.y.to_numpy(), a[m].to_numpy())})
        for m, sub in (("e5", g), ("e5_free", g), ("e6", a), ("e6_free", a)):
            ovr.append({"setting": st, "eval_window": w, "method": m, **override_counts(sub.y.to_numpy(), sub.pred.to_numpy(), sub[m].to_numpy())})
    tab, ovr = pd.DataFrame(tables), pd.DataFrame(ovr)
    stream = pd.read_csv(rep / "stream.csv")
    stream["setting"] = np.where(stream.eval_window == stream.train_through, "own window", "boundary")
    calls = (stream.groupby("setting").escalated.sum() / stream.groupby("setting").rows.sum() * 1000).round(1).to_dict()
    sent = ev[ev.llm >= 0]
    esc = sent[sent.escalated]
    bt = pd.DataFrame(batches)
    summary = {"pool": pool, "seed": cfg["seed"], "backend": lc["backend"], "model": lc["model_id"] if lc["backend"] == "hf" else "stub (not an LLM)",
               "strength": cen["strength"], "rows_sampled": len(ev), "rows_verified": len(sent),
               "valid_first_try": float(sent.first_try_valid.mean()), "valid_after_retries": float(sent.valid.mean()), "fallbacks": int((~sent.valid).sum()),
               "grounded": float(sent.grounded[sent.valid].mean()), "payload_bytes_mean": float(sent.payload_bytes.mean()),
               "llm_calls_per_1k_full_stream_e5": calls, "llm_calls_per_1k_e6": 1000.0,
               "batch_seconds": latency_summary(1000 * bt.seconds.to_numpy()) if len(bt) else None, "rows_per_second": float(bt.rows.sum() / bt.seconds.sum()) if len(bt) else None,
               "llm_accuracy_on_escalated": float((esc.llm == esc.y).mean()) if len(esc) else None, "edge_accuracy_on_escalated": float((esc.pred == esc.y).mean()) if len(esc) else None,
               "llm_says_override_on_escalated": float((esc.llm != esc.pred).mean()) if len(esc) else None, "d3_allows_on_escalated": float(esc.d3.mean()) if len(esc) else None,
               "mean_by_setting": tab.groupby(["rows_set", "setting", "method"])[["macro_f1", "fp_rate", "miss_rate"]].mean().round(4).reset_index().to_dict("records"),
               "overrides_by_setting": ovr.groupby(["setting", "method"])[["overrides", "fixed", "broke"]].sum().reset_index().to_dict("records")}
    tab.round(6).to_csv(rep / "scores.csv", index=False)
    ovr.to_csv(rep / "overrides.csv", index=False)
    bt.round(4).to_csv(rep / "batches.csv", index=False)
    (rep / "summary.json").write_text(dumps(summary, indent=2), encoding="utf-8")
    write_report(rep, summary, tab, ovr, cfg["timeline"]["windows"])


NAMES = {"pred": "E4 edge", "e5": "E5 gated, D3", "e5_free": "gated, LLM label whenever valid", "e6": "E6 always-on, D3", "e6_free": "always-on, LLM label whenever valid"}


def write_report(rep, s, tab, ovr, windows) -> None:
    fc = rep / "fit_check.json"
    check = json.loads(fc.read_text(encoding="utf-8")) if fc.exists() else None
    stub = s["backend"] == "stub"
    lines = [f"# M11: LLM verification of E4 (`e4_added`, D11), `{s['pool']}` pool", "",
             f"Seed {s['seed']}. Generated by `scripts/m11_verify.py`. Verifier: **{s['model']}**. Gate: E4's, adaptive strength {s['strength']:g} (README, M11)."]
    if stub:
        lines += ["", "**The stub is a nearest-centroid rule, not an LLM. These numbers test the pipeline and are not results.**"]
    if check:
        lines += ["", f"**Models against M9:** the rerun `e4_added` (K = {check['k']}) differs from M9's report by at most {check['largest_difference']:.6f} macro-F1 in any window."]
    lb = s["batch_seconds"] or {}
    lines += ["", f"**Rows:** {s['rows_sampled']:,} sampled, {s['rows_verified']:,} sent to the verifier. **Valid replies:** {s['valid_first_try']:.4f} on the first try, {s['valid_after_retries']:.4f} after retries ({s['fallbacks']} fallbacks). "
              f"**Grounded** (every reason names a feature or class from the evidence): {s['grounded']:.3f} of valid replies.",
              f"**Payload:** {s['payload_bytes_mean']:.0f} bytes on average. **Batch latency:** mean {lb.get('mean_ms', float('nan')):.0f} ms, P95 {lb.get('p95_ms', float('nan')):.0f} ms; {s['rows_per_second'] or 0:.2f} rows per second.",
              "**LLM calls per 1k rows (E5, full stream):** " + ", ".join(f"{k} {v}" for k, v in s["llm_calls_per_1k_full_stream_e5"].items()) + "; E6: 1000.", "",
              f"**On escalated rows:** the verifier's label is right on {s['llm_accuracy_on_escalated'] or 0:.3f} of them and the edge's on {s['edge_accuracy_on_escalated'] or 0:.3f}. "
              f"The verifier disagrees with the edge on {s['llm_says_override_on_escalated'] or 0:.3f}, and D3 allows the change on {s['d3_allows_on_escalated'] or 0:.3f}.", "",
              "## Mean over windows", "", "| rows | setting | method | macro-F1 | false-positive rate | miss rate |", "|---|---|---|---:|---:|---:|"]
    for r in s["mean_by_setting"]:
        lines.append(f"| {r['rows_set']} | {r['setting']} | {NAMES[r['method']]} | {r['macro_f1']:.4f} | {r['fp_rate']:.4f} | {r['miss_rate']:.4f} |")
    lines += ["", "`sample`: up to the configured rows per client and window; `always_on`: the first rows of that sample, every one sent to the verifier. "
              "`boundary`: the model through t−1 on window t's rows (H3's high-drift setting); `own window`: the model on its own window.", "",
              "## Overrides, summed over windows", "", "| setting | method | changed | fixed an edge error | broke a correct label |", "|---|---|---:|---:|---:|"]
    for r in s["overrides_by_setting"]:
        lines.append(f"| {r['setting']} | {NAMES[r['method']]} | {r['overrides']} | {r['fixed']} | {r['broke']} |")
    lines += ["", "## By window, sampled rows", "", "| setting | window | method | macro-F1 | false-positive rate | benign rows |", "|---|---|---|---:|---:|---:|"]
    for r in tab[tab.rows_set == "sample"].itertuples():
        fp = "n/a" if pd.isna(r.fp_rate) else f"{r.fp_rate:.4f}"
        lines.append(f"| {r.setting} | {windows[int(r.eval_window)]} | {NAMES[r.method]} | {r.macro_f1:.4f} | {fp} | {r.benign_rows} |")
    lines += ["", "Stores: `evidence.parquet`, `verified.parquet`, `scores.csv`, `overrides.csv`, `batches.csv`, `stream.csv`, `llm_log.jsonl` (every prompt and reply)."]
    (rep / "VERIFY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--only", choices=["fit", "llm"], default=None)
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    pool = args.pool or ("dev" if cfg["resources"]["use_dev_sample"] else "full")
    rep = REPO_ROOT / "reports" / "m11" / pool / (args.tag or "")
    rep.mkdir(parents=True, exist_ok=True)
    with Run(cfg, f"m11_verify_{pool}") as run:
        if args.only != "llm":
            run_fit(cfg, pool, args.tag, rep, run.logger)
        if args.only != "fit":
            run_llm(cfg, pool, rep, run.logger)
        run.logger.info("report -> %s", rep)


if __name__ == "__main__":
    main()
