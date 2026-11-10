"""M12: LLM adaptation-policy agent on E4 (`e4_added`, D11). Three arms on the same rows and seeds:
  none       E4 as it is (the default policy at every decision), checked against M9's e4_added scores;
  heuristic  a fixed rule-based policy on the same evidence;
  llm        the policy the LLM picks (llm.backend: hf on a GPU; policy.backend=heuristic routes the heuristic through the LLM path).
Each client decides once per window, at its first update point (an ADWIN trigger, else the window's end). See README, M12.

Usage: python scripts/m12_policy.py [--pool dev|full] [--env ...] [--tag seed43] [--set ...] [--only none|heuristic|llm] [--report]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from m7_distill import LOCAL, edge_cfg   # noqa: E402
from m9_e4 import HEADLINE, StreamClients, run_pass   # noqa: E402

from ids.data.io import model_features   # noqa: E402
from ids.data.schema import CATEGORIES   # noqa: E402
from ids.drift.adwin import Adwin   # noqa: E402
from ids.drift.shift import bin_edges, bin_shares, psi   # noqa: E402
from ids.eval.explain import ensemble_shap, ranking_change   # noqa: E402
from ids.eval.harness import Evaluator, client_clusters, load_test_sets   # noqa: E402
from ids.eval.system import latency_summary   # noqa: E402
from ids.fl.distill import reliability_from_val   # noqa: E402
from ids.llm.policy import DEFAULT, FOCUS_WEIGHT, GROUPS, PRESERVE_WEIGHT, STYLE, HeuristicBackend, decide, heuristic   # noqa: E402
from ids.llm.verifier import make_backend   # noqa: E402
from ids.models.edge import val_split   # noqa: E402
from ids.models.incremental import IncrementalEdge   # noqa: E402
from ids.partition.io import load_dpub, load_rows   # noqa: E402
from ids.utils import Run, load_config, rng, set_global_seed   # noqa: E402
from ids.utils.config import REPO_ROOT   # noqa: E402
from ids.utils.jsonl import JsonlWriter, dumps   # noqa: E402
from ids.utils.run import memory_note   # noqa: E402

ARMS = ("none", "heuristic", "llm")


def r4(v) -> float:
    return round(float(v), 4)


def share(v: np.ndarray) -> dict:
    u, n = np.unique(v, return_counts=True)
    return {CATEGORIES[k]: r4(x / len(v)) for k, x in zip(u, n)}


class PolicyClients(StreamClients):
    """e4_added's clients, with a policy per client and window applied to that window's updates."""

    def __init__(self, cfg, window_rows, k, arm, backend, feats, log, jsonl):
        super().__init__(cfg, window_rows, k=k, trigger="error", end_all=True)
        self.arm, self.backend, self.feats, self.log, self.jsonl = arm, backend, feats, log, jsonl
        self.pc = cfg["policy"]
        self.policy, self.prev, self.decisions = {}, {c: None for c in self.ids}, []
        self.seen = {c: set() for c in self.ids}
        self.gi = {g: [feats.index(f) for f in fs] for g, fs in GROUPS.items()}

    def evidence(self, c, t, i, j) -> dict:
        """What the client reports to the cloud at its decision point: summaries only, no rows."""
        X = np.concatenate([x for x, _ in self.pending[c]])
        y = np.concatenate([v for _, v in self.pending[c]])
        r = rng(self.seed, "m12_evidence", c, t)
        take = r.permutation(len(y))[: self.pc["evidence_rows"]]
        Xs, ys = X[take], y[take]
        m = self.models[c]
        err_new = float((m.predict_components(Xs)["calibrated"].argmax(1) != ys).mean())
        err_val = float((self.val[c]["calibrated"].argmax(1) != self.val[c]["y"]).mean())
        base = np.concatenate(self.baseline[c]) if self.baseline[c] else None
        p = np.zeros(len(self.feats))
        if base is not None:
            for f in range(len(self.feats)):
                e = bin_edges(base[:, f], self.cfg["drift"]["shift_bins"])
                p[f] = psi(bin_shares(base[:, f], e), bin_shares(Xs[:, f], e))
        n = self.pc["shap_rows"]
        phi_new = np.abs(ensemble_shap(m, Xs[:n])["phi"]).mean(0)
        if len(self.buffers[c]):
            bx, _ = self.buffers[c].get()
            phi_old = np.abs(ensemble_shap(m, bx[r.permutation(len(bx))[:n]])["phi"]).mean(0)
            sd = ranking_change(phi_new, phi_old, 5)["change"]
        else:
            sd = 0.0
        bl = self.buffers[c].get()
        return {"adwin_fired": j is not None, "fired_at_round_share": None if j is None else r4((j + 1) / self.sc["chunks"]),
                "rows_new": int(len(y)), "error_rate_new_rows": r4(err_new), "error_rate_validation": r4(err_val),
                "new_classes": [CATEGORIES[k] for k in sorted(set(np.unique(y).tolist()) - self.seen[c])],
                "class_share_new_rows": share(ys), "class_share_buffer": share(bl[1]) if bl is not None else {},
                "per_class_f1_validation": {CATEGORIES[k]: r4(self.reliability[i][k]) for k in sorted(self.seen[c])},
                "feature_shift_share": r4((p > self.cfg["drift"]["psi_alert"]).mean()) if base is not None else 0.0,
                "most_shifted_features": [self.feats[f] for f in np.argsort(-p, kind="stable")[:5]] if base is not None else [],
                "shap_drift": r4(sd), "top_shap_features_new_rows": [self.feats[f] for f in np.argsort(-phi_new, kind="stable")[:5]],
                "previous_policy": self.prev[c]}

    def decide_for(self, c, t, i, j) -> dict:
        tic = time.perf_counter()
        if self.arm == "none":
            rec = {"policy": dict(DEFAULT), "valid": True, "attempts": 0, "first_try_valid": True}
            ev = None
        else:
            ev = self.evidence(c, t, i, j)
            if self.arm == "heuristic":
                rec = {"policy": heuristic(ev), "valid": True, "attempts": 0, "first_try_valid": True}
            else:
                rec = decide(ev, self.backend, self.pc["max_retries"], self.jsonl, key=f"t{t}_c{c}")
        pol = rec["policy"]
        self.policy[c] = pol
        self.prev[c] = {k: v for k, v in pol.items() if k != "reasons"}
        self.decisions.append({"train_through": t, "client": c, "round": j, **rec, "seconds": round(time.perf_counter() - tic, 3), "evidence": ev})
        return pol

    def run_window(self, t, on_trigger) -> None:
        """M9's run_window, with the decision at the first update point. Urgency low skips that update: at a trigger, the
        window's end still updates; at the end, the rows wait for the next window (as M9's e4 does)."""
        self.updated_now = set()
        for i, c in enumerate(self.ids):
            Xn, yn = self.window_rows[t][c]
            if t == 0:
                self.policy[c] = dict(DEFAULT)
                self.fit_on(c, t, i, Xn, yn, ("m6", LOCAL, c, t))
                self.updates.append({"train_through": t, "client": c, "round": None, "rows": len(Xn)})
                continue
            self.policy.pop(c, None)
            order = rng(self.seed, "m9_order", c, t).permutation(len(Xn))
            fired = False
            for j, idx in enumerate(np.array_split(order, self.sc["chunks"])):
                self.pending[c].append((Xn[idx], yn[idx]))
                if fired:
                    continue
                if self.rise_in(c, self.models[c].predict_components(Xn[idx]), yn[idx]):
                    fired = True
                    if self.decide_for(c, t, i, j)["update_urgency"] != "low":
                        self._fit_pending(c, t, i, j)
                        on_trigger(c, j)
            if self.pending[c]:
                pol = self.policy.get(c) or self.decide_for(c, t, i, None)
                if fired or pol["update_urgency"] != "low":
                    self._fit_pending(c, t, i, None)

    def fit_on(self, c, t, i, Xn, yn, key) -> None:
        """M9's fit_on with the client's current policy applied; the default policy changes nothing."""
        pol = self.policy.get(c, DEFAULT)
        if len(self.buffers[c]):
            bx, by = self.buffers[c].get()
            X, y = np.concatenate([Xn, bx]), np.concatenate([yn, by])
        else:
            X, y = Xn, yn
        vm = val_split(y, self.cfg["models"]["val_frac"], self.seed, key)
        row_scale = None
        if pol["preserve_old_behavior"] and len(y) > len(Xn):
            rs = np.ones(len(y))
            rs[len(Xn):] = PRESERVE_WEIGHT
            row_scale = rs[~vm]
        fs = None
        if pol["feature_focus_groups"]:
            fs = np.ones(len(self.feats))
            for g in pol["feature_focus_groups"]:
                fs[self.gi[g]] = FOCUS_WEIGHT
        if pol["update_scope"] == "full" and self.models[c].n_updates:
            self.models[c] = IncrementalEdge(edge_cfg(self.cfg, len(CATEGORIES)), self.cfg["incremental"]["new_trees"])
        lr, ts = STYLE[pol["training_style"]]
        self.models[c].update(X[~vm], y[~vm], X[vm], y[vm], allowed=self.allowed_features(c, t, X[~vm], y[~vm]),
                              row_scale=row_scale, feature_scale=fs, lr_scale=lr, tree_scale=ts)
        keys = None
        if pol["buffer_selection"] == "boundary":
            s = np.sort(self.models[c].predict_components(Xn)["calibrated"], 1)
            keys = s[:, -1] - s[:, -2]          # margin: the least sure rows are kept first
        self.buffers[c].add(Xn, yn, keys)
        self.seen[c] |= set(np.unique(y[~vm]).tolist())
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


def summarize(ev, name, eps) -> dict:
    s = ev.summary(name, HEADLINE, eps=eps)
    M = ev.matrix(name, HEADLINE)
    nxt = {int(t): float(M.loc[t, t + 1]) for t in M.index if t + 1 in M.columns and not pd.isna(M.loc[t, t + 1])}
    return {**{k: s[k] for k in ("macro_f1_by_window", "window_avg_macro_f1", "window_avg_macro_f1_excl_t0", "final_macro_f1", "mean_forgetting_by_class")},
            "next_window_macro_f1": nxt, "next_window_avg": float(np.mean(list(nxt.values()))) if nxt else None}


def run_arm(cfg, pool, tag, arm, rep, log) -> None:
    tl, feats, target = cfg["timeline"]["headline"], model_features(cfg), cfg["labels"]["headline_target"]
    m9 = json.loads((REPO_ROOT / "reports" / "m9" / pool / (tag or "") / "summary.json").read_text(encoding="utf-8"))
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
    pc = cfg["policy"]
    backend = None
    if arm == "llm":
        backend = HeuristicBackend() if pc["backend"] == "heuristic" else make_backend({**cfg["llm"], "backend": pc["backend"]})
    log.info("[%s] rows loaded, K = %d, backend %s: %s", arm, m9["k"], getattr(backend, "name", None), memory_note())
    jsonl = JsonlWriter(rep / f"llm_log_{arm}.jsonl") if arm == "llm" else None
    clients = PolicyClients(cfg, window_rows, m9["k"], arm, backend, feats, log, jsonl)
    name = f"m12_{arm}"
    out = run_pass(cfg, name, window_rows, X_pub, y_pub, intro, ev, log, k=m9["k"], clients=clients)
    s = summarize(ev, name, cfg["eval"]["recover_eps"])
    if arm == "none":
        then = next(x for x in m9["summaries"] if x["method"] == "e4_added")["macro_f1_by_window"]
        s["against_m9_e4_added"] = max(abs(s["macro_f1_by_window"][int(t)] - v) for t, v in then.items())
        log.info("[none] largest difference from M9's e4_added per window: %.6f", s["against_m9_e4_added"])
    s["updates"] = out["updates"]
    s["fit"] = out["fit"]
    (rep / f"arm_{arm}.json").write_text(dumps(s, indent=2), encoding="utf-8")
    with open(rep / f"decisions_{arm}.jsonl", "w", encoding="utf-8") as f:
        for d in clients.decisions:
            f.write(dumps(d) + "\n")


def report(rep, pool, cfg) -> None:
    arms = {a: json.loads((rep / f"arm_{a}.json").read_text(encoding="utf-8")) for a in ARMS if (rep / f"arm_{a}.json").exists()}
    dec = {a: pd.read_json(rep / f"decisions_{a}.jsonl", lines=True) for a in arms if (rep / f"decisions_{a}.jsonl").exists()}
    windows = cfg["timeline"]["windows"]
    L = [f"# M12: adaptation-policy agent on E4 (`e4_added`), `{pool}` pool", "", f"Seed {cfg['seed']}. Generated by `scripts/m12_policy.py`. "
         f"Policy backend for the llm arm: **{cfg['policy']['backend'] if cfg['policy']['backend'] != 'hf' else cfg['llm']['model_id']}**.", ""]
    if "none" in arms:
        L.append(f"**The none arm against M9's `e4_added`:** largest difference {arms['none']['against_m9_e4_added']:.6f} macro-F1 per window.")
    if cfg["policy"]["backend"] == "heuristic":
        L.append("**The llm arm ran the heuristic through the LLM path (prompt, parse, log): a pipeline check, not an LLM result.**")
    L += ["", "| arm | window avg | avg excl. t0 | final | next-window avg | class forgetting |", "|---|---:|---:|---:|---:|---:|"]
    for a, s in arms.items():
        L.append(f"| {a} | {s['window_avg_macro_f1']:.4f} | {s['window_avg_macro_f1_excl_t0']:.4f} | {s['final_macro_f1']:.4f} | {s['next_window_avg'] or 0:.4f} | {s['mean_forgetting_by_class'] or 0:.4f} |")
    L += ["", "| arm | " + " | ".join(windows) + " |", "|---|" + "---:|" * len(windows)]
    for a, s in arms.items():
        L.append(f"| {a} | " + " | ".join(f"{s['macro_f1_by_window'][str(t)]:.4f}" for t in range(len(windows))) + " |")
    summary = {"pool": pool, "seed": cfg["seed"], "backend": cfg["policy"]["backend"], "arms": {a: {k: v for k, v in s.items() if k not in ("updates", "fit")} for a, s in arms.items()}}
    for a, d in dec.items():
        if a == "none" or not len(d):
            continue
        pol = pd.json_normalize(d.policy)
        st = {"decisions": len(d), "valid": float(d.valid.mean()), "first_try_valid": float(d.first_try_valid.mean()),
              "seconds": latency_summary(1000 * d.seconds.to_numpy()), "at_trigger": float(d["round"].notna().mean())}
        st["shares"] = {k: pol[k].value_counts(normalize=True).round(3).to_dict() for k in ("update_urgency", "training_style", "preserve_old_behavior", "buffer_selection", "update_scope")}
        st["focus_groups"] = pd.Series([g for x in pol.feature_focus_groups for g in x]).value_counts().to_dict()
        summary[f"decisions_{a}"] = st
        L += ["", f"## Decisions, {a} arm", "", f"{st['decisions']} decisions ({st['at_trigger']:.2f} at an ADWIN trigger); valid {st['valid']:.3f}, first try {st['first_try_valid']:.3f}; "
              f"mean {st['seconds']['mean_ms'] / 1000:.1f} s per decision.", ""]
        for k, v in st["shares"].items():
            L.append(f"- `{k}`: " + ", ".join(f"{x} {y:.2f}" for x, y in v.items()))
        L.append("- `feature_focus_groups`: " + (", ".join(f"{g} {n}" for g, n in st["focus_groups"].items()) or "none"))
    if "heuristic" in dec and "llm" in dec and len(dec["llm"]) == len(dec["heuristic"]):
        same = [a == b for a, b in zip(pd.json_normalize(dec["llm"].policy).drop(columns="reasons").to_dict("records"),
                                       pd.json_normalize(dec["heuristic"].policy).drop(columns="reasons").to_dict("records"))]
        summary["llm_equals_heuristic"] = float(np.mean(same))
        L += ["", f"**llm and heuristic arms, same decision on** {np.mean(same):.3f} of decision points (meaningful while both arms reach the same points)."]
    L += ["", "Stores: `arm_<arm>.json`, `decisions_<arm>.jsonl` (each decision with its evidence), `llm_log_llm.jsonl` (every prompt and reply)."]
    (rep / "summary.json").write_text(dumps(summary, indent=2), encoding="utf-8")
    (rep / "POLICY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--only", choices=ARMS, default=None)
    ap.add_argument("--report", action="store_true", help="only rebuild POLICY.md from the arms' files")
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    pool = args.pool or ("dev" if cfg["resources"]["use_dev_sample"] else "full")
    rep = REPO_ROOT / "reports" / "m12" / pool / (args.tag or "")
    rep.mkdir(parents=True, exist_ok=True)
    with Run(cfg, f"m12_policy_{pool}") as run:
        if not args.report:
            for arm in ARMS if args.only is None else (args.only,):
                run_arm(cfg, pool, args.tag, arm, rep, run.logger)
        report(rep, pool, cfg)
        run.logger.info("report -> %s", rep)


if __name__ == "__main__":
    main()
