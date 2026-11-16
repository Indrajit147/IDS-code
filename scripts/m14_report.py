"""M14 report: every tier-1 table and figure from stored per-client confusion counts under reports/, no models.

Reads each source's confusions.csv (slices csv_client:<c>, written by the evaluator since M14) and, for M11,
client_counts.csv (built here once from the gitignored verified.parquet). Checks that every source reproduces
its own stored window average, lists each source's run and commit, and refuses to run when a source run was
dirty in a file outside reports/. Writes reports/m14/full/: REPORT.md, headline.csv, ablation.csv,
hypotheses.csv, sources.csv, macro_f1_by_window.png.

    python scripts/m14_report.py [--draws 10000]
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ids.data.schema import CATEGORIES
from ids.eval.plots import fig_curves
from ids.eval.stats import (client_draws, false_positive_rate_cm, holm, macro_f1_cm, pair_scores,
                            paired_bootstrap, series_score)
from ids.utils.config import REPO_ROOT, load_config

REP, RUNS = REPO_ROOT / "reports", REPO_ROOT / "results" / "runs"
SEEDS, WINDOWS, C = (42, 43, 44), range(6), 15
K = len(CATEGORIES)
FLOOR = 0.005        # size floor for a macro-F1 difference (README, M14 decision rule)

# name -> (milestone, method in its confusions.csv, sub-folder); seed 42 sits at the milestone root
SOURCES = {
    "E1 centralized": ("m4", "E1", ""), "static": ("m4", "static", ""),
    "naive incremental": ("m6", "naive_incremental", ""), "local + rehearsal": ("m6", "local_rehearsal", ""),
    "E2 FedAvg": ("m8", "fedavg", ""), "E2 FedProx": ("m8", "fedprox", ""),
    "E3": ("m7", "e3", ""), "FedMD": ("m7", "fedmd", ""), "e3_guard (addition)": ("m7", "e3_guard", ""),
    "E4 (e4_added)": ("m9", "e4_added", ""), "e4 (ADWIN replaces schedule)": ("m9", "e4", ""),
    "no ADWIN (fixed_k30)": ("m9", "fixed_k30", ""),
    "no feature selection": ("m9", "e4_added_allfeat", "ablation"),
    "no rehearsal": ("m9", "e4_added_noreh", "ablation"),
    "E7 federated GBDT": ("m13", "e7", ""),
    "E3, D_pub benign-only": ("m7", "e3", "dpub_benign"),
    "E4, D_pub benign-only": ("m9", "e4_added", "dpub_benign"),
}
M11 = {"E4 edge": "pred", "E5 gated": "e5", "E6 always-on": "e6"}


def folder(m: str, sub: str, seed: int) -> Path:
    base = REP / m / "full"
    if sub == "dpub_benign":                     # variant runs keep every seed in its own folder
        return base / sub / f"seed{seed}"
    return (base if seed == 42 else base / f"seed{seed}") / sub


# ---------- counts ----------

@lru_cache(maxsize=None)
def _confusions(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["slice"].str.contains("_client:")].copy()
    d["client"] = d["slice"].str.split(":").str[1].astype(int)
    d["i"], d["j"] = d["true"].map(CATEGORIES.index), d["pred"].map(CATEGORIES.index)
    return d


def counts(path: Path, method: str, pairs: list[tuple[int, int]], prefix: str = "csv_client") -> np.ndarray:
    """(P, C, K, K) per-client counts of method on each (train_through, eval_window) pair."""
    d = _confusions(path)
    d = d[(d["model"] == method) & d["slice"].str.startswith(prefix + ":")]
    out = np.zeros((len(pairs), C, K, K))
    for p, (t, w) in enumerate(pairs):
        sel = d[(d["train_through"] == t) & (d["eval_window"] == w)]
        if sel.empty:
            raise ValueError(f"{path}: no per-client counts for {method} at ({t}, {w})")
        np.add.at(out[p], (sel["client"].to_numpy(), sel["i"].to_numpy(), sel["j"].to_numpy()), sel["n"].to_numpy())
    return out


def current_pairs(path: Path, method: str) -> list[tuple[int, int]]:
    """The newest model at each window (drift_metrics.current_series): static keeps its t0 model."""
    ts = sorted(_confusions(path).query("model == @method")["train_through"].unique())
    return [(max(t for t in ts if t <= w), w) for w in WINDOWS]


def stack(name: str) -> dict:
    """Counts of one source over the seeds: current series and next-window pairs (for time-to-recover)."""
    m, method, sub = SOURCES[name]
    cur, nxt = [], []
    for s in SEEDS:
        path = folder(m, sub, s) / "confusions.csv"
        pairs = current_pairs(path, method)
        cur.append(counts(path, method, pairs))
        if pairs[-1][0] == 5:                    # retrained every window
            nxt.append(counts(path, method, [(t, t + 1) for t in range(5)]))
        check_window_average(name, s, path, method, macro_f1_cm(cur[-1].sum(1)).mean())
    return {"cur": np.stack(cur), "next": np.stack(nxt) if nxt else None}


def check_window_average(name, seed, path, method, ours):
    summ = json.loads((path.parent / "summary.json").read_text())["summaries"]
    ref = [x["window_avg_macro_f1"] for x in summ if x["method"] == method and x["slice"] == "csv_only"]
    assert ref and abs(ref[0] - ours) < 1e-9, (name, seed, ref, ours)


def m11_client_counts(seed: int) -> Path:
    """Per-client counts of M11's sampled rows, written once from verified.parquet (gitignored)."""
    d = folder("m11", "", seed)
    out = d / "client_counts.csv"
    if out.exists():
        return out
    v = pd.read_parquet(d / "verified.parquet")
    rows = []
    for col in M11.values():
        for rows_set, sel in (("sample", v), ("always_on", v[v["always_on"].astype(bool)])):
            g = sel.groupby(["train_through", "eval_window", "client", "y", col]).size()
            for (t, w, c, y, p), n in g.items():
                rows.append((col, t, w, f"{rows_set}_client:{c}", CATEGORIES[y], CATEGORIES[p], n))
    pd.DataFrame(rows, columns=["model", "train_through", "eval_window", "slice", "true", "pred", "n"]).to_csv(out, index=False)
    return out


def m11_stack(col: str, setting: str, rows_set: str) -> np.ndarray:
    pairs = [(w - 1, w) for w in range(1, 6)] if setting == "boundary" else [(w, w) for w in WINDOWS]
    out = []
    for s in SEEDS:
        path = m11_client_counts(s)
        out.append(counts(path, col, pairs, prefix=f"{rows_set}_client"))
        ref = [x for x in json.loads((path.parent / "summary.json").read_text())["mean_by_setting"]
               if x["rows_set"] == rows_set and x["setting"] == setting and x["method"] == col]
        ours = macro_f1_cm(out[-1].sum(1)).mean(), false_positive_rate_cm(out[-1].sum(1)).mean()
        assert ref and abs(ref[0]["macro_f1"] - ours[0]) < 6e-5 and abs(ref[0]["fp_rate"] - ours[1]) < 6e-5, (col, s, ref, ours)
    return np.stack(out)


# ---------- statistics on (B,) draws ----------

def final_t5(c, w):
    return pair_scores(c[:, -1:], w).mean(-1).mean(-1)


def variance_t1_t5(c, w):
    return pair_scores(c, w).mean(1)[:, 1:].var(-1, ddof=1)


def fp_rate(c, w):
    return series_score(c, w, false_positive_rate_cm)


def ttr_stat(eps: float):
    """Mean over windows d = 1..5 of M4's time-to-recover on the seed-averaged scores. A drop that never
    recovers counts as one window past the timeline's end (6 - d + 1): an addition, so the mean is finite."""
    def stat(both, w):
        cur, nxt = both
        mc, mn = pair_scores(cur, w).mean(1), pair_scores(nxt, w).mean(1)     # (B, 6), (B, 5)
        out = np.zeros(len(w))
        for d in range(1, 6):
            pre, cold = mc[:, d - 1], mn[:, d - 1]
            back = mc[:, d:] >= (pre - eps)[:, None]                          # models updated at or after d
            first = np.where(back.any(1), back.argmax(1) + 1, 6 - d + 1)
            out += np.where(cold >= pre - eps, 0, first)
        return out / 5
    return stat


def boot_two(a, b, n, stat):
    """paired_bootstrap for stats that need the (cur, next) pair of arrays."""
    w = client_draws(C, n, 0)
    d = stat(a, w) - stat(b, w)
    rest = d[1:]
    p = min(1.0, 2 * min(float((rest <= 0).mean()), float((rest >= 0).mean())))
    return {"diff": float(d[0]), "lo": float(np.quantile(rest, 0.025)), "hi": float(np.quantile(rest, 0.975)), "p": p, "draws": n - 1}


def per_seed(c, stat):
    """Observed stat per seed: mean and sample std over seeds."""
    v = np.array([stat(c[i:i + 1], np.ones((1, C)))[0] for i in range(len(c))])
    return v.mean(), v.std(ddof=1), v


# ---------- provenance ----------

def provenance(d: Path) -> dict:
    """The run that wrote folder d: the run id its markdown cites, else the latest run.log saying 'report -> d'."""
    rel = d.relative_to(REPO_ROOT).as_posix()
    rid = None
    for md in sorted(d.glob("*.md")):
        if md.name != "SEEDS.md" and (m := re.search(r"\d{8}-\d{6}_m\d+_[a-z0-9_]+", md.read_text(encoding="utf-8"))):
            rid = m.group(0)
            break
    if rid is None and RUNS.exists():
        hits = [r for r in sorted(RUNS.iterdir()) if (r / "run.log").exists() and
                re.search(r"report -> \S*?" + re.escape(rel) + r"\s*$", (r / "run.log").read_text(encoding="utf-8", errors="ignore"), re.M)]
        rid = hits[-1].name if hits else None
    env = RUNS / str(rid) / "environment.json"
    if rid is None or not env.exists():
        return {"folder": rel, "run": rid or "not found", "commit": "not recorded locally", "dirty_outside_reports": ""}
    e = json.loads(env.read_text())
    # porcelain lines "XY path" (the first may have lost its leading space); renames end in "-> path"
    bad = [x for x in e.get("git_changed", []) if not x.split()[-1].strip('"').startswith("reports/")]
    if e.get("git_dirty") and not e.get("git_changed"):
        bad = ["dirty, changed files not recorded"]
    return {"folder": rel, "run": rid, "commit": e["git_commit"][:7], "dirty_outside_reports": "; ".join(bad)}


# ---------- report ----------

def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def verdict(r, floor):
    excl = r["lo"] > 0 or r["hi"] < 0
    big = floor is None or abs(r["diff"]) >= floor
    return "yes" if excl and big and r.get("p_holm", r["p"]) < 0.05 else "no"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10_000)
    args = ap.parse_args()
    n, eps = args.draws + 1, load_config()["eval"]["recover_eps"]
    out = REP / "m14" / "full"
    out.mkdir(parents=True, exist_ok=True)

    # provenance first: refuse on a dirty source
    folders = {folder(m, sub, s) for m, _, sub in SOURCES.values() for s in SEEDS} | {folder("m11", "", s) for s in SEEDS}
    prov = pd.DataFrame([provenance(d) for d in sorted(folders)])
    prov.to_csv(out / "sources.csv", index=False)
    dirty = prov[prov["dirty_outside_reports"] != ""]
    if not dirty.empty:
        raise SystemExit(f"refusing: dirty source runs\n{dirty.to_string()}")

    S = {k: stack(k) for k in SOURCES}

    # headline and ablation tables: mean ± std over seeds, and the seed-averaged score with its client-bootstrap interval
    w = client_draws(C, n, 0)
    def row(name, c):
        m, sd, _ = per_seed(c, series_score)
        f, fsd, _ = per_seed(c, final_t5)
        b = series_score(c, w)
        return {"method": name, "window_avg": m, "window_avg_std": sd, "lo": np.quantile(b[1:], 0.025),
                "hi": np.quantile(b[1:], 0.975), "t5": f, "t5_std": fsd}
    headline = pd.DataFrame([row(k, S[k]["cur"]) for k in SOURCES if "D_pub" not in k and k not in
                             ("no ADWIN (fixed_k30)", "no feature selection", "no rehearsal")])
    headline.to_csv(out / "headline.csv", index=False)

    m11 = {(lab, st, rs): m11_stack(col, st, rs) for lab, col in M11.items()
           for st in ("boundary", "own window") for rs in ("sample", "always_on") if (col, rs) != ("e6", "sample")}
    calls = {s: json.loads((folder("m11", "", s) / "summary.json").read_text()) for s in SEEDS}
    llm_rows = []
    for (lab, st, rs), c in m11.items():
        mf, sd, _ = per_seed(c, series_score)
        fp, fsd, _ = per_seed(c, fp_rate)
        llm_rows.append({"rows": rs, "setting": st, "method": lab, "macro_f1": mf, "macro_f1_std": sd, "fp_rate": fp, "fp_rate_std": fsd})
    llm = pd.DataFrame(llm_rows)
    llm.to_csv(out / "llm.csv", index=False)

    base = S["E4 (e4_added)"]["cur"]
    abl = []
    for name, b in (("no ADWIN (fixed_k30)", base), ("no feature selection", base), ("no rehearsal", base),
                    ("E3, D_pub benign-only", S["E3"]["cur"]), ("E4, D_pub benign-only", base)):
        r = paired_bootstrap(S[name]["cur"], b, n)
        abl.append({"ablation": name, "against": "E3" if name.startswith("E3") else "E4", **r})
    for lab, ref, st, rs in (("no LLM verification: E5 gated minus E4 edge", "E4 edge", "boundary", "sample"),
                             ("no LLM verification: E5 gated minus E4 edge", "E4 edge", "own window", "sample"),
                             ("always-on vs gated: E6 minus E5", "E5 gated", "boundary", "always_on"),
                             ("always-on vs gated: E6 minus E5", "E5 gated", "own window", "always_on")):
        a = m11[("E5 gated" if ref == "E4 edge" else "E6 always-on", st, rs)]
        r = paired_bootstrap(a, m11[(ref, st, rs)], n)
        abl.append({"ablation": f"{lab} ({st}, {rs} rows)", "against": ref, **r})
    abl = pd.DataFrame(abl)
    abl["result"] = [verdict(r, FLOOR) for r in abl.to_dict("records")]
    abl.to_csv(out / "ablation.csv", index=False)

    # primary hypotheses, Holm over all five p-values
    e3, e4 = S["E3"], S["E4 (e4_added)"]
    hyp = [
        {"id": "H1a", "test": "E3 minus naive incremental, window-avg macro-F1", **paired_bootstrap(e3["cur"], S["naive incremental"]["cur"], n), "floor": FLOOR},
        {"id": "H1b", "test": "E3 minus static, window-avg macro-F1", **paired_bootstrap(e3["cur"], S["static"]["cur"], n), "floor": FLOOR},
        {"id": "H2a", "test": "E4 minus E3, variance of macro-F1 over t1-t5 (lower is better)", **paired_bootstrap(e4["cur"], e3["cur"], n, stat=variance_t1_t5), "floor": None},
        {"id": "H2b", "test": f"E4 minus E3, mean time-to-recover in windows, eps {eps} (lower is better)",
         **boot_two((e4["cur"], e4["next"]), (e3["cur"], e3["next"]), n, ttr_stat(eps)), "floor": None},
        {"id": "H3", "test": "E5 gated minus E4 edge, benign false-positive rate, boundary windows t1-t5, sampled rows (lower is better)",
         **paired_bootstrap(m11[("E5 gated", "boundary", "sample")], m11[("E4 edge", "boundary", "sample")], n, stat=fp_rate), "floor": None},
    ]
    for h, p in zip(hyp, holm([h["p"] for h in hyp])):
        h["p_holm"] = p
    for h in hyp:
        h["result"] = verdict(h, h["floor"])
    hyp = pd.DataFrame(hyp)
    hyp.to_csv(out / "hypotheses.csv", index=False)
    ttr_obs = {k: ttr_stat(eps)((S[k]["cur"], S[k]["next"]), np.ones((1, C)))[0] for k in ("E3", "E4 (e4_added)")}
    var_obs = {k: variance_t1_t5(S[k]["cur"], np.ones((1, C)))[0] for k in ("E3", "E4 (e4_added)")}

    # per-window figure, seed-averaged current series
    style = {"E3": (0, False), "E4 (e4_added)": (1, False), "local + rehearsal": (2, False), "E1 centralized": (3, False),
             "E7 federated GBDT": (3, True), "static": (1, True)}       # plots.SERIES has 4 slots; dashed = baseline
    fig = {(k, st): pd.Series(macro_f1_cm(S[k]["cur"].sum(2)).mean(0), index=list(WINDOWS)) for k, st in style.items()}
    fig_curves(fig, [f"t{w}" for w in WINDOWS], out / "macro_f1_by_window.png", "csv_only, mean over seeds 42, 43, 44")

    # markdown
    L = ["# M14 tier 1: headline, ablations and hypotheses (`full` pool)", "",
         f"Generated by `scripts/m14_report.py` from stored per-client counts only (`csv_only` rows, D7). Seeds {', '.join(map(str, SEEDS))}. "
         f"Intervals: client bootstrap, {args.draws:,} draws, the same 15-client draw for every seed and both methods, seeds averaged inside each draw. "
         f"A difference counts as a result when its 95% interval excludes 0, its Holm-adjusted p (hypotheses) or raw p (ablations) is under 0.05, "
         f"and, for macro-F1, its size is at least {FLOOR}. Every source reproduces its own stored window average exactly; sources and commits: `sources.csv`.", "",
         "## Headline", "", "| method | window avg (mean ± std over seeds) | 95% interval, seed mean | t5 (mean ± std) |", "|---|---:|---:|---:|"]
    L += [f"| {r.method} | {fmt(r.window_avg)} ± {fmt(r.window_avg_std)} | {fmt(r.lo)} to {fmt(r.hi)} | {fmt(r.t5)} ± {fmt(r.t5_std)} |" for r in headline.itertuples()]
    L += ["", "E5 and E6 are scored on M11's sampled rows (500 per client and window; always-on: the first 50 of them), so they are compared with the E4 edge on the same rows, not with the table above.", "",
          "| rows | setting | method | macro-F1 (mean ± std) | benign false-positive rate (mean ± std) |", "|---|---|---|---:|---:|"]
    L += [f"| {r.rows} | {r.setting} | {r.method} | {fmt(r.macro_f1)} ± {fmt(r.macro_f1_std)} | {fmt(r.fp_rate)} ± {fmt(r.fp_rate_std)} |" for r in llm.itertuples()]
    L += ["", f"LLM calls per 1k flows, E5 (full stream, boundary): {', '.join(str(calls[s]['llm_calls_per_1k_full_stream_e5']['boundary']) for s in SEEDS)}; E6: 1000.", "",
          "## Ablations (paired, window-avg macro-F1 unless stated; difference = ablation minus base)", "",
          "| ablation | base | difference | 95% interval | p | result |", "|---|---|---:|---:|---:|---|"]
    L += [f"| {r.ablation} | {r.against} | {r.diff:+.4f} | {r.lo:+.4f} to {r.hi:+.4f} | {r.p:.4f} | {r.result} |" for r in abl.itertuples()]
    L += ["", "D_pub benign-only moves attack rows from D_pub back to the clients, so its rows are not identical to the headline's; the client draw is paired, the rows within a client are not.",
          "", "## Hypotheses (Holm over the five tests below)", "",
          "| id | test | difference | 95% interval | p | Holm p | result |", "|---|---|---:|---:|---:|---:|---|"]
    L += [f"| {r.id} | {r.test} | {r.diff:+.5f} | {r.lo:+.5f} to {r.hi:+.5f} | {r.p:.4f} | {r.p_holm:.4f} | {r.result} |" for r in hyp.itertuples()]
    L += ["", f"Observed, seed-averaged: variance over t1-t5 E3 {var_obs['E3']:.5f}, E4 {var_obs['E4 (e4_added)']:.5f}; "
          f"mean time-to-recover E3 {ttr_obs['E3']:.2f}, E4 {ttr_obs['E4 (e4_added)']:.2f} windows "
          "(a drop that never recovers counts as one window past t5).", "",
          "Not in tier 1: the τc/τd frontier and H4 (tier 3), the sensitivity variants (tier 2).", "",
          "![per window](macro_f1_by_window.png)", ""]
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
