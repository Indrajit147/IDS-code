"""M14 tier 3, from stored logs only (no training, no GPU): the τc/τd frontier, H4 (a) groundedness, H4 (b) schema-valid rate.

Called by scripts/m14_report.py; also runs alone: python scripts/m14_tier3.py
Inputs per seed: reports/m11/full/<seed>/{evidence,verified}.parquet and llm_log.jsonl (gitignored, local),
reports/m12/full/<seed>/decisions_llm.jsonl. Writes into reports/m14/full/: frontier.csv, frontier_budgets.csv,
frontier.png, h4.csv and TIER3.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ids.data.schema import CATEGORIES
from ids.drift.gate import adaptive_taus, escalate
from ids.eval.stats import false_positive_rate_cm, macro_f1_cm
from ids.llm.verifier import parse_reply
from ids.utils.config import REPO_ROOT, load_config

REP = REPO_ROOT / "reports"
SEEDS, BUDGETS = (42, 43, 44), (0.05, 0.10, 0.20, 0.30)
K = len(CATEGORIES)
UP = re.compile(r"\b(high|higher|highest|increase[sd]?|increasing|elevated|large|larger|above)\b", re.I)
DOWN = re.compile(r"\b(low|lower|lowest|decrease[sd]?|decreasing|reduced|small|smaller|below)\b", re.I)


def m11_dir(seed: int) -> Path:
    return REP / "m11" / "full" / ("" if seed == 42 else f"seed{seed}")


def m12_dir(seed: int) -> Path:
    return REP / "m12" / "full" / ("" if seed == 42 else f"seed{seed}")


def window_mean(df: pd.DataFrame, col: str, metric) -> float:
    """The metric per eval window on these rows, then the mean over windows (M11's VERIFY rule)."""
    out = []
    for _, g in df.groupby("eval_window"):
        cm = np.zeros((K, K))
        np.add.at(cm, (g["y"].to_numpy(), g[col].to_numpy()), 1)
        out.append(float(metric(cm)))
    return float(np.mean(out))


# ---------- τc/τd frontier ----------

def frontier(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E4's gate (adaptive thresholds plus the drift hold) swept over a base (τc, τd) grid on M11's sampled rows.
    Escalation rate and share of edge errors escalated per setting; the budget points are chosen on own-window
    rows (in-distribution, as validation) and read out on boundary rows (drifted, out of sample). For the LLM,
    E5 at each chosen point is scored on the always-on rows, where Gemma's reply exists whatever the gate said."""
    taus_c = np.round(np.arange(0.40, 0.99, 0.02), 2)
    taus_d = cfg["gate"]["tune_tau_d"]
    grid, picks = [], []
    for seed in SEEDS:
        d = m11_dir(seed)
        v = pd.read_parquet(d / "verified.parquet")
        strength = json.loads((d / "centroids.json").read_text())["strength"]
        v["setting"] = np.where(v.eval_window == v.train_through, "own window", "boundary")
        err = (v.pred != v.y).to_numpy()
        masks = {}
        for tc in taus_c:
            for td in taus_d:
                a, b = adaptive_taus(tc, td, v.score.to_numpy(), strength)
                m = escalate(v.conf.to_numpy(), v.dis.to_numpy(), a, b, v.hold.to_numpy())
                masks[(tc, td)] = m
                for st in ("own window", "boundary"):
                    s = (v.setting == st).to_numpy()
                    grid.append({"seed": seed, "setting": st, "tau_c": tc, "tau_d": td, "escalation_rate": m[s].mean(),
                                 "error_capture": (m & err & s).sum() / max((err & s).sum(), 1)})
        g = pd.DataFrame([r for r in grid if r["seed"] == seed])
        own = g[g.setting == "own window"]
        for budget in BUDGETS:
            ok = own[own.escalation_rate <= budget]
            best = ok.sort_values(["error_capture", "escalation_rate"], ascending=[False, True]).iloc[0]
            m = masks[(best.tau_c, best.tau_d)]
            v["e5_tau"] = np.where(m & v.d3, v.llm, v.pred)
            ao = v[v.always_on]
            for st in ("own window", "boundary"):
                row = g[(g.setting == st) & (g.tau_c == best.tau_c) & (g.tau_d == best.tau_d)].iloc[0]
                a = ao[ao.setting == st]
                picks.append({"seed": seed, "budget": budget, "setting": st, "tau_c": best.tau_c, "tau_d": best.tau_d,
                              "escalation_rate": row.escalation_rate, "error_capture": row.error_capture,
                              "edge_macro_f1": window_mean(a, "pred", macro_f1_cm), "e5_macro_f1": window_mean(a, "e5_tau", macro_f1_cm),
                              "edge_fp_rate": window_mean(a, "pred", false_positive_rate_cm), "e5_fp_rate": window_mean(a, "e5_tau", false_positive_rate_cm),
                              "llm_changed": int((a.e5_tau != a.pred).sum()), "always_on_rows": len(a)})
    return pd.DataFrame(grid), pd.DataFrame(picks)


def frontier_figure(grid: pd.DataFrame, picks: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from ids.eval.plots import GRID, INK2, SERIES
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, st in enumerate(("own window", "boundary")):
        g = grid[grid.setting == st].groupby(["tau_c", "tau_d"])[["escalation_rate", "error_capture"]].mean()
        g = g.sort_values("escalation_rate")
        upper = g[g.error_capture >= g.error_capture.cummax()]          # best capture reached so far: the frontier
        ax.scatter(g.escalation_rate, g.error_capture, s=6, color=SERIES[i], alpha=0.25, linewidths=0)
        ax.plot(upper.escalation_rate, upper.error_capture, color=SERIES[i], lw=2, label=f"{st} rows")
        p = picks[picks.setting == st].groupby("budget")[["escalation_rate", "error_capture"]].mean()
        ax.scatter(p.escalation_rate, p.error_capture, s=40, color=SERIES[i], edgecolor="white", zorder=3)
    for b in BUDGETS:
        ax.axvline(b, color=GRID, lw=0.8, zorder=0)
        ax.annotate(f"{b:.0%}", (b, 0.02), xytext=(3, 0), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlabel("escalation rate (sampled rows)"); ax.set_ylabel("share of edge errors escalated")
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.0); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("E4 gate frontier, mean over seeds; dots: budget points chosen on own-window rows", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------- H4 (a) and (b) ----------

def reason_grounded(reason: str, top: list[tuple[str, float]]) -> tuple[bool, bool]:
    """(names a top-5 SHAP feature, grounded): grounded also needs a stated direction (up or down words) to match the
    sign of every named feature's SHAP value (towards the edge's label); a reason stating both directions fails."""
    named = [s for f, s in top if re.search(rf"\b{re.escape(f)}\b", reason, re.I)]
    if not named:
        return False, False
    up, down = bool(UP.search(reason)), bool(DOWN.search(reason))
    if up and down:
        return True, False
    if up or down:
        sign = 1 if up else -1
        return True, all(np.sign(s) == sign for s in named)
    return True, True


def h4_verifier(seed: int) -> dict:
    d = m11_dir(seed)
    ev = pd.read_parquet(d / "evidence.parquet")
    feats = json.loads((d / "centroids.json").read_text())["features"]
    replies = {}
    with open(d / "llm_log.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["valid"]:
                replies[r["key"]] = r["output"]
    n_reasons = n_named = n_grounded = n_rows_all = 0
    for key, text in replies.items():
        row = ev.loc[key]
        reply, _ = parse_reply(text, CATEGORIES[int(row.pred)], CATEGORIES, feats)
        assert reply is not None, (seed, key)
        top = [(row[f"f{j}"], row[f"s{j}"]) for j in range(1, 6)]
        res = [reason_grounded(s, top) for s in reply["reasons"]]
        n_reasons += len(res)
        n_named += sum(a for a, _ in res)
        n_grounded += sum(b for _, b in res)
        n_rows_all += all(b for _, b in res)
    v = pd.read_parquet(d / "verified.parquet")
    sent = v[v.llm >= 0]
    return {"seed": seed, "source": "M11 verifier (Gemma 2 2B)", "replies": len(replies), "reasons": n_reasons,
            "names_top5_feature": n_named / n_reasons, "grounded_strict": n_grounded / n_reasons,
            "replies_all_grounded": n_rows_all / len(replies), "grounded_m11_weak": float(sent.grounded[sent.valid].mean()),
            "valid_first_try": float(sent.first_try_valid.mean()), "valid_after_retries": float(sent.valid.mean())}


NO_NEW = re.compile(r"no new class|new_classes (is|are) (empty|none)|without new class", re.I)
NEW = re.compile(r"new class(es)? (is |are )?(detected|appeared|present|found|emerged)", re.I)
NO_ADWIN = re.compile(r"(adwin|drift) (did not|didn't|has not|hasn't) (fire|detect)|no drift (was )?detected|adwin_fired (is )?false", re.I)
ADWIN = re.compile(r"adwin (fired|detected)|drift (was )?detected|adwin_fired (is )?true", re.I)


def h4_policy(seed: int) -> dict | None:
    """M12's window-level reasons carry no SHAP signs: only explicit contradictions of two yes/no facts are checked
    (new classes, ADWIN fired), so the share is a lower bound on contradictions."""
    p = m12_dir(seed) / "decisions_llm.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    n_reasons = n_bad = low_new = has_new = 0
    for r in recs:
        e, reasons = r["evidence"], (r["policy"] or {}).get("reasons", []) if r["valid"] else []
        new, fired = bool(e["new_classes"]), bool(e["adwin_fired"])
        for s in reasons:
            n_reasons += 1
            n_bad += bool((new and NO_NEW.search(s)) or (not new and NEW.search(s)) or (fired and NO_ADWIN.search(s)) or (not fired and ADWIN.search(s)))
        if r["valid"] and new:
            has_new += 1
            low_new += r["policy"]["update_urgency"] == "low"
    return {"seed": seed, "source": "M12 policy (Gemma 2 2B)", "replies": len(recs), "reasons": n_reasons,
            "contradicts_evidence": n_bad / max(n_reasons, 1), "urgency_low_with_new_class": low_new / max(has_new, 1),
            "decisions_with_new_class": has_new,
            "valid_first_try": float(np.mean([r["first_try_valid"] for r in recs])), "valid_after_retries": float(np.mean([r["valid"] for r in recs]))}


def main(out: Path | None = None) -> None:
    out = out or REP / "m14" / "full"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    grid, picks = frontier(cfg)
    grid.to_csv(out / "frontier.csv", index=False)
    picks.to_csv(out / "frontier_budgets.csv", index=False)
    frontier_figure(grid, picks, out / "frontier.png")
    h4 = pd.DataFrame([h4_verifier(s) for s in SEEDS] + [r for s in SEEDS if (r := h4_policy(s))])
    h4.to_csv(out / "h4.csv", index=False)

    P = picks.groupby(["budget", "setting"]).agg(["mean", "std"])
    L = ["# M14 tier 3: τc/τd frontier and H4 (`full` pool)", "",
         "Generated by `scripts/m14_tier3.py` (called by `scripts/m14_report.py`) from M11's and M12's stored rows and logs; nothing is retrained and no LLM is called. Seeds 42, 43, 44 (M12: 42, 43).", "",
         "## τc/τd frontier (D10)", "",
         "E4's gate as run in M11 (adaptive thresholds from the drift score, plus the drift hold), swept over base τc 0.40 to 0.98 (step 0.02) and τd in `gate.tune_tau_d`, "
         "on M11's sampled rows (500 per client and window, so each client and window counts equally; full-stream rates differ). "
         "For each budget the point with the most edge errors escalated at that rate or under is chosen on **own-window** rows and read out on **boundary** rows, "
         "the drifted setting it wasn't chosen on. The LLM columns score E5 at that point on the always-on rows (50 per client and window), where Gemma's reply exists whatever the gate said, with D3 as in M11. Mean ± std over seeds.", "",
         "| budget | setting | τc, τd (per seed) | escalation rate | edge errors escalated | edge macro-F1 | E5 macro-F1 | edge FP rate | E5 FP rate |",
         "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for (b, st), g in picks.groupby(["budget", "setting"], sort=False):
        s = lambda c: f"{P.loc[(b, st)][(c, 'mean')]:.4f} ± {P.loc[(b, st)][(c, 'std')]:.4f}"
        taus = "; ".join(f"{a:.2f}, {d:g}" for a, d in zip(g.tau_c, g.tau_d))
        L.append(f"| {b:.0%} | {st} | {taus} | {s('escalation_rate')} | {s('error_capture')} | {s('edge_macro_f1')} | {s('e5_macro_f1')} | {s('edge_fp_rate')} | {s('e5_fp_rate')} |")
    L += ["", "![frontier](frontier.png)", "", "## H4 (a) groundedness and (b) schema-valid rate", "",
          "**Verifier (M11).** A reason counts as grounded when it names a feature in that row's top-5 SHAP (stored with signs in `evidence.parquet`) and, if it states a direction "
          "(high, low, increased, decreased and similar words), that direction matches the sign of every named feature's SHAP value towards the edge's label; "
          "a reason stating both directions fails, a reason naming no top-5 feature fails. The direction words are matched anywhere in the reason, so a word about a probability "
          "(\"the probability is higher\") is read as a feature direction: the rule is strict and crude, and its effect shows in the gap between the first two columns. "
          "`M11 weak` is M11's check (every reason names a feature or class from the evidence), per reply.", "",
          "| seed | replies | reasons | names a top-5 feature | grounded (strict) | replies with every reason grounded | M11 weak, per reply | valid, first try | valid after retries |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in h4[h4.source.str.startswith("M11")].itertuples():
        L.append(f"| {r.seed} | {r.replies:,} | {r.reasons:,} | {r.names_top5_feature:.3f} | {r.grounded_strict:.3f} | {r.replies_all_grounded:.3f} | {r.grounded_m11_weak:.3f} | {r.valid_first_try:.3f} | {r.valid_after_retries:.4f} |")
    pol = h4[h4.source.str.startswith("M12")]
    if len(pol):
        L += ["", "**Policy agent (M12).** Its reasons are window-level and its evidence has no SHAP signs, so only the weaker check is possible: does a reason explicitly contradict "
              "one of two yes/no facts it was given (new classes present, ADWIN fired). Keyword matching catches only explicit contradictions, so the share is a lower bound. "
              "`urgency low with a new class` is the share of valid decisions that delay the update although a new class is in the window.", "",
              "| seed | decisions | reasons | reasons contradicting the evidence | urgency low with a new class (decisions) | valid, first try | valid after retries |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in pol.itertuples():
            L.append(f"| {r.seed} | {r.replies} | {r.reasons} | {r.contradicts_evidence:.3f} | {r.urgency_low_with_new_class:.3f} ({int(r.decisions_with_new_class)}) | {r.valid_first_try:.3f} | {r.valid_after_retries:.3f} |")
    L.append("")
    (out / "TIER3.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
