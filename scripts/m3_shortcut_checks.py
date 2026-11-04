"""M3 evidence for decision D6, the perfect t0 score and the profiling rows.
Writes reports/m3/<pool>/shortcut_checks.{json,md}.

1. IAT A/B: the same E1 model at one window, with and without IAT.
2. t0 separability: how well each single feature splits Benign from DoS at t0.
3. Number/Weight A/B: E1 at t0, t1 and t5 without those two features.
4. Profiling rows: E1 with and without them, and whether the two benign sources can be told apart.

Usage: python scripts/m3_shortcut_checks.py [--pool dev] [--window 1] [--env ...]
"""
from __future__ import annotations

import argparse
import importlib.util

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from ids.data.io import model_features
from ids.data.schema import CATEGORIES
from ids.eval.metrics import detection_metrics
from ids.models.edge import EdgeEnsemble
from ids.partition.build import SRC_CSV, SRC_PROFILING
from ids.partition.io import load_rows
from ids.utils import Run, load_config, set_global_seed
from ids.utils.config import REPO_ROOT
from ids.utils.jsonl import dumps

_spec = importlib.util.spec_from_file_location("m3_baselines", REPO_ROOT / "scripts" / "m3_baselines.py")
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)


def iat_ab(cfg, pool, tl, window, target="y_category"):
    base, out = model_features(cfg), {}
    for name, feats in (("without_iat", base), ("with_iat", [*base, "IAT"])):
        tr = load_rows(cfg, pool, tl, windows=list(range(window + 1)), split="train", features=feats)
        te = load_rows(cfg, pool, tl, windows=[window], split="test", features=feats)
        X, y = tr[feats].to_numpy(np.float32), tr[target].to_numpy()
        vm = m3.val_split(y, cfg["models"]["val_frac"], cfg["seed"], ("wifi", window))
        model = EdgeEnsemble(m3.edge_config(cfg, len(CATEGORIES))).fit(X[~vm], y[~vm], X[vm], y[vm])
        met = detection_metrics(te[target].to_numpy(), model.predict_proba(te[feats].to_numpy(np.float32)), CATEGORIES)
        out[name] = {"n_features": len(feats), "macro_f1": met["macro_f1"],
                     "per_class_f1": {c: d["f1"] for c, d in met["per_class"].items()}}
    return out


def t0_separability(cfg, pool, tl, target="y_category"):
    feats = model_features(cfg)
    tr = load_rows(cfg, pool, tl, windows=[0], split="train", features=feats)
    te = load_rows(cfg, pool, tl, windows=[0], split="test", features=feats)
    scores = {}
    for f in feats:
        tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=cfg["seed"])
        tree.fit(tr[[f]].to_numpy(), tr[target].to_numpy())
        scores[f] = float(f1_score(te[target], tree.predict(te[[f]].to_numpy()), average="macro"))
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def e1_at(cfg, pool, tl, window, feats, keep_profiling=True, target="y_category"):
    """E1 at one window; also the benign recall on each source of benign rows."""
    tr = load_rows(cfg, pool, tl, windows=list(range(window + 1)), split="train", features=feats)
    te = load_rows(cfg, pool, tl, windows=[window], split="test", features=feats)
    if not keep_profiling:
        tr, te = tr[tr["source"] == SRC_CSV], te[te["source"] == SRC_CSV]
    X, y = tr[feats].to_numpy(np.float32), tr[target].to_numpy()
    vm = m3.val_split(y, cfg["models"]["val_frac"], cfg["seed"], ("wifi", window))
    model = EdgeEnsemble(m3.edge_config(cfg, len(CATEGORIES))).fit(X[~vm], y[~vm], X[vm], y[vm])
    yt, p = te[target].to_numpy(), model.predict_proba(te[feats].to_numpy(np.float32))
    met = detection_metrics(yt, p, CATEGORIES)
    src = te["source"].to_numpy()
    recall = {name: float((p[(src == v) & (yt == 0)].argmax(1) == 0).mean())
              for v, name in ((SRC_CSV, "csv"), (SRC_PROFILING, "profiling")) if ((src == v) & (yt == 0)).any()}
    return {"macro_f1": met["macro_f1"], "benign_f1": met["per_class"]["benign"]["f1"], "benign_recall_by_source": recall}


def feature_and_source_checks(cfg, pool, tl, windows=(0, 1, 5)):
    base = model_features(cfg)
    no_nw = [f for f in base if f not in ("Number", "Weight")]
    return {int(w): {"base": e1_at(cfg, pool, tl, w, base), "no_number_weight": e1_at(cfg, pool, tl, w, no_nw),
                     "no_profiling": e1_at(cfg, pool, tl, w, base, keep_profiling=False)} for w in windows}


def benign_source_auc(cfg, pool, tl):
    """Hold-out AUC of telling CSV benign rows from profiling benign rows, class-balanced."""
    base = model_features(cfg)
    d = load_rows(cfg, pool, tl, windows=list(range(len(cfg["timeline"]["windows"]))), split=None, features=base)
    d = d[d["y_category"] == 0]
    idx = np.random.default_rng(cfg["seed"]).permutation(len(d))
    cut = int(0.7 * len(d))
    tr, te = d.iloc[idx[:cut]], d.iloc[idx[cut:]]
    out = {"n_csv": int((d["source"] == SRC_CSV).sum()), "n_profiling": int((d["source"] == SRC_PROFILING).sum())}
    no_nw = [f for f in base if f not in ("Number", "Weight")]
    for name, feats in (("all_features", base), ("no_number_weight", no_nw)):
        clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, class_weight="balanced", random_state=cfg["seed"],
                                 n_jobs=cfg["resources"]["n_jobs"], verbose=-1)
        clf.fit(tr[feats].to_numpy(), tr["source"] == SRC_PROFILING)
        out[f"auc_{name}"] = float(roc_auc_score(te["source"] == SRC_PROFILING, clf.predict_proba(te[feats].to_numpy())[:, 1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--pool", default="dev")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])
    tl = cfg["timeline"]["headline"]
    win = cfg["timeline"]["windows"][args.window]

    with Run(cfg, f"m3_shortcut_checks_{args.pool}") as run:
        ab = iat_ab(cfg, args.pool, tl, args.window)
        sep = t0_separability(cfg, args.pool, tl)
        checks = feature_and_source_checks(cfg, args.pool, tl)
        auc = benign_source_auc(cfg, args.pool, tl)
        result = {"run": run.dir.name, "pool": args.pool, "window": win, "iat_ab": ab, "t0_single_feature_macro_f1": sep,
                  "feature_and_source_checks": checks, "benign_source": auc}
        run.save_json("shortcut_checks.json", result)
        rep = REPO_ROOT / "reports" / "m3" / args.pool
        rep.mkdir(parents=True, exist_ok=True)
        (rep / "shortcut_checks.json").write_text(dumps(result, indent=2), encoding="utf-8")
        a, b = ab["without_iat"], ab["with_iat"]
        rows = "\n".join(f"| {c} | {a['per_class_f1'][c]:.3f} | {b['per_class_f1'][c]:.3f} |" for c in a["per_class_f1"])
        top = "\n".join(f"| {f} | {v:.4f} |" for f, v in list(sep.items())[:8])
        wnames = cfg["timeline"]["windows"]
        labels = {"base": "all 43 features, with profiling rows", "no_number_weight": "without Number and Weight",
                  "no_profiling": "without profiling rows (train and test)"}
        chk = ""
        for w, by in checks.items():
            for k, v in by.items():
                rec = v["benign_recall_by_source"]
                chk += (f"| {wnames[w]} | {labels[k]} | {v['macro_f1']:.4f} | {v['benign_f1']:.3f} | "
                        f"{rec.get('csv', float('nan')):.3f} | {rec['profiling']:.3f} |\n" if "profiling" in rec else
                        f"| {wnames[w]} | {labels[k]} | {v['macro_f1']:.4f} | {v['benign_f1']:.3f} | {rec['csv']:.3f} | n/a |\n")
        (rep / "shortcut_checks.md").write_text(f"""# M3 shortcut checks (`{args.pool}` pool)

Run `{run.dir.name}`. Generated by `scripts/m3_shortcut_checks.py`.

## IAT A/B: E1 at {win}, same model with and without IAT

Macro-F1 goes from **{a['macro_f1']:.4f}** ({a['n_features']} features) to **{b['macro_f1']:.4f}** ({b['n_features']} features).

| class | without IAT | with IAT |
|---|---:|---:|
{rows}

## t0 separability: Benign vs DoS from one feature (depth-2 tree, test-set macro-F1)

| feature | macro-F1 |
|---|---:|
{top}

## Number/Weight and the profiling rows: E1

| window | variant | macro-F1 | benign F1 | benign recall, CSV rows | benign recall, profiling rows |
|---|---|---:|---:|---:|---:|
{chk}
## Can the two benign sources be told apart?

Hold-out AUC of a LightGBM classifier on benign rows only ({auc['n_csv']:,} CSV, {auc['n_profiling']:,} profiling): **{auc['auc_all_features']:.4f}** with all features, **{auc['auc_no_number_weight']:.4f}** without Number and Weight.
""", encoding="utf-8")
        run.logger.info("IAT A/B %s: %.4f -> %.4f; best t0 features %s", win, a["macro_f1"], b["macro_f1"], list(sep.items())[:5])


if __name__ == "__main__":
    main()
