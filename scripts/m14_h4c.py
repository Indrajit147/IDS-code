"""M14, H4 (c) for the record: SHAP-only against SHAP + LLM explanations, scored by a person on a blinded sheet.

Not preregistered: the rubric was fixed on 2026-11-17, after M11's reasons had been read (README, M14).
Sample: 100 rows of M11 seed 42 that got a valid Gemma reply, 50 own-window and 50 boundary, drawn with a fixed seed.
Each row gives two texts, a SHAP-only template and Gemma's own text (reasons, features to watch, safety notes);
the 200 texts are shuffled into one sheet, so a scorer never sees a pair side by side. Gemma's prose is still
recognizable, so the sheet hides pairing and order, not the arm.

Scorer items (y/n), in sheet.csv:
  action      names a concrete operator action: what to do and to what (block or rate-limit a source, isolate a
              device, inspect a named host or flow). "Further investigation is recommended" or "monitor X" is not.
  consistent  every claim in the text is supported by the evidence column (the payload Gemma saw); a claim the
              evidence can't settle counts as not supported.
Automatic items (in key.csv, never shown to the scorer):
  label_correct  the label the text gives is the row's true class
  action_lexicon a strict action-verb list (block, isolate, rate-limit, ...), a proxy only: it is not the rubric

Run: python scripts/m14_h4c.py. The first run writes sheet.csv and key.csv; later runs never overwrite the sheet,
they only score it. Fill the two columns with y or n and run again. Don't open key.csv before scoring.
Inputs (gitignored, local): reports/m11/full/{evidence.parquet, llm_log.jsonl}. Writes reports/m14/full/h4c/.
"""
from __future__ import annotations

import json
import re
from math import comb

import numpy as np
import pandas as pd

from ids.data.schema import CATEGORIES
from ids.llm.verifier import parse_reply
from m14_tier3 import REP, m11_dir

OUT = REP / "m14" / "full" / "h4c"
N_PER_SETTING, SEED = 50, 20261117
ACTION = re.compile(r"\b(block|isolat|quarantin|rate[- ]?limit|throttl|firewall|disconnect|shut ?down|blacklist|blocklist"
                    r"|drop (the )?(traffic|packets|flows)|(alert|notify|page) (the )?(operator|admin|team|soc))\w*", re.I)


def shap_text(row) -> str:
    lab = CATEGORIES[int(row.pred)]
    lines = [f"Label: {lab} (probability {row[f'p{int(row.pred)}']:.2f})", "Top features by SHAP:"]
    for j in range(1, 6):
        s = row[f"s{j}"]
        lines.append(f"- {row[f'f{j}']} = {row[f'x{j}']:.4g}, pushes {'towards' if s > 0 else 'away from'} {lab} (SHAP {s:+.3f})")
    return "\n".join(lines)


def llm_text(reply: dict) -> str:
    lines = [f"Label: {reply['final_label']} (confidence {reply['confidence_bucket']})", "Reasons:"]
    lines += [f"- {r}" for r in reply["reasons"]]
    if reply["feature_recommendations"]:
        lines.append("Features to watch: " + ", ".join(reply["feature_recommendations"]))
    if reply["safety_notes"].strip():
        lines.append("Safety notes: " + reply["safety_notes"].strip())
    return "\n".join(lines)


def make() -> None:
    d = m11_dir(42)
    ev = pd.read_parquet(d / "evidence.parquet")
    feats = json.loads((d / "centroids.json").read_text())["features"]
    got = {}
    with open(d / "llm_log.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["valid"]:
                got[r["key"]] = r
    keys = np.array(sorted(got))
    own = ev.loc[keys, "train_through"].to_numpy() == ev.loc[keys, "eval_window"].to_numpy()
    rng = np.random.default_rng(SEED)
    picked = [(k, "own_window") for k in rng.choice(keys[own], N_PER_SETTING, replace=False)]
    picked += [(k, "boundary") for k in rng.choice(keys[~own], N_PER_SETTING, replace=False)]
    items = []
    for k, setting in picked:
        row, r = ev.loc[k], got[k]
        reply, err = parse_reply(r["output"], CATEGORIES[int(row.pred)], CATEGORIES, feats)
        assert reply is not None, (k, err)
        evidence = r["prompt"].split("Evidence:\n", 1)[1].split("\n\nAnswer", 1)[0]
        for arm, text, label in (("shap", shap_text(row), CATEGORIES[int(row.pred)]),
                                 ("shap_llm", llm_text(reply), reply["final_label"])):
            items.append({"row": int(k), "setting": setting, "arm": arm, "true": CATEGORIES[int(row.y)],
                          "label_correct": label == CATEGORIES[int(row.y)], "action_lexicon": bool(ACTION.search(text)),
                          "evidence": evidence, "text": text})
    items = pd.DataFrame(items).sample(frac=1, random_state=SEED).reset_index(drop=True)
    items.insert(0, "item", np.arange(1, len(items) + 1))
    OUT.mkdir(parents=True, exist_ok=True)
    items[["item", "evidence", "text"]].assign(action="", consistent="").to_csv(OUT / "sheet.csv", index=False)
    items.drop(columns=["evidence", "text"]).to_csv(OUT / "key.csv", index=False)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant pairs (b: only arm 1 yes, c: only arm 2 yes)."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(df: pd.DataFrame, col: str) -> dict:
    """Rates per arm and the shap_llm - shap difference over rows, with exact McNemar on the pairs."""
    w = df.pivot(index="row", columns="arm", values=col).dropna().astype(bool)
    b, c = int((w.shap_llm & ~w.shap).sum()), int((w.shap & ~w.shap_llm).sum())
    return {"item": col, "pairs": len(w), "shap": w.shap.mean(), "shap_llm": w.shap_llm.mean(),
            "diff": w.shap_llm.mean() - w.shap.mean(), "llm_only_yes": b, "shap_only_yes": c, "mcnemar_p": mcnemar_exact(b, c)}


def full_log_lexicon() -> tuple[int, int]:
    """(replies whose text matches the action list, valid replies) over all of M11 seed 42's log, not just the sample."""
    hit = n = 0
    with open(m11_dir(42) / "llm_log.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["valid"]:
                n += 1
                hit += bool(ACTION.search(r["output"]))
    return hit, n


def score() -> pd.DataFrame:
    key, sheet = pd.read_csv(OUT / "key.csv"), pd.read_csv(OUT / "sheet.csv", dtype=str, keep_default_na=False)
    df = key.merge(sheet[["item", "action", "consistent"]].assign(item=lambda s: s.item.astype(int)), on="item")
    yn = {"y": True, "yes": True, "1": True, "n": False, "no": False, "0": False}
    rows = [paired(df, "label_correct"), paired(df, "action_lexicon")]
    for col in ("action", "consistent"):
        v = df[col].str.strip().str.lower()
        bad = v[(v != "") & ~v.isin(yn)]
        assert bad.empty, f"{col}: use y or n, got {sorted(set(bad))[:5]}"
        if (v != "").all():
            rows.append(paired(df.assign(**{col: v.map(yn)}), col))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "h4c.csv", index=False)
    scored = [c for c in ("action", "consistent") if c in set(res["item"])]
    lines = ["# H4 (c): SHAP-only against SHAP + LLM explanations (M11 seed 42, 100 rows)", "",
             "Not preregistered: rubric fixed on 2026-11-17, after M11's reasons were read. The prompt never asked "
             "Gemma for operator actions. See the docstring of `scripts/m14_h4c.py` for the rubric.", "",
             "| item | pairs | SHAP only | SHAP + LLM | difference | exact McNemar p |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['item']} | {r.pairs} | {r.shap:.2f} | {r.shap_llm:.2f} | {r['diff']:+.2f} | {r.mcnemar_p:.3g} |"
              for _, r in res.iterrows()]
    hit, n = full_log_lexicon()
    lines += ["", f"Action list over every valid reply of seed 42 (not just the sample): {hit} of {n} ({hit / n:.4f}).",
              "", f"Scorer items done: {', '.join(scored) or 'none yet (fill sheet.csv with y or n and rerun)'}. "
              "`action_lexicon` is a keyword proxy, not the rubric."]
    (OUT / "H4C.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res


def main() -> None:
    if not (OUT / "sheet.csv").exists():
        make()
    print(score().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
