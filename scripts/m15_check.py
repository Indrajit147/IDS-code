"""M15 acceptance check: does a fresh run reproduce the committed seed-42 numbers within ±0.005 macro-F1?

  --part A  after notebooks/m15_notebook_a.ipynb: each headline cell (CSV rows; window average and every window) of the
            reports just written, against the same file at --ref (default HEAD, the commit the clone was made from).
  --part B  after notebooks/m15_notebook_b.ipynb: M10's and M11's refits of e4_added against M9 (their fit_check.json).

Writes reports/m15/<pool>/CHECK_<part>.md and check_<part>.csv; exits 1 if any cell is outside the tolerance or missing.
Run: python scripts/m15_check.py --part A [--ref HEAD] [--pool full]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pandas as pd

from ids.utils.config import REPO_ROOT

TOL = 0.005
SLICE = "csv_only"
CELLS_A = [("m4", "E1"), ("m4", "static"), ("m6", "naive_incremental"), ("m6", "local_rehearsal"),
           ("m7", "local_rehearsal"), ("m7", "e3"), ("m9", "e4_added"), ("m13", "e7")]


def cells(summary: dict, method: str) -> dict[str, float]:
    """{'window_avg': x, 't0': ..., 't5': ...} for one method's CSV-row summary, empty if the method isn't there."""
    s = [x for x in summary["summaries"] if x["method"] == method and x["slice"] == SLICE]
    if not s:
        return {}
    return {"window_avg": s[0]["window_avg_macro_f1"], **{f"t{int(w)}": v for w, v in s[0]["macro_f1_by_window"].items()}}


def compare(new: dict[str, float], ref: dict[str, float]) -> list[dict]:
    """One row per reference cell; a cell missing from the new run counts as a failure."""
    return [{"cell": k, "ref": r, "new": new.get(k), "diff": None if k not in new else new[k] - r,
             "ok": k in new and abs(new[k] - r) <= TOL} for k, r in ref.items()]


def committed(path: str, ref: str) -> dict:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{ref}:{path}"], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"{path} is not in {ref}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def part_a(pool: str, ref: str) -> pd.DataFrame:
    rows = []
    for m, method in CELLS_A:
        path = f"reports/{m}/{pool}/summary.json"
        f = REPO_ROOT / path
        new = cells(json.loads(f.read_text(encoding="utf-8")), method) if f.exists() else {}
        old = cells(committed(path, ref), method)
        assert old, (path, method, "not in the committed summary")
        rows += [{"source": m, "method": method, **r} for r in compare(new, old)]
    return pd.DataFrame(rows)


def part_b(pool: str) -> pd.DataFrame:
    rows = []
    for m in ("m10", "m11"):
        f = REPO_ROOT / "reports" / m / pool / "fit_check.json"
        if not f.exists():
            rows.append({"source": m, "method": "e4_added", "cell": "fit_check.json", "ref": None, "new": None, "diff": None, "ok": False})
            continue
        for t, v in json.loads(f.read_text(encoding="utf-8"))["by_window"].items():
            new = next(x for k, x in v.items() if k not in ("m9", "difference"))   # M11 refits through M10's fit, so its key is "m10" too
            rows.append({"source": m, "method": "e4_added", "cell": f"t{t}", "ref": v["m9"], "new": new,
                         "diff": v["difference"], "ok": abs(v["difference"]) <= TOL})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B"], required=True)
    ap.add_argument("--pool", default="full")
    ap.add_argument("--ref", default="HEAD")
    a = ap.parse_args()
    df = part_a(a.pool, a.ref) if a.part == "A" else part_b(a.pool)
    out = REPO_ROOT / "reports" / "m15" / a.pool
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"check_{a.part.lower()}.csv", index=False)
    worst = df["diff"].abs().max()
    verdict = "PASS" if df.ok.all() else "FAIL"
    against = f"the files at `{a.ref}`" if a.part == "A" else "M9's committed scores"
    lines = [f"# M15 check, part {a.part} ({a.pool} pool, seed 42): {verdict}", "",
             f"{len(df)} cells against {against}; tolerance ±{TOL}; largest difference {worst:.6f}; "
             f"{int((~df.ok).sum())} outside the tolerance or missing.", "",
             "| source | method | cell | committed | new | difference |", "|---|---|---|---:|---:|---:|"]
    fmt = lambda x: "missing" if x is None or pd.isna(x) else f"{x:.6f}"
    lines += [f"| {r.source} | {r.method} | {r.cell} | {fmt(r.ref)} | {fmt(r.new)} | {fmt(r['diff'])}{'' if r.ok else ' ✗'} |"
              for _, r in df.iterrows()]
    (out / f"CHECK_{a.part}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{verdict}: largest difference {worst:.6f} over {len(df)} cells -> {out / f'CHECK_{a.part}.md'}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
