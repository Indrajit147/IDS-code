"""M8: choose FedAvg's learning rate on one pool, from validation rows only, and write reports/m8/<pool>/LR_SWEEP.csv.

The rule, fixed before any result of this sweep: the highest mean best-validation macro-F1 over windows, averaged
over the seeds, among the rates where every seed finished. Ties go to the lower rate. Each pass runs in its own process.

Usage: python scripts/m8_lr_sweep.py --pool full --env kaggle --lrs 0.003 0.01 0.03 0.1 --seeds 42 43
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from ids.utils.config import REPO_ROOT


def pick_lr(table: pd.DataFrame, n_seeds: int) -> float:
    """table has one row per finished pass, with columns lr and mean_val."""
    g = table.groupby("lr").mean_val.agg(["mean", "count"])
    g = g[g["count"] == n_seeds].sort_index()
    if g.empty:
        raise SystemExit("no learning rate finished on every seed")
    return float(g["mean"].idxmax())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="full")
    ap.add_argument("--env", default=None)
    ap.add_argument("--lrs", nargs="+", type=float, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()
    rep = REPO_ROOT / "reports" / "m8" / args.pool
    env = ["--env", args.env] if args.env else []
    rows = []
    for lr in args.lrs:
        for seed in args.seeds:
            tag = f"sweep_lr{lr:g}_s{seed}"
            code = subprocess.run([sys.executable, "-X", "faulthandler", "scripts/m8_fedavg.py", *env, "--pool", args.pool, "--tag", tag, "--only", "fedavg",
                                   "--set", "resources.n_jobs=4", f"seed={seed}", f"fedavg.lr={lr:g}"], cwd=REPO_ROOT).returncode
            print(f"lr {lr:g}, seed {seed}: exit code {code}", flush=True)
            if code:
                continue
            log = pickle.loads((rep / tag / "checkpoints" / "fedavg.pkl").read_bytes())["log"]
            shutil.rmtree(rep / tag)
            val = [r["best_val"] for r in log]
            rows.append({"lr": lr, "seed": seed, "mean_val": float(np.mean(val)), "val_by_window": " ".join(f"{v:.4f}" for v in val),
                         "rounds_by_window": " ".join(str(r["rounds_run"]) for r in log)})
    table = pd.DataFrame(rows)
    best = pick_lr(table, len(args.seeds))
    table["chosen"] = table.lr == best
    table.to_csv(rep / "LR_SWEEP.csv", index=False)
    print(table.to_string(), f"\nchosen lr: {best:g}")


if __name__ == "__main__":
    main()
