"""M0 smoke run: config + seeding + run directory + data availability check.

Usage:  python scripts/m0_smoke_run.py [--env local|kaggle] [--set key=value ...]
"""
from __future__ import annotations

import argparse

from ids.utils import Run, data_path, load_config, set_global_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(env=args.env, overrides=args.set)
    set_global_seed(cfg["seed"])

    with Run(cfg, "m0_smoke") as run:
        checks = {}
        for key in ("raw_csv", "zenodo_dir"):
            p = data_path(cfg, key)
            checks[key] = {"path": str(p), "exists": p.exists(),
                           "bytes": p.stat().st_size if p.is_file() else None}
        run.save_json("data_check.json", checks)
        run.log("data_check", **checks)
        run.logger.info("env=%s run_dir=%s", cfg["env"], run.dir)
        for k, v in checks.items():
            run.logger.info("%-10s exists=%s  %s", k, v["exists"], v["path"])


if __name__ == "__main__":
    main()
