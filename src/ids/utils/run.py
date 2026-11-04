"""Run directories: every run snapshots its config + environment for reproducibility."""
from __future__ import annotations

import importlib.metadata as md
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import REPO_ROOT, results_root
from .jsonl import JsonlWriter, dumps

TRACKED_PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "xgboost",
                    "lightgbm", "shap", "torch", "transformers", "bitsandbytes", "pyyaml"]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def environment_info(env: str) -> dict:
    pkgs = {}
    for p in TRACKED_PACKAGES:
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "env": env,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": pkgs,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


class Run:
    """Context for one experiment run under results_root/<timestamp>_<name>/."""

    def __init__(self, cfg: dict, name: str):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = results_root(cfg) / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.cfg = cfg
        self.events = JsonlWriter(self.dir / "events.jsonl")
        self.save_json("experiment_config.json", cfg)
        self.save_json("environment.json", environment_info(cfg.get("env", "?")))
        self.logger = logging.getLogger(f"ids.run.{name}")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(fh)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, name: str, obj) -> Path:
        p = self.path(name)
        p.write_text(dumps(obj, indent=2), encoding="utf-8")
        return p

    def log(self, event: str, **fields) -> None:
        self.events.write({"event": event, **fields})

    def close(self) -> None:
        for h in list(self.logger.handlers):
            h.close()
            self.logger.removeHandler(h)

    def __enter__(self):
        self.log("run_start")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.log("run_end", ok=exc_type is None, error=repr(exc) if exc else None)
        self.close()
        return False


def latest_run(cfg: dict, pattern: str) -> Path:
    """Newest run folder matching a glob; timestamps in the names sort chronologically."""
    runs = sorted(results_root(cfg).glob(pattern))
    if not runs:
        sys.exit(f"no run matching {pattern} under {results_root(cfg)}; run that milestone's script first")
    return runs[-1]


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
