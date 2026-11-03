"""Layered YAML config: default -> env profile -> experiment -> CLI overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# src/ids/utils/config.py -> repo root is three levels above the package dir.
# Override with IDS_REPO_ROOT when the package is not run from a checkout.
REPO_ROOT = Path(os.environ.get("IDS_REPO_ROOT", Path(__file__).resolve().parents[3]))
CONFIG_DIR = REPO_ROOT / "configs"


def detect_env() -> str:
    """'kaggle' inside a Kaggle kernel, else 'local'. IDS_ENV forces a value."""
    if "IDS_ENV" in os.environ:
        return os.environ["IDS_ENV"]
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").is_dir():
        return "kaggle"
    return "local"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_overrides(items: list[str] | None) -> dict:
    """['gate.tau_conf=0.7', 'seed=1'] -> nested dict (values parsed as YAML)."""
    out: dict = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {item!r}")
        node = out
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def load_config(env: str | None = None, experiment: str | Path | None = None,
                overrides: dict | list[str] | None = None) -> dict[str, Any]:
    env = env or detect_env()
    cfg = _read_yaml(CONFIG_DIR / "default.yaml")
    env_file = CONFIG_DIR / "env" / f"{env}.yaml"
    if env_file.exists():
        cfg = deep_merge(cfg, _read_yaml(env_file))
    if experiment:
        exp = Path(experiment)
        if not exp.exists():
            exp = CONFIG_DIR / "experiments" / f"{experiment}.yaml"
        cfg = deep_merge(cfg, _read_yaml(exp))
    if isinstance(overrides, list):
        overrides = parse_overrides(overrides)
    cfg = deep_merge(cfg, overrides or {})
    cfg["env"] = env
    return cfg


def _abs(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else base / p


def data_path(cfg: dict, key: str) -> Path:
    """Resolve paths.<key> against paths.data_root.

    On Kaggle, inputs live under /kaggle/input/<dataset-slug>/..., so if the
    direct path is missing we search data_root for a file/dir with that name.
    """
    root = _abs(cfg["paths"]["data_root"], REPO_ROOT)
    p = _abs(cfg["paths"][key], root)
    if p.exists() or cfg.get("env") != "kaggle":
        return p
    hits = sorted(root.rglob(Path(cfg["paths"][key]).name))
    return hits[0] if hits else p


def results_root(cfg: dict) -> Path:
    return _abs(cfg["paths"]["results_root"], REPO_ROOT)
