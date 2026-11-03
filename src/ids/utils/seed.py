"""Deterministic seeding: one global seed, stable derived seeds per client/window."""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects child processes only
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base: int, *keys: object) -> int:
    """Stable across runs/platforms (unlike hash()): derive_seed(42, 'client', 3, 't2')."""
    s = ":".join([str(base), *map(str, keys)])
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")


def rng(base: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base, *keys))
