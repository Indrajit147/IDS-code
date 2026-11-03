"""Synthetic drift timeline: window in which each attack19 class is introduced."""
from __future__ import annotations

from fnmatch import fnmatch

from ..data.schema import ATTACKS19


def intro_windows(definition: dict[str, list[str]], windows: list[str]) -> dict[str, int]:
    """{attack19 name: index of introduction window}; every class exactly once, Benign at t0."""
    unknown = set(definition) - set(windows)
    if unknown:
        raise ValueError(f"timeline uses undeclared windows: {sorted(unknown)}")
    intro: dict[str, int] = {}
    for w, patterns in definition.items():
        for pat in patterns or []:
            hits = [a for a in ATTACKS19 if fnmatch(a, pat)]
            if not hits:
                raise ValueError(f"pattern {pat!r} in {w} matches no class")
            for a in hits:
                if a in intro:
                    raise ValueError(f"class {a!r} introduced twice")
                intro[a] = windows.index(w)
    missing = [a for a in ATTACKS19 if a not in intro]
    if missing:
        raise ValueError(f"classes never introduced: {missing}")
    if intro["Benign"] != 0:
        raise ValueError("Benign must be present from the first window")
    return intro


def window_definition(cfg: dict, name: str) -> dict:
    windows = cfg["timeline"]["windows"]
    intro = intro_windows(cfg["timeline"]["definitions"][name], windows)
    return {
        "timeline": name,
        "windows": windows,
        "intro_window": intro,
        "classes_by_window": {w: [a for a in ATTACKS19 if intro[a] <= i] for i, w in enumerate(windows)},
        "new_classes_by_window": {w: [a for a in ATTACKS19 if intro[a] == i] for i, w in enumerate(windows)},
        "semantics": "class rows are split evenly over windows intro..last (cumulative stream)",
    }
