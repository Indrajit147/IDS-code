"""Drift-robustness metrics over a train x eval matrix.

M is a DataFrame of scores: index = train_through (model trained through window t),
columns = eval_window (test set of window w). A static model has a single row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _latest(M: pd.DataFrame, t: int) -> int | None:
    """The newest model available at window t (a static model stays on its only row)."""
    rows = [r for r in M.index if r <= t]
    return max(rows) if rows else None


def current_series(M: pd.DataFrame) -> pd.Series:
    """Score at each window of the newest model available at that window."""
    out = {w: M.loc[_latest(M, w), w] for w in M.columns if _latest(M, w) is not None}
    return pd.Series(out, dtype=float)


def window_average(series: pd.Series, skip=()) -> float:
    return float(series.drop(index=list(skip), errors="ignore").mean())


def forgetting_by_window(M: pd.DataFrame) -> dict[int, float]:
    """F_max - F_final per window's test set, over models trained at or after that window.
    Windows seen by only one model have no history and are left out."""
    last, out = M.index.max(), {}
    for w in M.columns:
        hist = M.loc[[t for t in M.index if t >= w], w].dropna()
        if len(hist) >= 2:
            out[int(w)] = float(hist.iloc[:-1].max() - M.loc[last, w])
    return out


def forgetting_by_class(per_class: dict[tuple[int, int], dict[str, float]]) -> dict[str, float]:
    """F_max - F_final per class, measured on the test set of the window that introduced the
    class, so the task stays fixed as the models change. per_class maps (t, w) to {class: F1}."""
    if not per_class:
        return {}
    last = max(t for t, _ in per_class)
    intro: dict[str, int] = {}
    for (_, w), scores in per_class.items():
        for c in scores:
            intro[c] = min(intro.get(c, w), w)
    out = {}
    for c, i in intro.items():
        hist = {t: s[c] for (t, w), s in per_class.items() if w == i and t >= i and c in s}
        if len(hist) >= 2 and last in hist:
            out[c] = float(max(v for t, v in hist.items() if t != last) - hist[last])
    return out


def time_to_recover(M: pd.DataFrame, eps: float) -> dict[int, dict]:
    """After each window d: pre = score just before d; cold = the pre-drift model on window d.
    recovered_after = 0 if cold stayed within eps of pre; otherwise the number of windows until a
    model updated at or after d is back within eps of pre; None if that never happens (censored)."""
    windows, out = sorted(M.columns), {}
    for d in windows[1:]:
        t_prev = _latest(M, d - 1)
        if t_prev is None:
            continue
        pre, cold = float(M.loc[t_prev, d - 1]), float(M.loc[t_prev, d])
        recovered = 0 if cold >= pre - eps else None
        if recovered is None:
            for k, w in enumerate(x for x in windows if x >= d):
                if float(M.loc[_latest(M, w), w]) >= pre - eps and _latest(M, w) >= d:
                    recovered = k + 1
                    break
        out[int(d)] = {"pre": pre, "cold": cold, "recovered_after": recovered}
    return out


def summarize_recovery(ttr: dict[int, dict]) -> dict:
    drops = [v for v in ttr.values() if v["recovered_after"] != 0]
    done = [v["recovered_after"] for v in drops if v["recovered_after"] is not None]
    return {"n_drops": len(drops), "n_recovered": len(done), "n_never": len(drops) - len(done),
            "mean_updates_to_recover": float(np.mean(done)) if done else None}
