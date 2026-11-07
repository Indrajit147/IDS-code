"""The edge gate: escalate a flow iff confidence < tau_c, or disagreement > tau_d, or drift is flagged."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .adwin import detect


def gate_signals(components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-flow label-free signals from EdgeEnsemble.predict_components, all in [0, 1]."""
    p = components["calibrated"]
    logs = np.log(p, where=p > 0, out=np.zeros_like(p))
    return {"confidence": p.max(1),
            "entropy": -(p * logs).sum(1) / np.log(p.shape[1]),
            "disagreement": 0.5 * np.abs(components["xgb"] - components["lgb"]).sum(1)}


def escalate(conf: np.ndarray, disagree: np.ndarray, tau_c: float, tau_d: float,
             drift: np.ndarray | None = None) -> np.ndarray:
    mask = (conf < tau_c) | (disagree > tau_d)
    return mask if drift is None else mask | drift


def drift_state(n: int, fire_positions: list[int], hold: int) -> np.ndarray:
    """Flows flagged because a detector fired within the previous `hold` flows of the stream."""
    state = np.zeros(n, bool)
    for pos in fire_positions:
        state[pos + 1:pos + 1 + hold] = True
    return state


def drift_score(streams: list[np.ndarray], n_pre: int, delta: float) -> tuple[np.ndarray, list[int]]:
    """Per post-boundary row: the largest rise ADWIN has reported so far on any label-free stream, in [0, 1], 0 until then.

    A row's own value can't move its score. Returns the score and the post-boundary positions of the first stream's detections."""
    score, fires = np.zeros(len(streams[0]) - n_pre), []
    for i, values in enumerate(streams):
        for pos, change in detect(values, delta):
            if pos < n_pre:
                continue
            if i == 0:
                fires.append(pos - n_pre)
            if change > 0:
                score[pos - n_pre + 1:] = np.maximum(score[pos - n_pre + 1:], min(change, 1.0))
    return score, fires


def adaptive_taus(tau_c: float, tau_d: float, score: np.ndarray, strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Thresholds per flow. A score or strength of 0 gives exactly (tau_c, tau_d); a higher one escalates more."""
    a = np.minimum(strength * np.asarray(score, dtype=float), 1.0)
    return tau_c + a * (1 - tau_c), tau_d * (1 - a)


def same_cost_mask(conf: np.ndarray, other: np.ndarray, rate: float) -> np.ndarray:
    """The fixed gate at a given escalation rate: rows already flagged by other, then the lowest confidence until the rate is met."""
    mask = other.copy()
    extra = int(round(rate * len(conf))) - int(mask.sum())
    if extra > 0:
        free = np.flatnonzero(~mask)
        mask[free[np.argsort(conf[free], kind="stable")[:extra]]] = True
    return mask


def gate_outcome(mask: np.ndarray, errors: np.ndarray) -> dict:
    """Cost (escalation rate) and benefit (share of edge errors sent to the cloud)."""
    n_err, n_esc = int(errors.sum()), int(mask.sum())
    return {"escalation_rate": float(mask.mean()) if len(mask) else 0.0,
            "error_capture": float(errors[mask].sum() / n_err) if n_err else None,
            "error_rate_in_escalated": float(errors[mask].mean()) if n_esc else None,
            "accuracy_not_escalated": float(1 - errors[~mask].mean()) if n_esc < len(mask) else None}


def frontier(conf: np.ndarray, disagree: np.ndarray, errors: np.ndarray, taus_c, taus_d) -> pd.DataFrame:
    rows = [{"tau_c": c, "tau_d": d, **gate_outcome(escalate(conf, disagree, c, d), errors)}
            for c in taus_c for d in taus_d]
    return pd.DataFrame(rows)


def tune(front: pd.DataFrame, max_escalation_rate: float) -> dict:
    """Most errors captured within the escalation budget; cheapest setting breaks ties."""
    ok = front[front["escalation_rate"] <= max_escalation_rate]
    if ok.empty:
        raise ValueError(f"no thresholds stay within an escalation rate of {max_escalation_rate}")
    best = ok.sort_values(["error_capture", "escalation_rate", "tau_c", "tau_d"],
                          ascending=[False, True, True, False]).iloc[0]
    return {k: float(best[k]) for k in ("tau_c", "tau_d", "escalation_rate", "error_capture")}
