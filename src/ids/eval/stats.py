"""M14 statistics: macro-F1 from confusion counts, a paired client bootstrap, Holm's correction.

The client is the resampling unit: clients hold separate test rows, while windows and rows within a client
are not independent. Seeds are averaged inside each draw, with the same clients drawn for every seed.
"""
from __future__ import annotations

import numpy as np


def macro_f1_cm(cm: np.ndarray) -> np.ndarray:
    """Macro-F1 over the classes present in the rows (the harness's rule), for (..., K, K) counts, true x pred."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    sup, pred = cm.sum(-1), cm.sum(-2)
    f1 = np.where(sup + pred > 0, 2 * tp / np.maximum(sup + pred, 1), 0.0)
    keep = sup > 0
    return (f1 * keep).sum(-1) / np.maximum(keep.sum(-1), 1)


def false_positive_rate_cm(cm: np.ndarray) -> np.ndarray:
    """Share of benign rows (class 0) given any attack label, for (..., K, K) counts; 0 when no benign rows."""
    benign = cm[..., 0, :].sum(-1)
    return np.where(benign > 0, (benign - cm[..., 0, 0]) / np.maximum(benign, 1), 0.0)


def pair_scores(counts: np.ndarray, weights: np.ndarray, metric=macro_f1_cm) -> np.ndarray:
    """counts (S seeds, P (train, eval) pairs, C clients, K, K); weights (B, C), how often each client is drawn.
    Returns (B, S, P): the metric on the drawn clients' pooled rows of each seed and pair."""
    return metric(np.einsum("bc,spckl->bspkl", weights, counts, optimize=True))


def series_score(counts: np.ndarray, weights: np.ndarray, metric=macro_f1_cm) -> np.ndarray:
    """(B,): the metric averaged over the pairs, then over the seeds."""
    return pair_scores(counts, weights, metric).mean(-1).mean(-1)


def client_draws(n_clients: int, n: int, seed: int) -> np.ndarray:
    """(n, C) draw counts; the first row is all ones, the observed score."""
    r = np.random.default_rng(seed)
    w = np.stack([np.bincount(r.integers(0, n_clients, n_clients), minlength=n_clients) for _ in range(n - 1)])
    return np.vstack([np.ones(n_clients, int), w]).astype(float)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0, stat=series_score) -> dict:
    """a, b: counts as in series_score, same clients and seeds. Difference stat(a) - stat(b) with a 95% percentile
    interval and a two-sided bootstrap p-value; stat maps (counts, weights) to (B,)."""
    assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[2], (a.shape, b.shape)
    w = client_draws(a.shape[2], n, seed)
    d = stat(a, w) - stat(b, w)
    rest = d[1:]
    p = min(1.0, 2 * min(float((rest <= 0).mean()), float((rest >= 0).mean())))
    return {"diff": float(d[0]), "lo": float(np.quantile(rest, 0.025)), "hi": float(np.quantile(rest, 0.975)), "p": p, "draws": n - 1}


def holm(p: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    order = np.argsort(p)
    m, out, run = len(p), np.empty(len(p)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        out[j] = run
    return out.tolist()
