"""Allocation primitives: window chunks, train/test split, Dirichlet client shares, JSD."""
from __future__ import annotations

import numpy as np


def proportional_counts(n: int, p: np.ndarray) -> np.ndarray:
    """Integer counts summing to n, proportional to p (largest-remainder rounding)."""
    p = np.asarray(p, float)
    p = p / p.sum()
    raw = p * n
    counts = np.floor(raw).astype(np.int64)
    rem = n - counts.sum()
    if rem:
        counts[np.argsort(-(raw - counts), kind="stable")[:rem]] += 1
    return counts


def split_by_counts(idx: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
    return np.split(idx, np.cumsum(counts)[:-1])


def dirichlet_shares(gen: np.random.Generator, n_clients: int, alpha: float) -> np.ndarray:
    return gen.dirichlet(np.full(n_clients, alpha))


def drift_shares(gen: np.random.Generator, base: np.ndarray, drifted: np.ndarray, alpha: float) -> np.ndarray:
    """Re-draw how the drifted clients share their combined mass; others keep theirs exactly."""
    out = base.copy()
    out[drifted] = gen.dirichlet(np.full(len(drifted), alpha)) * base[drifted].sum()
    return out / out.sum()


def train_test_mask(gen: np.random.Generator, n: int, test_frac: float) -> np.ndarray:
    """Boolean test mask with exactly round(n*test_frac) True (at least 1 if n >= 2)."""
    k = int(round(n * test_frac))
    if n >= 2:
        k = min(max(k, 1), n - 1)
    m = np.zeros(n, bool)
    m[gen.choice(n, k, replace=False)] = True
    return m


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1])."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
