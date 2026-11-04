"""Unsupervised distribution-shift metrics per feature: PSI, KL and Wasserstein.
The proposal uses these for reporting, not as hard triggers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def bin_edges(ref: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile edges of the reference. Many features are zero-inflated, so duplicates collapse."""
    return np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))


def bin_shares(x: np.ndarray, edges: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Shares over right-closed bins, with open bins at both ends for values outside the reference."""
    counts = np.bincount(np.searchsorted(edges, x, side="left"), minlength=len(edges) + 1).astype(float)
    p = counts / max(counts.sum(), 1) + eps
    return p / p.sum()


def psi(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((q - p) * np.log(q / p)))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(current || reference)."""
    return float(np.sum(q * np.log(q / p)))


def scaled_wasserstein(ref: np.ndarray, cur: np.ndarray) -> float:
    """Wasserstein distance in units of the reference's standard deviation, so features compare."""
    std = float(np.std(ref))
    return float(wasserstein_distance(ref, cur) / (std if std > 0 else 1.0))


def feature_shift(ref: np.ndarray, cur: np.ndarray, names: list[str], n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        edges = bin_edges(ref[:, j], n_bins)
        p, q = bin_shares(ref[:, j], edges), bin_shares(cur[:, j], edges)
        rows.append({"feature": name, "psi": psi(p, q), "kl": kl(p, q),
                     "wasserstein": scaled_wasserstein(ref[:, j], cur[:, j])})
    return pd.DataFrame(rows)
