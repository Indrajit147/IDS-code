"""Bounded, class-balanced rehearsal buffer for one client."""
from __future__ import annotations

import numpy as np


class RehearsalBuffer:
    """Keeps at most `capacity` rows, split evenly across seen classes, with unused share going to larger classes.

    Each row gets a random key once, and a class keeps its smallest keys. That is a uniform sample of every
    row the class has offered, at any quota, so shrinking a class when a new one appears needs no special case.
    """

    def __init__(self, capacity: int, seed_rng: np.random.Generator):
        self.capacity, self.rng = capacity, seed_rng
        self.keys: dict[int, np.ndarray] = {}
        self.rows: dict[int, np.ndarray] = {}
        self.peak_bytes = 0

    def _quotas(self) -> dict[int, int]:
        """Water-filling: small classes keep everything, the rest share what is left, largest counts last."""
        sizes = sorted(self.rows, key=lambda c: (len(self.rows[c]), c))
        left, out = self.capacity, {}
        for i, c in enumerate(sizes):
            out[c] = min(len(self.rows[c]), left // (len(sizes) - i))
            left -= out[c]
        return out

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        for c in np.unique(y):
            c = int(c)
            m = y == c
            keys = self.rng.random(int(m.sum()))
            self.keys[c] = np.concatenate([self.keys[c], keys]) if c in self.keys else keys
            self.rows[c] = np.concatenate([self.rows[c], X[m]]) if c in self.rows else X[m].copy()
        for c, q in self._quotas().items():
            keep = np.argsort(self.keys[c])[:q]
            self.keys[c], self.rows[c] = self.keys[c][keep], self.rows[c][keep]
        self.peak_bytes = max(self.peak_bytes, self.nbytes)

    def get(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.rows:
            return None
        cs = sorted(self.rows)
        return np.concatenate([self.rows[c] for c in cs]), np.concatenate([np.full(len(self.rows[c]), c, np.int64) for c in cs])

    def __len__(self) -> int:
        return sum(len(r) for r in self.rows.values())

    @property
    def nbytes(self) -> int:
        return sum(self.rows[c].nbytes + self.keys[c].nbytes for c in self.rows)
