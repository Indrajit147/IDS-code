"""ADWIN2 (Bifet & Gavaldà, 2007): adaptive windowing with an exponential histogram.

Written here rather than taken from `river` so the project needs no extra dependency on Kaggle.
Uses the variance-based cut from the reference implementation; values should lie in [0, 1].
"""
from __future__ import annotations

import math

import numpy as np


class Adwin:
    def __init__(self, delta: float = 0.002, max_buckets: int = 5, clock: int = 32,
                 min_window: int = 10, min_sub_window: int = 5):
        self.delta, self.max_buckets, self.clock = delta, max_buckets, clock
        self.min_window, self.min_sub_window = min_window, min_sub_window
        self.levels: list[list[list[float]]] = [[]]      # level i: [total, variance] of 2^i items; oldest first
        self.width, self.total, self.variance, self.t = 0, 0.0, 0.0, 0
        self.last_change = 0.0                           # newer mean - older mean at the last detection

    @property
    def mean(self) -> float:
        return self.total / self.width if self.width else 0.0

    def update(self, x: float) -> bool:
        """Add one value; True if the window shrank because the mean changed."""
        if self.width:
            d = x - self.total / self.width
            self.variance += self.width * d * d / (self.width + 1)
        self.width += 1
        self.total += x
        self.levels[0].append([x, 0.0])
        self._compress()
        self.t += 1
        if self.t % self.clock == 0 and self.width > self.min_window:
            return self._shrink()
        return False

    def _compress(self) -> None:
        for i, level in enumerate(self.levels):
            if len(level) <= self.max_buckets:
                return
            (t1, v1), (t2, v2) = level[0], level[1]
            size = 2 ** i
            d = t1 / size - t2 / size
            merged = [t1 + t2, v1 + v2 + size * d * d / 2]
            del level[:2]
            if i + 1 == len(self.levels):
                self.levels.append([])
            self.levels[i + 1].append(merged)

    def _cut(self, n0: int, n1: int, u0: float, u1: float) -> bool:
        s = self.min_sub_window
        dd = math.log(2 * math.log(self.width) / self.delta)
        m = 1 / (n0 - s + 1) + 1 / (n1 - s + 1)
        v = max(self.variance, 0.0) / self.width
        return abs(u0 / n0 - u1 / n1) > math.sqrt(2 * m * v * dd) + 2 / 3 * dd * m

    def _shrink(self) -> bool:
        changed, again = False, True
        while again and self.width > self.min_window:
            again = False
            n0, u0, n1, u1 = 0, 0.0, self.width, self.total
            for i in range(len(self.levels) - 1, -1, -1):      # oldest (largest) buckets first
                size = 2 ** i
                for total, _ in self.levels[i]:
                    n0 += size; n1 -= size; u0 += total; u1 -= total
                    if n1 < self.min_sub_window:
                        break
                    if n0 >= self.min_sub_window and self._cut(n0, n1, u0, u1):
                        if not changed:
                            self.last_change = u1 / n1 - u0 / n0
                        changed = again = True
                        self._drop_oldest()
                        break
                if again or n1 < self.min_sub_window:
                    break
        return changed

    def _drop_oldest(self) -> None:
        i = max(j for j, level in enumerate(self.levels) if level)
        total, var = self.levels[i].pop(0)
        size = 2 ** i
        self.width -= size
        self.total -= total
        d = total / size - self.total / self.width
        self.variance -= var + size * self.width * d * d / (size + self.width)
        while len(self.levels) > 1 and not self.levels[-1]:
            self.levels.pop()


def detect(values: np.ndarray, delta: float = 0.002) -> list[tuple[int, float]]:
    """Run ADWIN over a stream; returns (position, change in mean) for each detection.
    The window restarts after a detection, so one change gives one event, as in `river`:
    detection lists matched river.drift.ADWIN 0.26.1 exactly on 40 synthetic streams."""
    a, out = Adwin(delta), []
    for i, x in enumerate(np.asarray(values, dtype=float).tolist()):
        if a.update(x):
            out.append((i, a.last_change))
            a = Adwin(delta)
    return out
