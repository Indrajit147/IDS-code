"""DA-FedDistill, the cloud side: soft predictions on D_pub -> pseudo-labels -> a global calibrator g_t.

g_t maps (p_xgb, p_lgb, drift features) to p_global. It is a softmax regression on log probabilities,
which starts as the geometric mean of the two boosters and stays close to that under its L2 penalty.
Nothing here takes a true label: the rows' classes are hidden from distillation (D2), so class balance
comes from the pseudo-labels too.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from ..models.edge import balanced_weights, fit_temperature

LOG_FLOOR = 1e-4       # the lowest probability a calibrator input can express


def aggregate_soft(preds: np.ndarray, reliability: np.ndarray | None = None) -> np.ndarray:
    """preds (clients, rows, classes) -> pseudo-labels (rows, classes). reliability (clients, classes) weights
    each client per class, so a client that can't judge a class has no say on that column."""
    if reliability is None:
        q = preds.mean(0)
    else:
        total = reliability.sum(0)
        q = np.einsum("irk,ik->rk", preds, reliability) / np.where(total > 0, total, 1.0)
    return q / q.sum(1, keepdims=True)


def reliability_from_val(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class F1 on a client's own validation rows; 0 for classes it has no rows of. The client reports this."""
    out = np.zeros(n_classes)
    for c in range(n_classes):
        tp, fp, fn = np.sum((pred == c) & (y == c)), np.sum((pred == c) & (y != c)), np.sum((pred != c) & (y == c))
        out[c] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return out


def pseudo_label_weights(q: np.ndarray, cap: float | None = None) -> np.ndarray:
    """Balanced row weights by pseudo-label, because DDoS is 65% of D_pub on the full pool (D2)."""
    return balanced_weights(q.argmax(1), cap)


def quantize(p: np.ndarray) -> np.ndarray:
    return np.round(p * 255).astype(np.uint8)


def dequantize(b: np.ndarray) -> np.ndarray:
    d = b.astype(np.float32)
    return d / np.maximum(d.sum(-1, keepdims=True), 1)


def uplink_bytes(arrays: list[np.ndarray], metadata_bytes: int = 0) -> dict:
    """One client's upload for one round: float32, 8-bit, and 8-bit after zlib, measured on the actual values."""
    n = sum(a.size for a in arrays)
    z = sum(len(zlib.compress(quantize(a).tobytes(), 6)) for a in arrays)
    return {"raw_bytes": n * 4 + metadata_bytes, "q8_bytes": n + metadata_bytes, "q8_zlib_bytes": z + metadata_bytes}


def drift_signals(px: np.ndarray, pl: np.ndarray, client_shift: np.ndarray) -> np.ndarray:
    """Label-free drift features per row: booster disagreement, normalized entropy of their mean, the client's PSI share."""
    m = np.clip((px + pl) / 2, 1e-12, 1)
    return np.column_stack([0.5 * np.abs(px - pl).sum(1), -(m * np.log(m)).sum(1) / np.log(px.shape[1]), client_shift])


def apply_temperatures(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """softmax(log p / T) with one temperature per row. It never changes a row's predicted class."""
    return _softmax(np.log(np.clip(p, 1e-12, 1)) / T[:, None])


def fit_client_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """A client's own temperature on its validation rows. g_t's target is a mean over clients, which is soft,
    so g_t comes out underconfident; the clients have the labels to correct that, and the cloud does not."""
    return fit_temperature(p, y)


def accept_calibrator(y: np.ndarray, local: np.ndarray, calibrated: np.ndarray) -> bool:
    """The guard: keep g_t only if it beats the client's own model on the client's validation rows."""
    labels = np.unique(y)
    f1 = lambda p: f1_score(y, p.argmax(1), labels=labels, average="macro", zero_division=0)
    return bool(f1(calibrated) > f1(local))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class GlobalCalibrator:
    def __init__(self, n_classes: int, n_extra: int, l2: float = 1e-4, max_iter: int = 200):
        self.k, self.e, self.l2, self.max_iter = n_classes, n_extra, l2, max_iter
        self.W = self._start()

    def _start(self) -> np.ndarray:
        k = self.k
        W = np.zeros((k, 2 * k + self.e + 1))
        W[np.arange(k), np.arange(k)] = W[np.arange(k), k + np.arange(k)] = 0.5
        return W

    def _inputs(self, px, pl, extra) -> np.ndarray:
        parts = [np.log(np.clip(px, LOG_FLOOR, 1)), np.log(np.clip(pl, LOG_FLOOR, 1))]
        if self.e:
            parts.append(extra)
        parts.append(np.ones((len(px), 1)))
        return np.concatenate(parts, axis=1)

    def fit(self, px, pl, extra, q, weight=None) -> "GlobalCalibrator":
        """Minimize the weighted cross-entropy to the soft targets q, with L2 toward the starting point."""
        F = self._inputs(px, pl, extra)
        w = np.ones(len(F)) if weight is None else np.asarray(weight, float)
        w = w / w.sum()
        start = self._start()

        def loss(v):
            W = v.reshape(start.shape)
            z = F @ W.T
            z = z - z.max(1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(1, keepdims=True))
            d = W - start
            grad = (F * w[:, None]).T @ (np.exp(logp) - q)
            return -np.sum(w * np.sum(q * logp, axis=1)) + self.l2 * np.sum(d * d), grad.T.ravel() + 2 * self.l2 * d.ravel()

        res = minimize(loss, start.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.W = res.x.reshape(start.shape)
        return self

    def predict(self, px, pl, extra) -> np.ndarray:
        return _softmax(self._inputs(px, pl, extra) @ self.W.T)

    def to_bytes(self) -> bytes:
        return self.W.astype(np.float32).tobytes()

    @classmethod
    def from_bytes(cls, blob: bytes, n_classes: int, n_extra: int) -> "GlobalCalibrator":
        c = cls(n_classes, n_extra)
        c.W = np.frombuffer(blob, np.float32).reshape(c.W.shape).astype(float)
        return c
