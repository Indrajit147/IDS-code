"""Edge detector: XGBoost + LightGBM, probability fusion and temperature calibration.

The ensemble always outputs probabilities over a FIXED global label space
(e.g. the 6 categories), even when a window has seen only some classes:
trees are trained on the locally present classes and their outputs are
scattered back into the global columns (absent classes get probability 0).

Fitting uses a held-out validation split for early stopping, the fusion
weight (chosen by validation log-loss) and the temperature (validation NLL).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.optimize import minimize_scalar

EPS = 1e-12


def balanced_weights(y: np.ndarray, cap: float | None = None) -> np.ndarray:
    """sklearn-style 'balanced' sample weights n / (k * n_c), optionally capped."""
    cls, cnt = np.unique(y, return_counts=True)
    w_c = len(y) / (len(cls) * cnt)
    if cap is not None:
        w_c = np.minimum(w_c, cap)
    lut = dict(zip(cls, w_c))
    return np.vectorize(lut.get)(y).astype(np.float32)


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    """softmax(log p / T), keeping structurally-absent classes (p == 0) at exactly 0."""
    with np.errstate(divide="ignore"):
        z = np.log(p) / T
    z -= np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(lambda T: nll(apply_temperature(p, T), y), bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


@dataclass
class EdgeConfig:
    n_classes: int
    xgb_params: dict = field(default_factory=dict)
    lgb_params: dict = field(default_factory=dict)
    early_stopping: int = 20
    weight_cap: float | None = 50.0
    fusion_grid: tuple = tuple(np.round(np.linspace(0, 1, 11), 2))
    calibrate: bool = True
    n_jobs: int = -1
    seed: int = 0


class EdgeEnsemble:
    def __init__(self, cfg: EdgeConfig):
        self.cfg = cfg
        self.classes_: np.ndarray | None = None     # global codes present at fit time
        self.xgb_: xgb.XGBClassifier | None = None
        self.lgb_: lgb.LGBMClassifier | None = None
        self.w_xgb_: float = 0.5
        self.T_: float = 1.0
        self.fit_info_: dict = {}

    # ---------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> "EdgeEnsemble":
        c = self.cfg
        self.classes_ = np.unique(y)
        lut = np.full(c.n_classes, -1)
        lut[self.classes_] = np.arange(len(self.classes_))
        yl, keep_v = lut[y], lut[yv] >= 0               # val rows of unseen classes can't be scored
        Xv_, yvl = Xv[keep_v], lut[yv[keep_v]]
        sw = balanced_weights(y, c.weight_cap)
        swv = balanced_weights(yv[keep_v], c.weight_cap)
        k = len(self.classes_)
        t0 = time.perf_counter()
        if k == 1:
            raise ValueError("need at least two classes to train a detector")
        obj = {"objective": "multi:softprob", "num_class": k} if k > 2 else {"objective": "binary:logistic"}
        self.xgb_ = xgb.XGBClassifier(**{"tree_method": "hist", "n_jobs": c.n_jobs, "random_state": c.seed,
                                         "early_stopping_rounds": c.early_stopping, "eval_metric": "mlogloss" if k > 2 else "logloss",
                                         **obj, **c.xgb_params})
        self.xgb_.fit(X, yl, sample_weight=sw, eval_set=[(Xv_, yvl)], sample_weight_eval_set=[swv], verbose=False)
        t1 = time.perf_counter()
        self.lgb_ = lgb.LGBMClassifier(**{"n_jobs": c.n_jobs, "random_state": c.seed, "verbose": -1, **c.lgb_params})
        self.lgb_.fit(X, yl, sample_weight=sw, eval_set=[(Xv_, yvl)], eval_sample_weight=[swv],
                      callbacks=[lgb.early_stopping(c.early_stopping, verbose=False)])
        t2 = time.perf_counter()

        px, pl = self._raw(Xv_)
        losses = {w: nll(w * px + (1 - w) * pl, yvl) for w in c.fusion_grid}
        self.w_xgb_ = float(min(losses, key=lambda w: (losses[w], abs(w - 0.5))))
        pf = self.w_xgb_ * px + (1 - self.w_xgb_) * pl
        self.T_ = fit_temperature(pf, yvl) if c.calibrate else 1.0
        self.fit_info_ = {
            "n_train": int(len(y)), "n_val": int(len(yvl)), "classes": self.classes_.tolist(),
            "xgb_best_iteration": int(getattr(self.xgb_, "best_iteration", -1) or -1),
            "lgb_best_iteration": int(self.lgb_.best_iteration_ or -1),
            "fit_seconds_xgb": round(t1 - t0, 2), "fit_seconds_lgb": round(t2 - t1, 2),
            "w_xgb": self.w_xgb_, "temperature": self.T_,
            "val_nll_by_w": {str(w): round(v, 5) for w, v in losses.items()},
        }
        return self

    # ---------------------------------------------------------------- predict
    def _raw(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Local-class probabilities from each model (columns = self.classes_)."""
        # LightGBM booster directly: skips the sklearn wrapper's per-call validation
        # (noticeable at batch size 1) and its feature-name warning on numpy input.
        pl = self.lgb_.booster_.predict(X, num_iteration=self.lgb_.best_iteration_ or None)
        if pl.ndim == 1:
            pl = np.column_stack([1 - pl, pl])
        return self.xgb_.predict_proba(X), pl

    def _scatter(self, p_local: np.ndarray) -> np.ndarray:
        out = np.zeros((len(p_local), self.cfg.n_classes), dtype=np.float64)
        out[:, self.classes_] = p_local
        return out

    def predict_components(self, X) -> dict[str, np.ndarray]:
        """Global-space probabilities: xgb, lgb, fused, calibrated (all needed by the M5 gate)."""
        px, pl = self._raw(X)
        fused = self.w_xgb_ * px + (1 - self.w_xgb_) * pl
        return {"xgb": self._scatter(px), "lgb": self._scatter(pl), "fused": self._scatter(fused),
                "calibrated": self._scatter(apply_temperature(fused, self.T_))}

    def predict_proba(self, X) -> np.ndarray:
        return self.predict_components(X)["calibrated"]

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(1)

    # ---------------------------------------------------------------- persistence
    def save(self, d: Path) -> dict:
        d.mkdir(parents=True, exist_ok=True)
        self.xgb_.save_model(d / "xgb.json")
        self.lgb_.booster_.save_model(str(d / "lgb.txt"))
        meta = {"classes": self.classes_.tolist(), "w_xgb": self.w_xgb_, "temperature": self.T_,
                "n_classes": self.cfg.n_classes, "fit_info": self.fit_info_}
        (d / "edge.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"xgb_bytes": (d / "xgb.json").stat().st_size, "lgb_bytes": (d / "lgb.txt").stat().st_size}


def measure_latency(model: EdgeEnsemble, X: np.ndarray, n_single: int = 300, batch: int = 10_000,
                    seed: int = 0) -> dict:
    """Edge inference cost: single-flow latency (P50/P95, ms) and batched throughput."""
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(X), n_single)
    model.predict_proba(X[:2])  # warm-up
    times = []
    for i in idx:
        t = time.perf_counter()
        model.predict_proba(X[i:i + 1])
        times.append((time.perf_counter() - t) * 1e3)
    b = X[:min(batch, len(X))]
    t = time.perf_counter()
    model.predict_proba(b)
    dt = time.perf_counter() - t
    times = np.asarray(times)
    return {"single_ms_p50": float(np.percentile(times, 50)), "single_ms_p95": float(np.percentile(times, 95)),
            "batch_rows": int(len(b)), "batch_us_per_row": float(dt / len(b) * 1e6)}
