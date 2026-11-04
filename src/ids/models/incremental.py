"""Incremental edge model: XGBoost continued boosting + LightGBM init_model over a fixed label space.

Differs from EdgeEnsemble in one structural way: both boosters always have one output per global
class, because continued boosting cannot add outputs when a new class appears later. A class not yet
seen therefore has a very low score rather than exactly zero, and it must stay trainable.

That needs care in LightGBM. Absent classes saturate (p around 1e-13), so the softmax Hessians of the
rows that later introduce them vanish and no leaf passes min_sum_hessian_in_leaf; setting that to zero
crashes, because LightGBM estimates leaf row counts from Hessians. So LightGBM trains with the softmax
gradients and a floored Hessian, and max_delta_step bounds the resulting leaf outputs. The standard
objective fails the regression test in tests/test_incremental.py: 0% correct on a new class.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from .edge import EdgeConfig, apply_temperature, balanced_weights, fit_temperature, nll

HESSIAN_FLOOR = 1e-2


def _softmax(raw: np.ndarray) -> np.ndarray:
    z = raw - raw.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _softmax_floored(raw: np.ndarray, data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Softmax gradients with the Hessian floored, for lgb.train's callable objective."""
    p, y, w = _softmax(raw), data.get_label().astype(int), data.get_weight()
    g = p.copy()
    g[np.arange(len(y)), y] -= 1
    h = np.maximum(p * (1 - p), HESSIAN_FLOOR)
    return g * w[:, None], h * w[:, None]


def _weighted_logloss(raw: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
    y, w = data.get_label().astype(int), data.get_weight()
    p = _softmax(raw)[np.arange(len(y)), y]
    return "mlogloss", float(-np.average(np.log(np.clip(p, 1e-15, 1)), weights=w)), False


class IncrementalEdge:
    def __init__(self, cfg: EdgeConfig, new_trees: int):
        self.cfg, self.new_trees = cfg, new_trees
        self.xgb_: xgb.Booster | None = None
        self.lgb_: lgb.Booster | None = None
        self.w_xgb_, self.T_ = 0.5, 1.0
        self.n_updates = 0

    def update(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> "IncrementalEdge":
        """Add trees on (X, y); the first call fits the full configured number of rounds."""
        c = self.cfg
        sw, swv = balanced_weights(y, c.weight_cap), balanced_weights(yv, c.weight_cap)
        xp = {k: v for k, v in c.xgb_params.items() if k != "n_estimators"}
        lp = {k: v for k, v in c.lgb_params.items() if k != "n_estimators"}

        rounds = self.new_trees if self.n_updates else c.xgb_params["n_estimators"]
        self.xgb_ = self._boost_xgb(xp, X, y, sw, Xv, yv, swv, rounds)
        rounds = self.new_trees if self.n_updates else c.lgb_params["n_estimators"]
        self.lgb_ = self._boost_lgb(lp, X, y, sw, Xv, yv, swv, rounds)

        px, pl = self._raw(Xv)
        losses = {w: nll(w * px + (1 - w) * pl, yv) for w in c.fusion_grid}
        self.w_xgb_ = float(min(losses, key=lambda w: (losses[w], abs(w - 0.5))))
        self.T_ = fit_temperature(self.w_xgb_ * px + (1 - self.w_xgb_) * pl, yv) if c.calibrate else 1.0
        self.n_updates += 1
        return self

    def _boost_xgb(self, params, X, y, sw, Xv, yv, swv, rounds) -> xgb.Booster:
        c = self.cfg
        params = {"objective": "multi:softprob", "num_class": c.n_classes, "tree_method": "hist",
                  "eval_metric": "mlogloss", "nthread": c.n_jobs, "seed": c.seed,
                  "base_score": 0.5, "min_child_weight": 1e-3, **params}   # base_score fixed: its default differs across versions
        b = xgb.train(params, xgb.DMatrix(X, y, weight=sw), rounds, evals=[(xgb.DMatrix(Xv, yv, weight=swv), "val")],
                      early_stopping_rounds=c.early_stopping, xgb_model=self.xgb_, verbose_eval=False)
        return b[: b.best_iteration + 1]

    def _boost_lgb(self, params, X, y, sw, Xv, yv, swv, rounds) -> lgb.Booster:
        c = self.cfg
        params = {"objective": _softmax_floored, "num_class": c.n_classes, "metric": "None",
                  "num_threads": c.n_jobs, "seed": c.seed, "verbose": -1, **params}
        train = lgb.Dataset(X, y, weight=sw)
        b = lgb.train(params, train, rounds, init_model=self.lgb_, valid_sets=[train.create_valid(Xv, yv, weight=swv)],
                      feval=_weighted_logloss, callbacks=[lgb.early_stopping(c.early_stopping, verbose=False)])
        # best_iteration counts the init model's rounds too; keep the trees up to it, so the next update continues from there
        return lgb.Booster(model_str=b.model_to_string(num_iteration=b.best_iteration))

    def _raw(self, X) -> tuple[np.ndarray, np.ndarray]:
        return self.xgb_.predict(xgb.DMatrix(X)), _softmax(self.lgb_.predict(X, raw_score=True))

    def predict_components(self, X) -> dict[str, np.ndarray]:
        """Same variants as EdgeEnsemble, so the harness and the M5 gate work unchanged."""
        px, pl = self._raw(X)
        fused = self.w_xgb_ * px + (1 - self.w_xgb_) * pl
        return {"xgb": px, "lgb": pl, "fused": fused, "calibrated": apply_temperature(fused, self.T_)}

    def tree_counts(self) -> dict[str, int]:
        return {"xgb": self.xgb_.num_boosted_rounds() * self.cfg.n_classes, "lgb": self.lgb_.num_trees()}

    def size_bytes(self) -> int:
        return len(self.xgb_.save_raw("json")) + len(self.lgb_.model_to_string())
