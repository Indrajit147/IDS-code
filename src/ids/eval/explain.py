"""TreeSHAP for the incremental edge ensemble, from XGBoost's and LightGBM's own exact implementations.

The edge averages the two models' probabilities, and SHAP is additive only in each model's margin. So a row is
explained by w * phi_xgb + (1 - w) * phi_lgb for one class, which sums exactly to that class's weighted margin.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr


def ensemble_shap(model, X: np.ndarray, cls: np.ndarray | None = None, threads: int | None = None) -> dict[str, np.ndarray]:
    """Attributions of each row's class cls (default: the class the edge predicts) and the weighted margins they sum to."""
    n, d = X.shape
    k = model.cfg.n_classes
    if threads is not None:
        model.xgb_.set_param("nthread", threads)
    dm = xgb.DMatrix(X, nthread=threads or -1)
    cx = model.xgb_.predict(dm, pred_contribs=True).reshape(n, k, d + 1)
    cl = model.lgb_.predict(X, pred_contrib=True, num_threads=threads or 0).reshape(n, k, d + 1)
    mx = model.xgb_.predict(dm, output_margin=True).reshape(n, k)
    ml = model.lgb_.predict(X, raw_score=True, num_threads=threads or 0).reshape(n, k)
    if threads is not None:
        model.xgb_.set_param("nthread", model.cfg.n_jobs)
    if cls is None:
        cls = model.predict_components(X)["calibrated"].argmax(1)
    w, r = model.w_xgb_, np.arange(n)
    phi = w * cx[r, cls] + (1 - w) * cl[r, cls]
    margin = w * mx + (1 - w) * ml
    gap = np.abs(phi.sum(1) - margin[r, cls]).max(initial=0.0)
    assert gap < 1e-3 * max(1.0, np.abs(margin).max(initial=0.0)), f"TreeSHAP doesn't add up to the margin: {gap}"
    return {"phi": phi[:, :d], "bias": phi[:, d], "cls": cls, "margin_argmax": margin.argmax(1)}


def top_k(phi: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and signed values of each row's k largest attributions by size, largest first."""
    idx = np.argsort(-np.abs(phi), axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(phi, idx, 1)


def ranking_change(a: np.ndarray, b: np.ndarray, k: int) -> dict[str, float]:
    """1 - Spearman rho between two mean-|attribution| vectors, and how many of their top k features are shared."""
    rho = spearmanr(a, b).statistic
    ta, tb = set(np.argsort(-a, kind="stable")[:k]), set(np.argsort(-b, kind="stable")[:k])
    return {"change": float(1 - rho), "top_overlap": len(ta & tb) / k}
