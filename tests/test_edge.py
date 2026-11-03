import numpy as np
import pytest

from ids.eval.metrics import detection_metrics, ece
from ids.models.edge import (EdgeConfig, EdgeEnsemble, apply_temperature, balanced_weights,
                             fit_temperature, nll)

FAST = {"n_estimators": 30, "max_depth": 4}
FAST_LGB = {"n_estimators": 30, "num_leaves": 15}


def _data(n=3000, classes=(0, 2, 5), seed=0):
    r = np.random.default_rng(seed)
    y = r.choice(classes, n)
    X = r.normal(size=(n, 6)).astype(np.float32)
    X[:, 0] += y * 1.5                       # learnable signal
    return X, y


def _fit(classes=(0, 2, 5), n_classes=6):
    X, y = _data(classes=classes)
    cut = int(len(y) * 0.8)
    m = EdgeEnsemble(EdgeConfig(n_classes=n_classes, xgb_params=FAST, lgb_params=FAST_LGB, n_jobs=2))
    return m.fit(X[:cut], y[:cut], X[cut:], y[cut:]), X, y


def test_global_label_space_with_absent_classes():
    m, X, y = _fit()
    comps = m.predict_components(X[:50])
    for name, p in comps.items():
        assert p.shape == (50, 6), name
        assert np.allclose(p.sum(1), 1), name
        assert np.all(p[:, [1, 3, 4]] == 0), name          # never-seen classes stay exactly 0
    acc = (m.predict(X) == y).mean()
    assert acc > 0.8
    assert 0 <= m.w_xgb_ <= 1 and 0.05 <= m.T_ <= 10


def test_binary_case_and_save(tmp_path):
    m, X, y = _fit(classes=(0, 1), n_classes=2)
    assert m.predict_proba(X[:5]).shape == (5, 2)
    sizes = m.save(tmp_path / "edge")
    assert sizes["xgb_bytes"] > 0 and sizes["lgb_bytes"] > 0
    assert (tmp_path / "edge" / "edge.json").exists()


def test_single_class_rejected():
    X, y = _data(classes=(3,))
    with pytest.raises(ValueError):
        EdgeEnsemble(EdgeConfig(n_classes=6, xgb_params=FAST, lgb_params=FAST_LGB)).fit(X, y, X, y)


def test_temperature_and_weights():
    r = np.random.default_rng(1)
    y = r.integers(0, 3, 2000)
    logits = r.normal(size=(2000, 3)); logits[np.arange(2000), y] += 1.0
    over = np.exp(logits * 4); over /= over.sum(1, keepdims=True)  # overconfident
    T = fit_temperature(over, y)
    assert T > 1.5 and nll(apply_temperature(over, T), y) < nll(over, y)
    assert ece(apply_temperature(over, T), y) < ece(over, y)
    p = np.array([[0.7, 0.3, 0.0]]); assert apply_temperature(p, 2.0)[0, 2] == 0
    w = balanced_weights(np.array([0, 0, 0, 1]))
    assert np.isclose(w[0] * 3, w[3] * 1)
    assert balanced_weights(np.array([0] * 999 + [1]), cap=10).max() == 10


def test_metrics_use_present_classes():
    y = np.array([0, 0, 2, 2])
    p = np.eye(6)[[0, 0, 2, 1]]
    met = detection_metrics(y, p, ["a", "b", "c", "d", "e", "f"])
    assert set(met["per_class"]) == {"a", "c"}
    assert np.isclose(met["accuracy"], 0.75)
