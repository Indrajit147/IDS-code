import numpy as np

from ids.eval.explain import ensemble_shap, ranking_change, top_k
from ids.models.edge import EdgeConfig
from ids.models.incremental import IncrementalEdge


def _fitted(updates=2):
    r = np.random.default_rng(7)
    X = r.normal(size=(5000, 12)).astype(np.float32)
    y = ((X[:, 0] + X[:, 3]) > 0).astype(int) + (X[:, 5] > 1)
    m = IncrementalEdge(EdgeConfig(n_classes=4, xgb_params={"max_depth": 4, "eta": 0.2, "n_estimators": 8},
                                   lgb_params={"num_leaves": 15, "learning_rate": 0.2, "n_estimators": 8},
                                   early_stopping=5, weight_cap=10.0, n_jobs=2, seed=1), new_trees=6)
    for _ in range(updates):    # continued boosting, as on the edge, with a class (3) never seen
        m.update(X[:4000], y[:4000], X[4000:], y[4000:])
    return m, X[4000:]


def test_attributions_add_up_to_the_weighted_margin_of_the_explained_class():
    m, X = _fitted()
    out = ensemble_shap(m, X)
    px, pl = m.xgb_.predict(__import__("xgboost").DMatrix(X), output_margin=True), m.lgb_.predict(X, raw_score=True)
    margin = m.w_xgb_ * px + (1 - m.w_xgb_) * pl
    r = np.arange(len(X))
    np.testing.assert_allclose(out["phi"].sum(1) + out["bias"], margin[r, out["cls"]], rtol=1e-4, atol=1e-4)
    assert (out["cls"] == m.predict_components(X)["calibrated"].argmax(1)).all()
    # The informative features carry the attribution, the noise features almost none
    imp = np.abs(out["phi"]).mean(0)
    assert set(np.argsort(-imp)[:3]) == {0, 3, 5}


def test_one_thread_gives_the_same_attributions():
    m, X = _fitted(1)
    a, b = ensemble_shap(m, X[:200]), ensemble_shap(m, X[:200], threads=1)
    np.testing.assert_allclose(a["phi"], b["phi"], rtol=1e-5, atol=1e-6)
    assert m.xgb_.predict(__import__("xgboost").DMatrix(X[:5])).shape == (5, 4)    # thread setting restored, model still usable


def test_top_k_and_ranking_change():
    phi = np.array([[0.1, -0.5, 0.3], [0.0, 0.2, -0.9]])
    idx, val = top_k(phi, 2)
    assert idx.tolist() == [[1, 2], [2, 1]] and val.tolist() == [[-0.5, 0.3], [-0.9, 0.2]]
    a = np.array([5.0, 4, 3, 2, 1])
    same = ranking_change(a, a * 2, 2)
    assert abs(same["change"]) < 1e-12 and same["top_overlap"] == 1.0
    rev = ranking_change(a, a[::-1].copy(), 2)
    assert abs(rev["change"] - 2.0) < 1e-12 and rev["top_overlap"] == 0.0
