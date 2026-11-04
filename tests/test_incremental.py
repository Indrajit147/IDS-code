import numpy as np

from ids.fl.rehearsal import RehearsalBuffer
from ids.models.edge import EdgeConfig
from ids.models.incremental import IncrementalEdge


def _rows(seed, classes, n=1500):
    """Each class is a separate blob in 4 features."""
    r = np.random.default_rng(seed)
    y = r.choice(classes, n)
    return (r.normal(size=(n, 4)) * 0.3 + y[:, None] * 2.0).astype(np.float32), y.astype(np.int64)


def _model(first=30, new=30):
    p = {"n_estimators": first, "max_delta_step": 10.0, "learning_rate": 0.1}
    return IncrementalEdge(EdgeConfig(n_classes=6, xgb_params={**p, "max_depth": 3}, lgb_params={**p, "num_leaves": 7, "min_child_samples": 5},
                                      early_stopping=10, n_jobs=2, seed=0), new_trees=new)


def test_both_boosters_learn_a_class_absent_from_the_first_fit():
    """Regression: LightGBM's default start for an absent class left it unable to learn that class at all."""
    m = _model()
    X, y = _rows(0, [0, 2])
    m.update(X, y, *_rows(1, [0, 2], 300))
    X, y = _rows(2, [0, 2, 4])
    Xv, yv = _rows(3, [0, 2, 4], 300)
    m.update(X, y, Xv, yv)
    Xt, yt = _rows(4, [4], 300)
    p = m.predict_components(Xt)
    for k in ("xgb", "lgb", "calibrated"):
        assert p[k].shape == (300, 6)
        assert (p[k].argmax(1) == 4).mean() > 0.95, k


def test_update_adds_trees_within_budget_and_keeps_old_classes():
    m = _model()
    X, y = _rows(0, [0, 2])
    m.update(X, y, *_rows(1, [0, 2], 300))
    before = m.tree_counts()
    X, y = _rows(2, [0, 2, 4])
    m.update(X, y, *_rows(3, [0, 2, 4], 300))
    after = m.tree_counts()
    assert before["xgb"] < after["xgb"] <= before["xgb"] + 30 * 6
    assert before["lgb"] < after["lgb"] <= before["lgb"] + 30 * 6
    Xt, yt = _rows(4, [0, 2], 300)
    assert (m.predict_components(Xt)["calibrated"].argmax(1) == yt).mean() > 0.95


def test_buffer_never_exceeds_capacity_and_is_class_balanced():
    b = RehearsalBuffer(300, np.random.default_rng(0))
    for s, classes in enumerate(([0, 1], [0, 1, 2], [0, 1, 2, 3])):
        b.add(*_rows(s, classes))
        assert len(b) <= 300
    X, y = b.get()
    assert len(X) == 300 and X.shape[1] == 4
    assert np.bincount(y).tolist() == [75, 75, 75, 75]
    assert b.peak_bytes <= 300 * (4 * 4 + 8)


def test_buffer_gives_unused_share_to_larger_classes():
    b = RehearsalBuffer(100, np.random.default_rng(0))
    b.add(np.zeros((10, 2), np.float32), np.zeros(10, np.int64))
    b.add(np.ones((500, 2), np.float32), np.ones(500, np.int64))
    assert np.bincount(b.get()[1]).tolist() == [10, 90]


def test_buffer_is_a_uniform_sample_over_all_rows_offered():
    """Rows from an early batch must not be favored or wiped out by later ones."""
    early = []
    for seed in range(200):
        b = RehearsalBuffer(50, np.random.default_rng(seed))
        b.add(np.zeros((500, 1), np.float32), np.zeros(500, np.int64))
        b.add(np.ones((500, 1), np.float32), np.zeros(500, np.int64))
        early.append((b.get()[0] == 0).mean())
    assert abs(np.mean(early) - 0.5) < 0.02


def test_buffer_copies_rows_rather_than_holding_the_callers_array():
    X = np.zeros((5, 2), np.float32)
    b = RehearsalBuffer(10, np.random.default_rng(0))
    b.add(X, np.zeros(5, np.int64))
    X[:] = 9
    assert (b.get()[0] == 0).all()
