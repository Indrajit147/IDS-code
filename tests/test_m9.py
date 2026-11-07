import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from ids.drift.gate import adaptive_taus, drift_score, escalate, same_cost_mask
from ids.models.edge import EdgeConfig
from ids.models.incremental import IncrementalEdge

scripts = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("m9_e4", scripts / "m9_e4.py")
m9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m9)


def test_zero_score_or_strength_gives_exactly_the_fixed_gate():
    r = np.random.default_rng(0)
    conf, dis = r.random(1000), r.random(1000)
    fixed = escalate(conf, dis, 0.6, 0.4)
    tc, td = adaptive_taus(0.6, 0.4, np.zeros(1000), 8)
    assert (escalate(conf, dis, tc, td) == fixed).all()
    tc, td = adaptive_taus(0.6, 0.4, r.random(1000), 0)
    assert (escalate(conf, dis, tc, td) == fixed).all()


def test_a_higher_score_only_adds_escalations():
    r = np.random.default_rng(1)
    conf, dis, score = r.random(1000), r.random(1000), r.random(1000)
    fixed = escalate(conf, dis, 0.6, 0.4)
    tc, td = adaptive_taus(0.6, 0.4, score, 2)
    assert (fixed & ~escalate(conf, dis, tc, td)).sum() == 0
    assert escalate(conf, dis, tc, td).sum() > fixed.sum()
    tc, td = adaptive_taus(0.6, 0.4, np.ones(1000), 100)     # capped: everything escalates, thresholds stay in range
    assert tc.max() == 1.0 and td.min() == 0.0


def test_drift_score_rises_after_the_change_and_never_before_it():
    r = np.random.default_rng(2)
    n0, n1 = 3000, 3000
    s = np.concatenate([r.binomial(1, 0.05, n0), r.binomial(1, 0.5, n1)]).astype(float)
    score, fires = drift_score([s], n0, 0.002)
    first = np.flatnonzero(score > 0)[0]
    assert fires and first == fires[0] + 1      # the row that fired it isn't scored by it
    assert 0.1 < score[-1] <= 1     # ADWIN's estimate when it fires, with old rows still in its window, so below the true 0.45
    assert (np.diff(score) >= 0).all()
    assert drift_score([r.binomial(1, 0.05, 6000).astype(float)], 3000, 0.002)[0].max() == 0


def test_a_fall_gives_no_score():
    r = np.random.default_rng(3)
    s = np.concatenate([r.binomial(1, 0.5, 3000), r.binomial(1, 0.05, 3000)]).astype(float)
    score, fires = drift_score([s], 3000, 0.002)
    assert fires and score.max() == 0


def test_pick_k_and_strength():
    assert m9.pick_k({30: 0.8, 10: 0.9, 20: 0.9}) == 10     # a tie goes to the smaller K
    rows = pd.DataFrame({"strength": [0, 0, 1, 1, 2, 2], "error_capture": [.2, .2, .3, .3, .9, .9], "escalation_rate": [.08, .08, .09, .09, .2, .2]})
    assert m9.pick_strength(rows, 0.10) == 1                 # the strongest catch is over the target
    assert m9.pick_strength(rows, 0.01) == 0                 # nothing fits: the fixed gate
    rows = pd.DataFrame({"strength": [0, 1], "error_capture": [.2, .2], "escalation_rate": [.05, .05]})
    assert m9.pick_strength(rows, 0.10) == 0                 # a tie goes to the weaker


def test_recovery_curve_applies_each_client_version_from_its_round():
    y = np.array([0, 0, 1, 1])
    cold = np.array([1, 1, 1, 1])
    final = y.copy()
    client0 = np.array([True, True, False, False])
    c = m9.recovery_curve(y, cold, [(client0, 2, np.array([0, 0]))], final, 4)
    assert len(c) == 5
    assert c[0] == c[1] < c[2] == c[3]                       # takes effect at round 2
    assert c[2] == 1.0 and c[4] == 1.0
    assert m9.recovery_share(c, 1.0, 0.02) == 0.5
    assert m9.recovery_share([0.2, 0.2, 0.3], 1.0, 0.02) is None
    assert m9.recovery_share([0.99, 0.2], 1.0, 0.02) == 0     # never dropped


def _edge(n_classes=3):
    return IncrementalEdge(EdgeConfig(n_classes=n_classes, xgb_params={"max_depth": 4, "eta": 0.2, "colsample_bytree": 0.8, "n_estimators": 8},
                                      lgb_params={"num_leaves": 15, "learning_rate": 0.2, "colsample_bytree": 0.8, "n_estimators": 8},
                                      early_stopping=5, weight_cap=10.0, n_jobs=2, seed=1), new_trees=8)


def test_new_trees_use_only_the_allowed_features_and_old_trees_still_count():
    r = np.random.default_rng(4)
    X = r.normal(size=(6000, 20))
    y = ((X[:, 0] + X[:, 15] + X[:, 16]) > 0).astype(int) + (X[:, 17] > 1)
    a, b = _edge(), _edge()
    tr, va = np.arange(5000), np.arange(5000, 6000)
    a.update(X[tr], y[tr], X[va], y[va])
    b.update(X[tr], y[tr], X[va], y[va])
    before = a.predict_components(X[va])["calibrated"]
    allowed = np.arange(5)
    rounds = a.xgb_.num_boosted_rounds()
    imp = a.lgb_.feature_importance("split")
    a.update(X[tr], y[tr], X[va], y[va], allowed=allowed)      # raises if a forbidden feature is split on
    new = a.xgb_[rounds:].get_score(importance_type="weight")
    assert {int(f[1:]) for f in new} <= set(allowed.tolist())
    assert np.flatnonzero(a.lgb_.feature_importance("split") > imp).max(initial=0) < 5
    assert a.predict_components(X[va])["calibrated"].shape == before.shape
    # The old trees still see real values at inference, so the forbidden signal is kept, not lost.
    assert (a.predict_components(X[va])["calibrated"].argmax(1) == y[va]).mean() > 0.8


def test_a_leak_into_a_forbidden_feature_trains_again_with_another_seed(monkeypatch):
    import xgboost as xgb
    r = np.random.default_rng(6)
    X = r.normal(size=(3000, 20))
    y = (X[:, 15] + X[:, 1] > 0).astype(int)
    e = _edge(2)
    e.update(X[:2500], y[:2500], X[2500:], y[2500:])
    rounds, real, seeds = e.xgb_.num_boosted_rounds(), xgb.train, []

    def leak_once(params, train, *a, **k):
        seeds.append(params["seed"])
        if len(seeds) == 1:     # the first attempt ignores the limit, so it splits on feature 15, as the signal is there
            train = xgb.DMatrix(X[:2500], y[:2500])
            params = {**params, "colsample_bytree": 1.0}
        return real(params, train, *a, **k)

    monkeypatch.setattr("ids.models.incremental.xgb.train", leak_once)
    e.update(X[:2500], y[:2500], X[2500:], y[2500:], allowed=np.arange(5))
    assert seeds == [1, 2]
    assert {int(f[1:]) for f in e.xgb_[rounds:].get_score(importance_type="weight")} <= set(range(5))


def test_same_cost_mask_meets_the_rate_and_keeps_what_is_already_flagged():
    conf = np.array([.9, .1, .5, .3, .7])
    other = np.array([True, False, False, False, False])
    m = same_cost_mask(conf, other, 0.6)
    assert m.tolist() == [True, True, False, True, False]
    assert same_cost_mask(conf, other, 0.0).tolist() == other.tolist()     # already over the rate: nothing is dropped


@pytest.mark.parametrize("n_allowed", [1, 3, 10])
def test_allowed_holds_at_small_k(n_allowed):
    r = np.random.default_rng(5)
    X = r.normal(size=(3000, 43))
    y = (X[:, 40] + X[:, 41] > 0).astype(int)
    e = _edge(2)
    e.update(X[:2500], y[:2500], X[2500:], y[2500:], allowed=np.arange(n_allowed))
    assert {int(f[1:]) for f in e.xgb_.get_score(importance_type="weight")} <= set(range(n_allowed))
