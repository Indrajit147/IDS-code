import numpy as np
import pandas as pd

from ids.drift.adwin import Adwin, detect
from ids.drift.gate import drift_state, escalate, frontier, gate_outcome, gate_signals, tune
from ids.drift.shift import bin_edges, bin_shares, feature_shift, kl, psi


def test_adwin_tracks_exact_window_statistics():
    x = np.random.default_rng(0).random(5000)
    a = Adwin()
    for v in x:
        a.update(float(v))
    w = x[-a.width:]
    assert np.isclose(a.total, w.sum()) and np.isclose(a.variance, ((w - w.mean()) ** 2).sum(), rtol=1e-6)
    assert sum(2 ** i * len(lv) for i, lv in enumerate(a.levels)) == a.width
    assert all(len(lv) <= a.max_buckets for lv in a.levels)


def test_adwin_quiet_on_stationary_streams():
    fires = sum(len(detect(np.random.default_rng(s).binomial(1, 0.2, 20000))) for s in range(5))
    assert fires <= 1


def test_adwin_detects_rise_soon_and_drops_old_data():
    r = np.random.default_rng(1)
    x = np.concatenate([r.binomial(1, 0.1, 5000), r.binomial(1, 0.5, 5000)])
    fires = detect(x)
    after = [p for p, change in fires if p >= 5000 and change > 0]
    assert after and after[0] - 5000 < 300
    assert not [p for p, _ in fires if p < 5000]
    a = Adwin()
    for v in x:
        a.update(float(v))
    assert a.width <= 5100 and abs(a.mean - 0.5) < 0.05


def test_detect_gives_one_event_per_change():
    r = np.random.default_rng(4)
    assert len(detect(np.concatenate([r.binomial(1, 0.1, 8000), r.binomial(1, 0.5, 8000)]))) == 1


def test_adwin_reports_falling_mean_as_negative():
    r = np.random.default_rng(2)
    fires = detect(np.concatenate([r.binomial(1, 0.6, 3000), r.binomial(1, 0.1, 3000)]))
    assert fires and fires[0][1] < 0


def test_bins_handle_constant_and_out_of_range_values():
    edges = bin_edges(np.zeros(100))
    assert bin_shares(np.ones(50), edges)[-1] > 0.9              # new values leave the reference's bin
    assert np.isclose(bin_shares(np.zeros(50), edges).sum(), 1)
    assert list(bin_edges(np.array([0.0] * 95 + [5.0] * 5))) == [0.0, 5.0]   # zero-inflated: edges collapse


def test_shift_metrics_grow_with_shift_and_are_zero_when_identical():
    r = np.random.default_rng(3)
    ref = r.normal(size=(5000, 2))
    same = feature_shift(ref, ref, ["a", "b"])
    assert np.allclose(same[["psi", "kl", "wasserstein"]], 0)
    moved = feature_shift(ref, ref + [0.2, 2.0], ["small", "big"]).set_index("feature")
    for m in ("psi", "kl", "wasserstein"):
        assert moved.loc["big", m] > moved.loc["small", m] > 0
    assert np.isclose(moved.loc["big", "wasserstein"], 2.0, atol=0.05)     # in units of the reference std


def test_psi_is_symmetric_and_kl_is_not():
    p, q = np.array([0.7, 0.3]), np.array([0.4, 0.6])
    assert np.isclose(psi(p, q), psi(q, p)) and not np.isclose(kl(p, q), kl(q, p))


def test_gate_signals_and_rule():
    comps = {"xgb": np.array([[1.0, 0.0], [0.5, 0.5]]), "lgb": np.array([[0.0, 1.0], [0.5, 0.5]]),
             "calibrated": np.array([[1.0, 0.0], [0.5, 0.5]])}
    s = gate_signals(comps)
    assert s["disagreement"].tolist() == [1.0, 0.0] and s["confidence"].tolist() == [1.0, 0.5]
    assert np.allclose(s["entropy"], [0.0, 1.0])                  # log 0 handled, normalized to [0, 1]
    assert escalate(s["confidence"], s["disagreement"], 0.8, 0.2).tolist() == [True, True]
    assert escalate(np.array([0.9]), np.array([0.1]), 0.8, 0.2).tolist() == [False]
    assert escalate(np.array([0.9]), np.array([0.1]), 0.8, 0.2, np.array([True])).tolist() == [True]


def test_drift_state_holds_for_following_flows_only():
    assert drift_state(6, [1], 2).tolist() == [False, False, True, True, False, False]
    assert drift_state(3, [2], 5).tolist() == [False] * 3
    assert not drift_state(3, [], 5).any()


def test_gate_outcome_and_tuning_respect_budget():
    o = gate_outcome(np.array([True, True, False, False]), np.array([1.0, 0.0, 1.0, 0.0]))
    assert o == {"escalation_rate": 0.5, "error_capture": 0.5, "error_rate_in_escalated": 0.5, "accuracy_not_escalated": 0.5}
    conf = np.array([0.1, 0.2, 0.6, 0.9, 0.95])
    err = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    f = frontier(conf, np.zeros(5), err, [0.15, 0.3, 0.7, 1.0], [0.5])
    assert tune(f, 0.4)["tau_c"] == 0.3
    assert tune(f, 1.0)["tau_c"] == 0.7                           # cheapest setting that captures every error
    assert isinstance(f, pd.DataFrame)
    try:
        tune(f.assign(escalation_rate=0.9), 0.1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
