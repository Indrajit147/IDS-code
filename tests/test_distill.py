import numpy as np
import pytest

from ids.drift.shift import psi_share
from ids.fl.distill import (GlobalCalibrator, accept_calibrator, aggregate_soft, apply_temperatures, dequantize,
                            drift_signals, fit_client_temperature, pseudo_label_weights, quantize, reliability_from_val,
                            uplink_bytes)
from ids.models.edge import EdgeConfig
from ids.models.incremental import IncrementalEdge


def test_mean_aggregation_averages_clients():
    preds = np.array([[[.8, .2]], [[.4, .6]]])
    assert aggregate_soft(preds) == pytest.approx(np.array([[.6, .4]]))


def test_reliability_ignores_a_client_on_the_class_it_cannot_judge():
    preds = np.array([[[.9, .1]], [[.1, .9]]])
    rel = np.array([[1., 1.], [0., 1.]])
    # class 0: only client A counts = .9; class 1: (.1 + .9) / 2 = .5; then renormalized
    assert aggregate_soft(preds, rel) == pytest.approx(np.array([[.9 / 1.4, .5 / 1.4]]))


def test_class_no_client_can_judge_gets_zero_not_nan():
    preds = np.array([[[.5, .5]], [[.5, .5]]])
    rel = np.array([[1., 0.], [1., 0.]])
    assert aggregate_soft(preds, rel) == pytest.approx(np.array([[1., 0.]]))


def test_reliability_is_per_class_f1_with_absent_classes_at_zero():
    r = reliability_from_val(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), 3)
    assert r == pytest.approx(np.array([.8, 2 / 3, 0.]))


def test_pseudo_label_weights_stop_the_majority_dominating():
    q = np.array([[.9, .1]] * 90 + [[.2, .8]] * 10)
    w = pseudo_label_weights(q)
    assert w[0] == pytest.approx(100 / (2 * 90))
    assert w[-1] == pytest.approx(100 / (2 * 10))


def test_quantize_round_trip_stays_on_the_simplex():
    p = np.array([[.5, .25, .25]])
    b = quantize(p)
    assert b.dtype == np.uint8 and b.tolist() == [[128, 64, 64]]
    d = dequantize(b)
    assert d.sum() == pytest.approx(1.0)
    assert np.abs(d - p).max() < 2 / 255


def test_uplink_bytes_counts_raw_quantized_and_compressed():
    p = np.tile(np.array([[1., 0, 0, 0, 0, 0]]), (1000, 1))
    u = uplink_bytes([p, p], metadata_bytes=8)
    assert u["raw_bytes"] == 2 * 1000 * 6 * 4 + 8
    assert u["q8_bytes"] == 2 * 1000 * 6 + 8
    assert u["q8_zlib_bytes"] < 1000       # constant rows compress heavily


def test_drift_signals_are_disagreement_and_normalized_entropy():
    px, pl = np.array([[1., 0.]]), np.array([[0., 1.]])
    s = drift_signals(px, pl, client_shift=np.array([.25]))
    assert s[0] == pytest.approx([1.0, 1.0, .25])     # total disagreement; the mean (.5, .5) has maximal entropy


def _flat(k, rows, cls):
    p = np.full((rows, k), .02 / (k - 1))
    p[np.arange(rows), cls] = .98
    return p


def test_untrained_calibrator_is_the_geometric_mean_of_the_boosters():
    c = GlobalCalibrator(2, 0)
    assert c.predict(np.array([[.8, .2]]), np.array([[.2, .8]]), None) == pytest.approx(np.array([[.5, .5]]))


def test_calibrator_learns_a_correction_the_boosters_get_wrong():
    cls = np.random.default_rng(0).integers(0, 3, 600)
    px = _flat(3, 600, cls)
    q = _flat(3, 600, (cls + 1) % 3)          # consensus says the next class
    c = GlobalCalibrator(3, 0, l2=1e-6).fit(px, px, None, q)
    assert (c.predict(px, px, None).argmax(1) == (cls + 1) % 3).mean() > .99


def test_class_weights_change_what_the_calibrator_predicts_for_ambiguous_rows():
    n = 1000
    px = np.full((n, 2), .5)
    q = np.array([[1., 0.]] * 900 + [[0., 1.]] * 100)
    plain = GlobalCalibrator(2, 0, l2=1e-6).fit(px, px, None, q).predict(px[:1], px[:1], None)
    bal = GlobalCalibrator(2, 0, l2=1e-6).fit(px, px, None, q, pseudo_label_weights(q)).predict(px[:1], px[:1], None)
    assert plain[0, 0] == pytest.approx(.9, abs=.01)
    assert bal[0, 0] == pytest.approx(.5, abs=.01)


def test_calibrator_uses_drift_features():
    n = 1000
    extra = np.r_[np.zeros(n // 2), np.ones(n // 2)][:, None]
    px = np.full((n, 2), .5)
    q = np.where(extra > .5, [[0., 1.]], [[1., 0.]])
    c = GlobalCalibrator(2, 1, l2=1e-6).fit(px, px, extra, q)
    assert c.predict(px[:1], px[:1], np.array([[0.]]))[0, 0] > .9
    assert c.predict(px[:1], px[:1], np.array([[1.]]))[0, 1] > .9


def test_calibrator_survives_the_trip_to_the_clients():
    cls = np.random.default_rng(0).integers(0, 3, 300)
    px = _flat(3, 300, cls)
    c = GlobalCalibrator(3, 2).fit(px, px, np.zeros((300, 2)), _flat(3, 300, (cls + 1) % 3))
    blob = c.to_bytes()
    assert len(blob) == 3 * (3 + 3 + 2 + 1) * 4
    back = GlobalCalibrator.from_bytes(blob, 3, 2)
    assert back.predict(px, px, np.zeros((300, 2))) == pytest.approx(c.predict(px, px, np.zeros((300, 2))), abs=1e-4)


def test_psi_share_counts_shifted_features():
    r = np.random.default_rng(0)
    ref = r.normal(size=(5000, 2))
    cur = r.normal(size=(5000, 2))
    assert psi_share(ref, cur, alert=.25) == 0.0
    cur[:, 1] += 5
    assert psi_share(ref, cur, alert=.25) == .5


def _rows(seed, classes, n=1500):
    r = np.random.default_rng(seed)
    y = r.choice(classes, n)
    return (r.normal(size=(n, 4)) * 0.3 + y[:, None] * 2.0).astype(np.float32), y.astype(np.int64)


def _model():
    p = {"n_estimators": 30, "max_delta_step": 10.0, "learning_rate": 0.1}
    return IncrementalEdge(EdgeConfig(n_classes=6, xgb_params={**p, "max_depth": 3}, lgb_params={**p, "num_leaves": 7, "min_child_samples": 5},
                                      early_stopping=10, n_jobs=2, seed=0), new_trees=30)


def _soft(y, k=6):
    q = np.full((len(y), k), .01)
    q[np.arange(len(y)), y] = 1 - .01 * (k - 1)
    return q


def test_digest_teaches_each_booster_a_class_the_client_never_saw():
    """FedMD digest: a client without class 4 learns it from the consensus on public rows alone."""
    m = _model()
    m.update(*_rows(0, [0, 2]), *_rows(1, [0, 2], 300))
    before = m.tree_counts()
    Xp, yp = _rows(2, [0, 2, 4])
    m.digest(Xp, _soft(yp), np.ones(len(yp)), rounds=30)
    after = m.tree_counts()
    assert after["xgb"] == before["xgb"] + 30 * 6
    assert after["lgb"] == before["lgb"] + 30 * 6
    Xt, _ = _rows(3, [4], 300)
    p = m.predict_components(Xt)
    for k in ("xgb", "lgb"):
        assert (p[k].argmax(1) == 4).mean() > .9, k
    Xo, yo = _rows(4, [0, 2], 300)
    assert (m.predict_components(Xo)["xgb"].argmax(1) == yo).mean() > .95


def test_local_update_continues_after_a_digest():
    m = _model()
    m.update(*_rows(0, [0, 2]), *_rows(1, [0, 2], 300))
    Xp, yp = _rows(2, [0, 2, 4])
    m.digest(Xp, _soft(yp), np.ones(len(yp)), rounds=5)
    n = m.tree_counts()
    m.update(*_rows(3, [0, 2, 4]), *_rows(4, [0, 2, 4], 300))
    assert m.tree_counts()["xgb"] > n["xgb"] and m.tree_counts()["lgb"] > n["lgb"]


def test_per_row_temperature_sharpens_without_changing_the_predicted_class():
    p = np.array([[.6, .4], [.3, .7]])
    out = apply_temperatures(p, np.array([.5, 1.0]))
    assert out[0] == pytest.approx([.36 / (.36 + .16), .16 / (.36 + .16)])    # p^(1/T), renormalized
    assert out[1] == pytest.approx([.3, .7])                                  # T = 1 leaves a row alone
    assert (out.argmax(1) == p.argmax(1)).all()


def test_temperature_undoes_an_underconfident_calibrator():
    """The target of g_t is a mean over clients, which is soft. A client's own temperature must recover from that."""
    r = np.random.default_rng(0)
    y = r.integers(0, 2, 4000)
    soft = np.full((4000, 2), .35)
    soft[np.arange(4000), y] = .65         # right every time, but only 65% sure
    T = fit_client_temperature(soft, y)
    assert T < .2
    assert apply_temperatures(soft, np.full(4000, T)).max(1).mean() > .95


def test_guard_keeps_the_local_model_when_the_calibrator_is_not_better():
    y = np.array([0, 0, 1, 1])
    good = np.array([[.9, .1], [.9, .1], [.1, .9], [.1, .9]])
    bad = np.array([[.9, .1], [.9, .1], [.9, .1], [.1, .9]])
    assert accept_calibrator(y, local=bad, calibrated=good)
    assert not accept_calibrator(y, local=good, calibrated=bad)
    assert not accept_calibrator(y, local=good, calibrated=good)      # a tie keeps what the client has
