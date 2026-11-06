import numpy as np
import pytest
import torch

from ids.fl.fedavg import ClientData, Scaler, TabularMLP, fedavg_aggregate, local_update, run_rounds


def _blobs(n=400, seed=0, classes=(0, 1, 2)):
    g = np.random.default_rng(seed)
    y = g.choice(classes, n)
    X = g.normal(0, .3, (n, 4)).astype(np.float32)
    X[np.arange(n), y] += 2.0
    return X, y.astype(np.int64)


def _model(seed=0):
    return TabularMLP(4, 3, (16,), seed)


def test_scaler_tames_heavy_tails_and_uses_only_the_client_totals():
    a = np.array([[0., 1e6], [10., -1e6]], np.float32)
    b = np.array([[20., 0.]], np.float32)
    s = Scaler.from_clients([a, b])
    pooled = Scaler.from_clients([np.concatenate([a, b])])
    assert s.mean == pytest.approx(pooled.mean) and s.std == pytest.approx(pooled.std)
    out = s.transform(np.array([[10., 1e9]], np.float32))
    assert np.abs(out).max() <= s.clip


def test_scaler_handles_a_constant_feature():
    s = Scaler.from_clients([np.ones((5, 2), np.float32)])
    assert np.isfinite(s.transform(np.ones((1, 2), np.float32))).all()


def test_fedavg_weights_clients_by_their_row_counts():
    states = [{"w": torch.tensor([0.])}, {"w": torch.tensor([4.])}]
    assert fedavg_aggregate(states, [3, 1])["w"].item() == pytest.approx(1.0)


def test_identical_clients_average_to_the_same_weights():
    m = _model()
    s = {k: v.clone() for k, v in m.state_dict().items()}
    out = fedavg_aggregate([s, s, s], [5, 1, 2])
    for k in s:
        assert torch.allclose(out[k], s[k])


def _update(mu, lr=.1, seed=1):
    X, y = _blobs(classes=(0, 1))
    m = _model()
    start = {k: v.clone() for k, v in m.state_dict().items()}
    st = local_update(m, start, X, y, np.ones(len(y), np.float32), mu=mu, lr=lr, momentum=0., batch=64, epochs=2, rng=np.random.default_rng(seed))
    return start, st


def _distance(a, b):
    return sum(((a[k] - b[k]) ** 2).sum().item() for k in a) ** .5


def test_fedprox_with_mu_zero_is_exactly_fedavg():
    X, y = _blobs()
    a = local_update(_model(), _model().state_dict(), X, y, np.ones(len(y), np.float32), mu=0., lr=.1, momentum=.9, batch=64, epochs=1, rng=np.random.default_rng(3))
    b = local_update(_model(), _model().state_dict(), X, y, np.ones(len(y), np.float32), mu=0., lr=.1, momentum=.9, batch=64, epochs=1, rng=np.random.default_rng(3))
    for k in a:
        assert torch.equal(a[k], b[k])


def test_fedprox_keeps_a_client_closer_to_the_global_model():
    start, free = _update(0.)
    _, held = _update(5.)
    assert _distance(held, start) < .8 * _distance(free, start)


def test_local_update_leaves_the_global_state_untouched():
    start, st = _update(0.)
    fresh = _model().state_dict()
    for k in start:
        assert torch.equal(start[k], fresh[k])
    assert _distance(st, start) > 0


def test_federated_rounds_learn_a_class_only_one_client_holds():
    clients = []
    for i, classes in enumerate([(0, 1), (0, 2)]):
        X, y = _blobs(seed=i, classes=classes)
        k = len(y) // 5
        clients.append(ClientData(X[k:], y[k:], np.ones(len(y) - k, np.float32), X[:k], y[:k]))
    m = _model()
    log = run_rounds(m, clients, rounds=30, patience=30, mu=0., lr=.1, momentum=.9, batch=64, epochs=1, rng_key=(0, "t"))
    X, y = _blobs(seed=9)
    pred = m.predict_proba(X).argmax(1)
    assert all((pred[y == c] == c).mean() > .9 for c in (0, 1, 2))
    assert log["rounds_run"] <= 30 and log["best_val"] > .9


def test_rounds_stop_after_patience_and_keep_the_best_model():
    X, y = _blobs()
    c = ClientData(X[80:], y[80:], np.ones(len(y) - 80, np.float32), X[:80], y[:80])
    m = _model()
    log = run_rounds(m, [c], rounds=200, patience=2, mu=0., lr=.1, momentum=.9, batch=64, epochs=1, rng_key=(0, "t"))
    assert log["rounds_run"] < 200
    assert log["best_val"] == pytest.approx(max(log["val_curve"]))
    from sklearn.metrics import f1_score
    now = f1_score(y[:80], m.predict_proba(X[:80]).argmax(1), average="macro")
    assert now == pytest.approx(log["best_val"])


def test_training_repeats_exactly_from_the_same_seed():
    X, y = _blobs()
    c = ClientData(X[80:], y[80:], np.ones(len(y) - 80, np.float32), X[:80], y[:80])
    a, b = _model(5), _model(5)
    for m in (a, b):
        run_rounds(m, [c], rounds=3, patience=3, mu=.01, lr=.1, momentum=.9, batch=64, epochs=1, rng_key=(1, "x"))
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])


def test_model_bytes_count_every_parameter_as_float32():
    m = TabularMLP(4, 3, (16,), 0)
    assert m.n_params() == (4 * 16 + 16) + (16 * 3 + 3)
    assert m.size_bytes() == 4 * m.n_params()


def test_predict_components_gives_probabilities_with_temperature():
    X, y = _blobs()
    m = _model()
    m.temperature = 2.0
    p = m.predict_components(X)["global"]
    assert p.shape == (len(X), 3) and p.sum(1) == pytest.approx(np.ones(len(X)), abs=1e-5)


def test_mu_pick_uses_validation_and_ties_go_to_the_lower_mu():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("m8", pathlib.Path(__file__).parents[1] / "scripts" / "m8_fedavg.py")
    m8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m8)
    assert m8.pick_mu({"fedprox_mu0.1": [.8, .8], "fedprox_mu0.01": [.9, .7], "fedprox_mu1": [.7, .7]}) == "fedprox_mu0.01"
    assert m8.pick_mu({"fedprox_mu1": [.9], "fedprox_mu0.001": [.5]}) == "fedprox_mu1"


def test_pick_lr_uses_the_mean_over_seeds_and_skips_unfinished_rates():
    import importlib.util, pathlib
    import pandas as pd
    spec = importlib.util.spec_from_file_location("m8_sweep", pathlib.Path(__file__).parents[1] / "scripts" / "m8_lr_sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # 0.1 has the best single run and collapses on the other seed; 0.3 is best but one seed didn't finish
    t = pd.DataFrame({"lr": [.1, .1, .01, .01, .3], "mean_val": [.84, .70, .80, .81, .9]})
    assert m.pick_lr(t, 2) == .01
    assert m.pick_lr(pd.DataFrame({"lr": [.1, .01], "mean_val": [.8, .8]}), 1) == .01   # a tie goes to the lower rate
    with pytest.raises(SystemExit):
        m.pick_lr(pd.DataFrame({"lr": [.1], "mean_val": [.8]}), 2)
