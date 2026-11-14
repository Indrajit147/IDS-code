import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from ids.eval.stats import client_draws, holm, macro_f1_cm, paired_bootstrap, series_score


def _counts(r, err, S=2, P=3, C=6, K=4, n=200):
    """Random per-client counts; err is the share of rows given a wrong label."""
    out = np.zeros((S, P, C, K, K))
    for idx in np.ndindex(S, P, C):
        y = r.integers(0, K - 1, n)          # class K-1 never occurs, so it must not count
        p = np.where(r.random(n) < err, r.integers(0, K, n), y)
        out[idx] = confusion_matrix(y, p, labels=np.arange(K))
    return out


def test_macro_f1_from_counts_matches_sklearn():
    r = np.random.default_rng(0)
    y, p = r.integers(0, 3, 500), r.integers(0, 5, 500)     # predicted classes absent from y
    cm = confusion_matrix(y, p, labels=np.arange(6))
    assert np.isclose(macro_f1_cm(cm), f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0))


def test_first_draw_is_the_observed_score():
    r = np.random.default_rng(1)
    c = _counts(r, 0.2)
    w = client_draws(c.shape[2], 5, seed=0)
    pooled = c.sum(2)                                        # all clients once
    assert np.isclose(series_score(c, w)[0], macro_f1_cm(pooled).mean())


def test_bootstrap_finds_a_real_gap_and_not_a_null_one():
    r = np.random.default_rng(2)
    good, bad = _counts(r, 0.05), _counts(r, 0.4)
    res = paired_bootstrap(good, bad, n=500)
    assert res["diff"] > 0.2 and res["lo"] > 0 and res["p"] < 0.01
    same = paired_bootstrap(good, good, n=200)
    assert same["diff"] == 0 and same["lo"] == 0 == same["hi"]


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
