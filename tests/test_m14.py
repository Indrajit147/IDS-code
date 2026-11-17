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


def test_false_positive_rate_and_custom_stat():
    from ids.eval.stats import false_positive_rate_cm
    cm = np.array([[8, 2, 0], [1, 5, 0], [0, 0, 3]])          # 2 of 10 benign rows flagged
    assert np.isclose(false_positive_rate_cm(cm), 0.2)
    assert false_positive_rate_cm(np.zeros((3, 3))) == 0
    r = np.random.default_rng(3)
    c = _counts(r, 0.2)
    res = paired_bootstrap(c, c, n=50, stat=lambda a, w: series_score(a, w, false_positive_rate_cm))
    assert res["diff"] == 0


def test_h4c_paired_items_and_exact_mcnemar():
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m14_h4c import ACTION, mcnemar_exact, paired
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0
    assert abs(mcnemar_exact(0, 6) - 2 / 64) < 1e-12                      # all six discordant pairs one way
    df = pd.DataFrame({"row": [1, 1, 2, 2, 3, 3], "arm": ["shap", "shap_llm"] * 3, "x": [1, 0, 1, 1, 0, 1]})
    r = paired(df, "x")
    assert (r["pairs"], r["llm_only_yes"], r["shap_only_yes"]) == (3, 1, 1) and r["diff"] == 0
    assert ACTION.search("Rate-limit the source") and ACTION.search("isolate the device")
    assert not ACTION.search("Further investigation is recommended; monitor Number")


def test_m15_check_compare_tolerance_and_missing_cells():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m15_check import cells, compare
    summ = {"summaries": [{"method": "e3", "slice": "csv_only", "window_avg_macro_f1": 0.9, "macro_f1_by_window": {"0": 1.0, "1": 0.8}},
                          {"method": "e3", "slice": "all", "window_avg_macro_f1": 0.1, "macro_f1_by_window": {"0": 0.1}}]}
    assert cells(summ, "e3") == {"window_avg": 0.9, "t0": 1.0, "t1": 0.8} and cells(summ, "e4") == {}
    r = {x["cell"]: x["ok"] for x in compare({"window_avg": 0.9049, "t0": 0.994}, cells(summ, "e3"))}
    assert r == {"window_avg": True, "t0": False, "t1": False}          # 0.0049 passes, 0.006 fails, a missing cell fails
