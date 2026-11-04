import numpy as np
import pandas as pd
import pytest

from ids.eval.drift_metrics import (current_series, forgetting_by_class, forgetting_by_window, summarize_recovery,
                                    time_to_recover, window_average)
from ids.eval.harness import Evaluator, TestSet, predict_by_client, predict_global
from ids.eval.metrics import detection_metrics
from ids.eval.system import fl_round_bytes, latency_summary, llm_calls_per_1k, payload_bytes, tree_cost
from ids.models.edge import EdgeConfig, EdgeEnsemble, measure_component_latency

NAMES = ["benign", "ddos", "dos"]
FAST = {"n_estimators": 20, "max_depth": 3}
FAST_LGB = {"n_estimators": 20, "num_leaves": 7, "max_depth": 3}


def _matrix():
    # t\w   0     1     2
    return pd.DataFrame([[1.0, 0.4, 0.3], [0.9, 0.8, 0.5], [0.7, 0.75, 0.95]], index=[0, 1, 2], columns=[0, 1, 2])


def test_current_series_and_average():
    s = current_series(_matrix())
    assert s.tolist() == [1.0, 0.8, 0.95]
    assert np.isclose(window_average(s, skip=(0,)), 0.875)
    static = _matrix().loc[[0]]                      # one row: the same model at every window
    assert current_series(static).tolist() == [1.0, 0.4, 0.3]


def test_forgetting_by_window():
    f = forgetting_by_window(_matrix())
    assert np.isclose(f[0], 0.3) and np.isclose(f[1], 0.05)   # 1.0-0.7, 0.8-0.75
    assert 2 not in f                                         # seen by a single model
    assert forgetting_by_window(_matrix().loc[[0]]) == {}


def test_forgetting_by_class_uses_intro_window_and_can_be_negative():
    pc = {(0, 0): {"a": 0.9}, (1, 0): {"a": 0.8}, (2, 0): {"a": 0.5},
          (0, 1): {"a": 0.1}, (1, 1): {"a": 0.6, "b": 0.4}, (2, 1): {"a": 0.7, "b": 0.7}}
    f = forgetting_by_class(pc)
    assert np.isclose(f["a"], 0.4)      # window 0 is class a's task: 0.9 -> 0.5
    assert np.isclose(f["b"], -0.3)     # improved after its introduction


def test_time_to_recover():
    t = time_to_recover(_matrix(), eps=0.02)
    assert t[1] == {"pre": 1.0, "cold": 0.4, "recovered_after": None}     # never back within 0.02 of 1.0
    assert t[2]["pre"] == 0.8 and t[2]["cold"] == 0.5 and t[2]["recovered_after"] == 1
    s = summarize_recovery(t)
    assert s == {"n_drops": 2, "n_recovered": 1, "n_never": 1, "mean_updates_to_recover": 1.0}


def test_time_to_recover_counts_later_windows_and_no_drop():
    M = pd.DataFrame([[0.9, 0.89, 0.2], [0.9, 0.9, 0.5], [0.9, 0.9, 0.6]], index=[0, 1, 2], columns=[0, 1, 2])
    t = time_to_recover(M, eps=0.02)
    assert t[1]["recovered_after"] == 0                                   # no real drop
    assert t[2]["recovered_after"] is None
    # a model that only updates later is measured until that update has happened
    M = pd.DataFrame([[0.9, 0.5, 0.9], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]], index=[0, 1, 2], columns=[0, 1, 2]).loc[[0, 2]]
    assert time_to_recover(M, eps=0.02)[1]["recovered_after"] == 2


def test_static_model_never_recovers_even_if_later_window_scores_high():
    M = pd.DataFrame([[0.9, 0.2, 0.95]], index=[0], columns=[0, 1, 2])
    assert time_to_recover(M, eps=0.02)[1]["recovered_after"] is None


def test_macro_auc_ovr():
    y = np.array([0, 0, 1, 1, 2, 2])
    perfect = np.eye(3)[y] * 0.9 + 0.1 / 3
    m = detection_metrics(y, perfect, NAMES)
    assert m["macro_auc_ovr"] == 1.0 and m["auroc_attack_vs_benign"] == 1.0
    assert "macro_auc_ovr" not in detection_metrics(np.zeros(4, int), np.eye(3)[[0, 0, 1, 2]], NAMES)


class _Fixed:
    """A model that always predicts one class, to make routing visible."""
    def __init__(self, c):
        self.c = c

    def predict_components(self, X):
        p = np.eye(3)[np.full(len(X), self.c)]
        return {"fused": p * 0.5 + 1 / 6, "calibrated": p}


def _test_sets():
    y = np.array([0, 0, 1, 1, 2, 2])
    ts = TestSet(np.zeros((6, 2), np.float32), y, np.array([0, 1, 0, 1, 0, 1]), np.array([0, 1, 0, 0, 0, 0]))
    return {0: ts, 1: ts}


def test_predict_by_client_routes_rows_and_rejects_unknown_clients():
    X, client = np.zeros((4, 1)), np.array([0, 1, 1, 0])
    p = predict_by_client({0: _Fixed(0), 1: _Fixed(2)})(X, client)["calibrated"]
    assert p.argmax(1).tolist() == [0, 2, 2, 0]
    with pytest.raises(ValueError):
        predict_by_client({0: _Fixed(0)})(X, client)


def test_evaluator_records_slices_alias_and_tables():
    ev = Evaluator(_test_sets(), NAMES, clusters={0: 0, 1: 1})
    ev.evaluate("m", 0, predict_global(_Fixed(1)))
    ev.evaluate("m", 1, predict_global(_Fixed(0)))
    ev.alias("static", "m", 0)
    allr = ev.frame("all")
    assert len(allr[allr.model == "m"]) == 8                   # 2 states x 2 windows x 2 variants
    assert set(ev.frame(None)["slice"]) == {"all", "csv_only", "cluster:0", "cluster:1"}
    assert ev.frame("csv_only").n.iloc[0] == 5                 # the one profiling row is dropped
    assert set(ev.frame("cluster:0").variant) == {"calibrated"}    # slices use the primary variant only
    M = ev.matrix("m")
    assert M.shape == (2, 2) and np.isclose(M.loc[1, 1], ev.primary("m").query("train_through == 1 and eval_window == 1").macro_f1.iloc[0])
    assert ev.matrix("static").index.tolist() == [0]
    assert ev.confusions[("m", 0, 0, "all")].sum() == 6 and ev.confusions[("static", 0, 1, "all")].sum() == 6
    s = ev.summary("m", "all", eps=0.02)
    assert s["window_avg_macro_f1"] == float(np.mean(list(s["macro_f1_by_window"].values())))
    assert ev.summary("static", "all")["mean_forgetting_by_class"] is None
    assert {"model", "slice", "macro_f1", "macro_auc_ovr"} <= set(ev.long_table().columns)
    assert set(ev.per_class_table()["class"]) <= set(NAMES)
    assert ev.confusion_table()["n"].gt(0).all()


def test_slices_absent_without_source_or_clusters():
    ts = TestSet(np.zeros((4, 1), np.float32), np.array([0, 0, 1, 1]), np.zeros(4, int))
    ev = Evaluator({0: ts}, NAMES)
    ev.evaluate("m", 0, predict_global(_Fixed(0)))
    assert set(ev.frame(None)["slice"]) == {"all"}


def _edge(classes=(0, 2, 5), n_classes=6):
    r = np.random.default_rng(0)
    y = r.choice(classes, 2000)
    X = r.normal(size=(2000, 5)).astype(np.float32)
    X[:, 0] += y * 1.5
    m = EdgeEnsemble(EdgeConfig(n_classes=n_classes, xgb_params=FAST, lgb_params=FAST_LGB, n_jobs=2))
    return m.fit(X[:1600], y[:1600], X[1600:], y[1600:]), X


@pytest.mark.parametrize("classes,n_classes", [((0, 2, 5), 6), ((0, 1), 2)])
def test_saved_model_predicts_identically(tmp_path, classes, n_classes):
    m, X = _edge(classes, n_classes)
    m.save(tmp_path / "edge")
    again = EdgeEnsemble.load(tmp_path / "edge", n_jobs=2)
    before, after = m.predict_components(X), again.predict_components(X)
    for k in before:
        assert np.allclose(before[k], after[k], atol=1e-10), k
    assert again.tree_counts() == m.tree_counts()


def test_component_latency_and_tree_cost():
    m, X = _edge()
    lat = measure_component_latency(m, X, threads=1, n_single=5, batch=100)
    assert set(lat) == {"xgb", "lgb", "ensemble"} and all(v["batch_us_per_row"] > 0 for v in lat.values())
    assert m.lgb_threads_ is None                          # thread pin is undone afterwards
    c = tree_cost(m, 3, 3)
    assert c["xgb_trees"] > 0 and c["max_comparisons_per_flow"] == 3 * (c["xgb_trees"] + c["lgb_trees"])


def test_system_accounting():
    s = latency_summary([1, 2, 3, 4])
    assert s["n"] == 4 and s["mean_ms"] == 2.5
    assert llm_calls_per_1k(5, 2000) == 2.5
    b = fl_round_bytes(100, 6, 15, metadata_bytes_per_client=10)
    assert b == {"per_client_bytes": 2410, "per_round_bytes": 36150}
    assert payload_bytes({"a": 1}) == len('{"a":1}')
