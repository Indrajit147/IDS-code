"""One evaluator for every experiment: runs a predictor over the test windows and records
detection metrics, confusion matrices and the drift-robustness summary in a single format.

A method is registered with evaluate(method, train_through, predictor). A predictor maps
(X, client ids) to {variant: probabilities}; the last variant is the primary one, which also gets
the slice-level metrics. Use predict_global for one shared model and predict_by_client for
per-client models, so centralized and federated methods go through identical code.

Slices of each window's test set:
  all           every row
  csv_only      rows from the main CSV capture, without the Zenodo benign profiling rows
  cluster:<k>   rows of the clients in label-distribution cluster k
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from ..partition.build import SRC_CSV
from ..partition.io import load_ble_rows, load_rows, partition_dir
from .drift_metrics import (current_series, forgetting_by_class, forgetting_by_window, summarize_recovery,
                            time_to_recover, window_average)
from .metrics import confusion, detection_metrics

Predictor = Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]]
METRICS = ["n", "accuracy", "macro_f1", "weighted_f1", "ece", "auroc_attack_vs_benign", "macro_auc_ovr"]


@dataclass
class TestSet:
    __test__ = False                      # not a pytest class, despite the name
    X: np.ndarray
    y: np.ndarray
    client: np.ndarray
    source: np.ndarray | None = None      # None = no csv_only slice (e.g. BLE)


def load_test_sets(cfg: dict, pool: str, timeline: str, target: str, feats: list[str]) -> dict[int, TestSet]:
    out = {}
    for w in range(len(cfg["timeline"]["windows"])):
        d = load_rows(cfg, pool, timeline, windows=[w], split="test", features=feats)
        out[w] = TestSet(d[feats].to_numpy(np.float32), d[target].to_numpy(), d["client"].to_numpy(), d["source"].to_numpy())
    return out


def load_ble_test_sets(cfg: dict, pool: str, timeline: str, feats: list[str], first_window: int) -> dict[int, TestSet]:
    out = {}
    for w in range(first_window, len(cfg["timeline"]["windows"])):
        d = load_ble_rows(cfg, pool, timeline, windows=[w], split="test")
        out[w] = TestSet(d[feats].to_numpy(np.float32), d["y_binary"].to_numpy(), d["client"].to_numpy())
    return out


def predict_global(model) -> Predictor:
    return lambda X, client: model.predict_components(X)


def predict_by_client(models: dict) -> Predictor:
    """Each row is scored by its own client's model."""
    def predict(X: np.ndarray, client: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for c, model in models.items():
            mask = client == c
            if not mask.any():
                continue
            for variant, p in model.predict_components(X[mask]).items():
                if variant not in out:
                    out[variant] = np.zeros((len(X), p.shape[1]))
                out[variant][mask] = p
        missing = set(np.unique(client)) - set(models)
        if missing:
            raise ValueError(f"no model for clients {sorted(missing)}")
        return out
    return predict


def client_clusters(cfg: dict, pool: str, timeline: str, n_clusters: int) -> dict[int, int]:
    """Clients grouped by training label mix (Ward linkage on 19-class shares over all windows).
    Deterministic; cluster ids are ordered by their smallest client id."""
    c = pd.read_csv(partition_dir(cfg, pool, timeline) / "counts.csv")
    t = c[c["split"] == 0].pivot_table(index="client", columns="y_attack19", values="n", aggfunc="sum", fill_value=0)
    shares = t.div(t.sum(1), axis=0)
    raw = fcluster(linkage(shares.to_numpy(), method="ward"), n_clusters, criterion="maxclust")
    order = {k: i for i, k in enumerate(dict.fromkeys(raw))}
    return {int(cl): order[k] for cl, k in zip(shares.index, raw)}


class Evaluator:
    def __init__(self, test_sets: dict[int, TestSet], class_names: list[str], clusters: dict[int, int] | None = None):
        self.test_sets, self.names, self.clusters = test_sets, class_names, clusters or {}
        self.records: list[dict] = []
        self.confusions: dict[tuple, np.ndarray] = {}     # (method, t, w, slice) -> matrix

    def _slices(self, ts: TestSet) -> dict[str, np.ndarray]:
        out = {}
        if ts.source is not None and (ts.source != SRC_CSV).any():
            out["csv_only"] = ts.source == SRC_CSV
        for k in sorted(set(self.clusters.values())):
            out[f"cluster:{k}"] = np.isin(ts.client, [c for c, v in self.clusters.items() if v == k])
        return out

    def _record(self, method, t, w, variant, slice_name, y, p, extra=None) -> None:
        met = detection_metrics(y, p, self.names)
        pc = met.pop("per_class")
        self.records.append({"model": method, "train_through": t, "eval_window": w, "variant": variant,
                             "slice": slice_name, **met, "per_class_f1": {c: d["f1"] for c, d in pc.items()},
                             "per_class": pc, **(extra or {})})

    def evaluate(self, method: str, train_through: int, predictor: Predictor, extra: dict | None = None) -> None:
        """Score one trained state on every test window."""
        for w, ts in self.test_sets.items():
            comps = predictor(ts.X, ts.client)
            primary = list(comps)[-1]
            for variant, p in comps.items():
                self._record(method, train_through, w, variant, "all", ts.y, p, extra if variant == primary else None)
            p = comps[primary]
            self.confusions[(method, train_through, w, "all")] = confusion(ts.y, p, len(self.names))
            for name, mask in self._slices(ts).items():
                if mask.any():
                    self._record(method, train_through, w, primary, name, ts.y[mask], p[mask])
                    self.confusions[(method, train_through, w, name)] = confusion(ts.y[mask], p[mask], len(self.names))

    def alias(self, new_method: str, method: str, train_through: int) -> None:
        """Register one trained state of a method under a second name, without re-predicting."""
        self.records += [{**r, "model": new_method} for r in self.records
                         if r["model"] == method and r["train_through"] == train_through]
        for (m, t, w, s), v in list(self.confusions.items()):
            if m == method and t == train_through:
                self.confusions[(new_method, t, w, s)] = v

    def frame(self, slice_name: str | None = "all") -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        return df if slice_name is None else df[df["slice"] == slice_name].reset_index(drop=True)

    def primary(self, method: str, slice_name: str = "all") -> pd.DataFrame:
        """Records of a method's primary variant (the one every slice is computed for)."""
        df = self.frame(slice_name)
        df = df[df["model"] == method]
        return df[df["variant"] == df["variant"].iloc[-1]] if slice_name == "all" else df

    def matrix(self, method: str, slice_name: str = "all", metric: str = "macro_f1") -> pd.DataFrame:
        d = self.primary(method, slice_name)
        return d.pivot(index="train_through", columns="eval_window", values=metric)

    def summary(self, method: str, slice_name: str = "all", eps: float = 0.02, skip_windows=(0,)) -> dict:
        M = self.matrix(method, slice_name)
        series = current_series(M)
        pc = {(r.train_through, r.eval_window): r.per_class_f1 for r in self.primary(method, slice_name).itertuples()}
        ttr = time_to_recover(M, eps)
        f_win, f_cls = forgetting_by_window(M), forgetting_by_class(pc)
        return {"method": method, "slice": slice_name,
                "macro_f1_by_window": {int(k): float(v) for k, v in series.items()},
                "window_avg_macro_f1": window_average(series),
                "window_avg_macro_f1_excl_t0": window_average(series, skip=skip_windows),
                "final_macro_f1": float(series.iloc[-1]),
                "drop_from_own_peak": float(series.max() - series.iloc[-1]),
                "forgetting_by_window": f_win, "forgetting_by_class": f_cls,
                "mean_forgetting_by_class": float(np.mean(list(f_cls.values()))) if f_cls else None,
                "time_to_recover": ttr, "recovery": summarize_recovery(ttr)}

    def long_table(self) -> pd.DataFrame:
        cols = ["model", "train_through", "eval_window", "variant", "slice", *METRICS]
        return self.frame(None).reindex(columns=cols)

    def per_class_table(self) -> pd.DataFrame:
        rows = [{"model": r["model"], "train_through": r["train_through"], "eval_window": r["eval_window"],
                 "variant": r["variant"], "slice": r["slice"], "class": c, **d}
                for r in self.records for c, d in r["per_class"].items()]
        return pd.DataFrame(rows)

    def confusion_table(self) -> pd.DataFrame:
        """Non-zero cells only, so the tracked report stays small."""
        rows = []
        for (m, t, w, s), cm in self.confusions.items():
            for i, j in zip(*np.nonzero(cm)):
                rows.append({"model": m, "train_through": t, "eval_window": w, "slice": s,
                             "true": self.names[i], "pred": self.names[j], "n": int(cm[i, j])})
        return pd.DataFrame(rows)
