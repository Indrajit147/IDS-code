"""Detection metrics. Macro scores use the classes PRESENT IN y_true, so a model is
not rewarded or penalised for classes that do not occur in the evaluated window;
predicting a class that is absent still costs precision of the true classes."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error."""
    conf, pred = p.max(1), p.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def detection_metrics(y: np.ndarray, p: np.ndarray, class_names: list[str]) -> dict:
    pred = p.argmax(1)
    labels = np.unique(y)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "ece": ece(p, y),
        "per_class": {class_names[c]: {"precision": float(a), "recall": float(b), "f1": float(f), "support": int(s)}
                      for c, a, b, f, s in zip(labels, prec, rec, f1, sup)},
    }
    # binary view (benign = class 0 vs any attack): AUROC of P(attack)
    yb = (y != 0).astype(int)
    if 0 < yb.sum() < len(yb):
        out["auroc_attack_vs_benign"] = float(roc_auc_score(yb, 1 - p[:, 0]))
    aucs = [roc_auc_score(y == c, p[:, c]) for c in labels if 0 < (y == c).sum() < len(y)]
    if aucs:
        out["macro_auc_ovr"] = float(np.mean(aucs))
    return out


def confusion(y: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y, p.argmax(1), labels=np.arange(n_classes))
