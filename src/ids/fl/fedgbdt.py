"""M13 / E7: horizontal federated XGBoost by histogram aggregation, relaxed privacy (README, M13).

Each client is one XGBoost worker process holding only its own rows. XGBoost's collective (Rabit) merges the
clients' quantile sketches into shared bin cuts and sums their gradient/Hessian histograms at every node, so all
workers grow the same trees; the validation metric for early stopping is summed the same way. These summaries
travel unencrypted (the relaxed privacy). The model each worker ends with is compared byte for byte.
"""
from __future__ import annotations

import multiprocessing as mp
import queue

import numpy as np
import xgboost as xgb
from xgboost import collective
from xgboost.tracker import RabitTracker

HIST_VALUE_BYTES = 8     # one gradient or Hessian sum, double precision


def _worker(args, part, params, rounds, prev, early_stopping, q) -> None:
    X, y, w, Xv, yv, wv = part
    with collective.CommunicatorContext(**args):
        b = xgb.train(params, xgb.DMatrix(X, y, weight=w), rounds, evals=[(xgb.DMatrix(Xv, yv, weight=wv), "val")],
                      early_stopping_rounds=early_stopping, xgb_model=None if prev is None else xgb.Booster(model_file=bytearray(prev)),
                      verbose_eval=False)
        if early_stopping:
            b = b[: b.best_iteration + 1]
        q.put((collective.get_rank(), bytes(b.save_raw("ubj"))))


def federated_train(parts: list[tuple], params: dict, rounds: int, prev: bytes | None = None, early_stopping: int | None = None) -> bytes:
    """parts: one (X, y, w, Xv, yv, wv) per client. Returns the shared booster as UBJSON bytes."""
    ctx = mp.get_context("spawn")      # the same on Windows and Kaggle; no inherited threads in the workers
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=len(parts))
    tracker.start()
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(tracker.worker_args(), p, params, rounds, prev, early_stopping, q)) for p in parts]
    for p in procs:
        p.start()
    out: dict[int, bytes] = {}
    while len(out) < len(procs):        # read before join: a large model in the queue blocks its worker's exit
        try:
            r, m = q.get(timeout=10)
            out[r] = m
        except queue.Empty:
            dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
            if dead:
                for p in procs:
                    p.kill()
                raise RuntimeError(f"a federated worker failed (exit codes {dead})")
    for p in procs:
        p.join()
    tracker.wait_for()
    if len(set(out.values())) != 1:
        raise RuntimeError("the clients ended with different models")
    return out[0]


def global_balanced_weights(counts: np.ndarray, cap: float) -> np.ndarray:
    """M3's balanced class weights n / (k * n_c), from the summed class counts of all clients. counts: per class."""
    present = counts > 0
    w = np.zeros(len(counts))
    w[present] = np.minimum(counts.sum() / (present.sum() * counts[present]), cap)
    return w


def histogram_upload_bytes(booster: xgb.Booster, first_tree: int, n_features: int, max_bin: int) -> int:
    """Upper bound on one client's uplink for the trees from first_tree on: one gradient and one Hessian histogram per node."""
    df = booster.trees_to_dataframe()
    nodes = int((df.Tree >= first_tree).sum())
    return nodes * n_features * max_bin * 2 * HIST_VALUE_BYTES
