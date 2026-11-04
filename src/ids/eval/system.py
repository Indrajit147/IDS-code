"""System-cost accounting shared by the edge, federation and LLM milestones."""
from __future__ import annotations

import json

import numpy as np


def latency_summary(times_ms) -> dict:
    t = np.asarray(times_ms, dtype=float)
    return {"n": int(t.size), "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95))}


def llm_calls_per_1k(n_calls: int, n_samples: int) -> float:
    return 1000.0 * n_calls / n_samples


def fl_round_bytes(n_rows: int, n_classes: int, n_clients: int, value_bytes: int = 4,
                   metadata_bytes_per_client: int = 0) -> dict:
    """Uplink cost of sharing soft labels on D_pub for one round (float32 by default)."""
    per_client = n_rows * n_classes * value_bytes + metadata_bytes_per_client
    return {"per_client_bytes": per_client, "per_round_bytes": per_client * n_clients}


def payload_bytes(payload) -> int:
    """Size of an escalated sample's evidence as compact UTF-8 JSON."""
    return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def tree_cost(model, xgb_depth: int, lgb_depth: int) -> dict:
    """Trees used at inference and an upper bound on comparisons per flow (trees x depth
    limit), the proposal's stand-in for FLOPs."""
    n = model.tree_counts()
    return {"xgb_trees": n["xgb"], "lgb_trees": n["lgb"],
            "max_comparisons_per_flow": n["xgb"] * xgb_depth + n["lgb"] * lgb_depth}
