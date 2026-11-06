"""FedAvg and FedProx on a small MLP: the classic weight-averaging references (E2).

Clients train on their own rows and send weights; the cloud averages them by row count. FedProx adds
(mu / 2) * ||w - w_global||^2 to each client's loss, and mu = 0 is exactly FedAvg.
Inputs are signed-log scaled because several features are heavy tailed (Rate reaches 2e6), and the scaling
statistics are built from per-client totals, so no rows leave a client.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from ..models.edge import fit_temperature
from ..utils import rng


def _signed_log(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


class Scaler:
    """Signed log, then z-score, then clip. A client sends its count, sum and sum of squares per feature."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, clip: float = 10.0):
        self.mean, self.std, self.clip = mean, std, clip

    @classmethod
    def from_clients(cls, parts: list[np.ndarray], clip: float = 10.0) -> "Scaler":
        n, s, ss = 0, 0.0, 0.0
        for X in parts:
            L = _signed_log(X.astype(np.float64))
            n, s, ss = n + len(L), s + L.sum(0), ss + (L * L).sum(0)
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean * mean, 0.0))
        return cls(mean, np.where(std > 1e-9, std, 1.0), clip)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip((_signed_log(X.astype(np.float64)) - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)


class TabularMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: tuple, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers, last = [], n_features
            for h in hidden:
                layers += [nn.Linear(last, h), nn.ReLU()]
                last = h
            self.net = nn.Sequential(*layers, nn.Linear(last, n_classes))
        self.temperature = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, X: np.ndarray, batch: int = 65536) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(X)
            return np.concatenate([torch.softmax(self.net(t[i:i + batch]) / self.temperature, 1).numpy() for i in range(0, len(t), batch)])

    def predict_components(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {"global": self.predict_proba(X)}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        return 4 * self.n_params()


@dataclass
class ClientData:
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    Xv: np.ndarray
    yv: np.ndarray


def _clone(state: dict) -> dict:
    return {k: v.detach().clone() for k, v in state.items()}


def fedavg_aggregate(states: list[dict], n_rows: list[int]) -> dict:
    total = float(sum(n_rows))
    return {k: sum(s[k] * (n / total) for s, n in zip(states, n_rows)) for k in states[0]}


def local_update(model: TabularMLP, global_state: dict, X: np.ndarray, y: np.ndarray, w: np.ndarray, *, mu: float, lr: float,
                 momentum: float, batch: int, epochs: int, rng: np.random.Generator) -> dict:
    """One client's work in one round. model is scratch space; global_state is read and not changed."""
    model.load_state_dict(global_state)
    anchor = [p.detach().clone() for p in model.parameters()] if mu else []
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    Xt, yt, wt = torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(w)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(len(y)))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            opt.zero_grad(set_to_none=True)
            wb = wt[idx]
            loss = (loss_fn(model(Xt[idx]), yt[idx]) * wb).sum() / wb.sum()
            loss.backward()
            for p, a in zip(model.parameters(), anchor):
                p.grad += mu * (p.detach() - a)
            opt.step()
    return _clone(model.state_dict())


def run_rounds(model: TabularMLP, clients: list[ClientData], *, rounds: int, patience: int, mu: float, lr: float, momentum: float,
               batch: int, epochs: int, rng_key: tuple) -> dict:
    """Federated rounds on one window. Keeps the round with the best macro-F1 on the clients' pooled validation rows."""
    Xv, yv = np.concatenate([c.Xv for c in clients]), np.concatenate([c.yv for c in clients])
    labels = np.unique(yv)
    n_rows = [len(c.y) for c in clients]
    state = _clone(model.state_dict())
    best, best_state, best_round, curve = -1.0, state, -1, []
    for r in range(rounds):
        parts = [local_update(model, state, c.X, c.y, c.w, mu=mu, lr=lr, momentum=momentum, batch=batch, epochs=epochs, rng=rng(*rng_key, r, i))
                 for i, c in enumerate(clients)]
        state = fedavg_aggregate(parts, n_rows)
        model.load_state_dict(state)
        model.temperature = 1.0
        score = float(f1_score(yv, model.predict_proba(Xv).argmax(1), labels=labels, average="macro", zero_division=0))
        curve.append(score)
        if score > best:
            best, best_state, best_round = score, state, r
        elif r - best_round >= patience:
            break
    model.load_state_dict(best_state)
    model.temperature = 1.0
    model.temperature = float(fit_temperature(model.predict_proba(Xv), yv))
    return {"rounds_run": len(curve), "best_round": best_round, "best_val": best, "val_curve": curve, "temperature": model.temperature}
