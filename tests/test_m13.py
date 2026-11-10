import numpy as np
import xgboost as xgb

from ids.fl.fedgbdt import federated_train, global_balanced_weights, histogram_upload_bytes
from ids.models.edge import balanced_weights

P = {"objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "max_depth": 3, "nthread": 1, "seed": 0, "eval_metric": "mlogloss"}


def part(seed, n=600):
    g = np.random.default_rng(seed)
    X = g.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1)
    return X, y


def test_federated_matches_pooled_and_continues():
    data = [part(s) for s in range(3)]
    parts = [(X, y, np.ones(len(y)), X[:100], y[:100], np.ones(100)) for X, y in data]
    m = federated_train(parts, P, 5, early_stopping=None)      # raises if the clients' models differ
    b = xgb.Booster(model_file=bytearray(m))
    assert b.num_boosted_rounds() == 5
    X, y = np.concatenate([d[0] for d in data]), np.concatenate([d[1] for d in data])
    pooled = xgb.train(P, xgb.DMatrix(X, y), 5)
    Xt = part(9)[0]
    agree = (b.predict(xgb.DMatrix(Xt)).argmax(1) == pooled.predict(xgb.DMatrix(Xt)).argmax(1)).mean()
    assert agree > 0.95, agree       # only the merged quantile sketch differs from pooled training
    b2 = xgb.Booster(model_file=bytearray(federated_train(parts, P, 3, prev=m)))
    assert b2.num_boosted_rounds() == 8
    assert histogram_upload_bytes(b2, 5 * 3, 4, 256) > 0


def test_global_weights_match_m3_on_pooled_labels():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    w = global_balanced_weights(np.bincount(y, minlength=4), cap=50.0)
    assert np.allclose(w[y], balanced_weights(y, 50.0)) and w[3] == 0
