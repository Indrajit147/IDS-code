import json

import numpy as np

from ids.fl.rehearsal import RehearsalBuffer
from ids.llm.policy import DEFAULT, HeuristicBackend, decide, heuristic, parse_policy, prompt

EV = {"adwin_fired": True, "fired_at_round_share": 0.3, "rows_new": 900, "error_rate_new_rows": 0.2, "error_rate_validation": 0.05,
      "new_classes": [], "class_share_new_rows": {"dos": 1.0}, "class_share_buffer": {"dos": 1.0}, "per_class_f1_validation": {"dos": 0.9},
      "feature_shift_share": 0.3, "most_shifted_features": ["Rate", "syn_count", "AVG"], "shap_drift": 0.1,
      "top_shap_features_new_rows": ["Rate"], "previous_policy": None}


class Scripted:
    def __init__(self, answers):
        self.answers = list(answers)

    def generate(self, prompts):
        return [self.answers.pop(0) for _ in prompts]


def test_contract_and_heuristic_round_trip():
    pol = heuristic(EV)
    assert pol["update_urgency"] == "medium" and pol["buffer_selection"] == "boundary" and pol["feature_focus_groups"] == ["tcp_flags", "rate_timing"]
    obj, err = parse_policy(HeuristicBackend().generate([prompt(EV)])[0])
    assert err is None and obj == pol
    for bad, why in [({**pol, "update_scope": "half"}, "update_scope"), ({**pol, "feature_focus_groups": ["x"]}, "feature_focus_groups"),
                     ({**pol, "preserve_old_behavior": "yes"}, "true or false"), ({**pol, "reasons": []}, "reasons"), ({**pol, "extra": 1}, "unexpected")]:
        obj, err = parse_policy(json.dumps(bad))
        assert obj is None and why in err, err


def test_decide_retries_then_defaults():
    r = decide(EV, Scripted(["nope", json.dumps(heuristic(EV))]), max_retries=2)
    assert r["valid"] and r["attempts"] == 2 and not r["first_try_valid"]
    r = decide(EV, Scripted(["nope"] * 3), max_retries=2)
    assert not r["valid"] and r["policy"] == DEFAULT


def test_buffer_keys_keep_smallest():
    b = RehearsalBuffer(2, np.random.default_rng(0))
    X, y = np.arange(4.0)[:, None], np.zeros(4, int)
    b.add(X, y, keys_in=np.array([0.9, 0.1, 0.5, 0.05]))
    assert sorted(b.get()[0].ravel().tolist()) == [1.0, 3.0]
