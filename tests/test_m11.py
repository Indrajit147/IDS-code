import json

from ids.llm.verifier import StubBackend, d3_allows, grounded, parse_reply, prompt, verify

CL = ["benign", "ddos", "dos", "recon", "mqtt", "spoofing"]
FEATS = ["Rate", "Number", "syn_count"]
PAY = {"edge_label": "dos", "probs": {c: 0.1 for c in CL}, "booster_labels": {"xgboost": "dos", "lightgbm": "ddos"}, "disagreement": 0.3,
       "top_features": [{"feature": "Rate", "value": 900.0, "shap": 1.2}, {"feature": "Number", "value": 9.5, "shap": -0.4}],
       "class_medians": {"Rate": {"dos": 1000.0, "ddos": 50.0}, "Number": {"dos": 9.5, "ddos": 1.0}},
       "recent_class_share": {c: 0.0 for c in CL}, "drift": {"flag": True, "score": 0.2}, "known_classes": ["dos", "ddos"]}


def reply(**kw):
    r = {"final_label": "dos", "override_edge": False, "confidence_bucket": "high", "reasons": ["Rate is near the dos median"],
         "feature_recommendations": ["Rate"], "safety_notes": ""}
    return {**r, **kw}


def test_contract_accepts_a_good_reply_inside_fences_and_text():
    text = "Sure:\n```json\n" + json.dumps(reply(reasons=["a {brace} in \"quotes\" about Rate"])) + "\n```"
    obj, err = parse_reply(text, "dos", CL, FEATS)
    assert err is None and obj["final_label"] == "dos"


def test_contract_rejects_each_violation():
    bad = [("not json", "no JSON"), (json.dumps(reply(final_label="worm")), "final_label"),
           (json.dumps(reply(final_label="ddos")), "override_edge must be true"),
           (json.dumps(reply(override_edge="yes")), "true or false"), (json.dumps(reply(confidence_bucket="sure")), "confidence_bucket"),
           (json.dumps(reply(reasons=[])), "reasons"), (json.dumps(reply(feature_recommendations=["IAT"])), "not model features"),
           (json.dumps({**reply(), "extra": 1}), "unexpected"), (json.dumps({k: v for k, v in reply().items() if k != "safety_notes"}), "missing")]
    for text, why in bad:
        obj, err = parse_reply(text, "dos", CL, FEATS)
        assert obj is None and why in err, (text, err)


class Scripted:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def generate(self, prompts):
        self.prompts += prompts
        return [self.answers.pop(0) for _ in prompts]


class Log:
    def __init__(self):
        self.rows = []

    def write(self, r):
        self.rows.append(r)


def test_retry_states_the_error_then_falls_back():
    items = [{"key": 1, "payload": PAY}, {"key": 2, "payload": PAY}]
    # item 1: bad then good; item 2: bad three times -> fallback
    b, log = Scripted(["nope", "nope", json.dumps(reply()), "nope", "nope"]), Log()
    recs, batches = verify(items, b, CL, FEATS, batch_size=8, max_retries=2, log=log)
    assert recs[0]["valid"] and not recs[0]["first_try_valid"] and recs[0]["attempts"] == 2
    assert not recs[1]["valid"] and recs[1]["attempts"] == 3
    assert recs[1]["reply"]["final_label"] == "dos" and recs[1]["reply"]["override_edge"] is False
    assert "previously" not in b.prompts[0] and "rejected: no JSON object found" in b.prompts[2]
    assert len(log.rows) == 5 and [r["rows"] for r in batches] == [2, 2, 1]


def test_d3_needs_override_disagreement_and_drift():
    r = reply(final_label="ddos", override_edge=True)
    assert d3_allows(r, True, True)
    assert not d3_allows(r, False, True) and not d3_allows(r, True, False) and not d3_allows(reply(), True, True)


def test_grounded_and_stub():
    assert grounded(reply(), PAY) and not grounded(reply(reasons=["looks fine"]), PAY)
    out = StubBackend().generate([prompt(PAY, CL)])[0]
    obj, err = parse_reply(out, "dos", CL, FEATS)
    assert err is None and obj["final_label"] == "dos"     # Rate 900 and Number 9.5 sit at the dos medians
