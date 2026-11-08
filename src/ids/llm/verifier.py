"""M11: the cloud LLM verifier (inference time only). See README, M11, for the design.

payload -> prompt -> backend text -> validated reply (retry with the error, then fall back) -> D3 policy.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

KEYS = ("final_label", "override_edge", "confidence_bucket", "reasons", "feature_recommendations", "safety_notes")
BUCKETS = ("low", "medium", "high")

INSTRUCTIONS = """You verify one decision of a network intrusion detector at an IoT medical gateway.
The edge model labelled a network flow. Check whether its label is consistent with the evidence:
the probabilities, the top features by SHAP (positive values pushed towards the edge's label),
the median value of those features for each known class (class_medians), the share of each class
among this client's recent flows (recent_class_share), and the drift state.
Possible labels: {classes}.

Evidence:
{payload}

Answer with one JSON object and nothing else, with exactly these keys:
"final_label": one of the possible labels,
"override_edge": true only if final_label differs from the edge's label,
"confidence_bucket": "low", "medium" or "high",
"reasons": 1 to 3 short statements, each naming a feature or probability from the evidence,
"feature_recommendations": 0 to 5 feature names from the evidence worth watching,
"safety_notes": a short string, empty if none."""

STRICT = """

Your previous answer was rejected: {error}
Reply with the JSON object only: no code fences, no text before or after it."""


def prompt(payload: dict, classes: list[str], error: str | None = None) -> str:
    text = INSTRUCTIONS.format(classes=", ".join(classes), payload=json.dumps(payload, separators=(",", ":")))
    return text + (STRICT.format(error=error) if error else "")


def _json_object(text: str):
    """The first balanced {...} in the text, parsed; code fences are tolerated."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_reply(text: str, edge_label: str, classes: list[str], features: list[str]) -> tuple[dict | None, str | None]:
    """(reply, None) if it meets the contract, else (None, what is wrong)."""
    obj = _json_object(text)
    if not isinstance(obj, dict):
        return None, "no JSON object found"
    missing = [k for k in KEYS if k not in obj]
    if missing:
        return None, f"missing keys {missing}"
    extra = sorted(set(obj) - set(KEYS))
    if extra:
        return None, f"unexpected keys {extra}"
    lab = obj["final_label"]
    if lab not in classes:
        return None, f"final_label {lab!r} is not one of {classes}"
    if not isinstance(obj["override_edge"], bool):
        return None, "override_edge must be true or false"
    if obj["override_edge"] != (lab != edge_label):
        return None, f"override_edge must be {str(lab != edge_label).lower()} when final_label is {lab!r} and the edge's label is {edge_label!r}"
    if obj["confidence_bucket"] not in BUCKETS:
        return None, f"confidence_bucket must be one of {list(BUCKETS)}"
    r = obj["reasons"]
    if not (isinstance(r, list) and 1 <= len(r) <= 3 and all(isinstance(s, str) and s.strip() for s in r)):
        return None, "reasons must be a list of 1 to 3 non-empty strings"
    fr = obj["feature_recommendations"]
    if not (isinstance(fr, list) and len(fr) <= 5 and all(isinstance(s, str) for s in fr)):
        return None, "feature_recommendations must be a list of at most 5 strings"
    bad = [s for s in fr if s not in features]
    if bad:
        return None, f"feature_recommendations {bad} are not model features"
    if not isinstance(obj["safety_notes"], str):
        return None, "safety_notes must be a string"
    return obj, None


def fallback(edge_label: str) -> dict:
    """The proposal's fallback: no override, SHAP-only explanation."""
    return {"final_label": edge_label, "override_edge": False, "confidence_bucket": "low", "reasons": ["fallback: SHAP only"],
            "feature_recommendations": [], "safety_notes": "LLM reply failed validation"}


def grounded(reply: dict, payload: dict) -> bool:
    """Every reason names a feature or class that is in the payload: a check the reply is tied to the evidence."""
    names = [f["feature"] for f in payload["top_features"]] + list(payload["probs"])
    return all(any(re.search(rf"\b{re.escape(n)}\b", s, re.IGNORECASE) for n in names) for s in reply["reasons"])


def d3_allows(reply: dict, booster_disagree: bool, drift_flag: bool) -> bool:
    """D3: the LLM may replace the edge's class only with high disagreement and a drift flag."""
    return bool(reply["override_edge"]) and booster_disagree and drift_flag


class StubBackend:
    """Rule-based stand-in for tests and the dev run, not an LLM: the known class whose medians are nearest on the top features."""
    name = "stub"

    def generate(self, prompts: list[str]) -> list[str]:
        out = []
        for p in prompts:
            ev = json.loads(p.split("Evidence:\n", 1)[1].split("\n\nAnswer with", 1)[0])
            dist = {}
            for c in ev["known_classes"]:
                d = [abs(np.log1p(abs(f["value"])) - np.log1p(abs(ev["class_medians"][f["feature"]][c]))) for f in ev["top_features"]]
                dist[c] = float(np.mean(d))
            lab = min(sorted(dist), key=dist.get) if dist else ev["edge_label"]
            top = ev["top_features"][0]["feature"] if ev["top_features"] else "probs"
            out.append(json.dumps({"final_label": lab, "override_edge": lab != ev["edge_label"], "confidence_bucket": "medium",
                                   "reasons": [f"{top} is closest to the {lab} median"], "feature_recommendations": [top] if ev["top_features"] else [],
                                   "safety_notes": ""}))
        return out


class HFBackend:
    """transformers on a GPU; 4-bit or 8-bit bitsandbytes as the proposal plans. Greedy decoding."""

    def __init__(self, model_id: str, quantization: str = "4bit", compute_dtype: str = "float16", max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        dt = getattr(torch, compute_dtype)
        q = {"4bit": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt),
             "8bit": BitsAndBytesConfig(load_in_8bit=True), "none": None}[quantization]
        self.torch, self.name, self.max_new_tokens = torch, model_id, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=q, dtype=dt, device_map="auto")
        self.model.eval()

    def generate(self, prompts: list[str]) -> list[str]:
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def make_backend(lc: dict):
    if lc["backend"] == "stub":
        return StubBackend()
    if lc["backend"] == "hf":
        return HFBackend(lc["model_id"], lc["quantization"], lc["compute_dtype"], lc["max_new_tokens"])
    raise ValueError(f"unknown llm.backend {lc['backend']!r}")


def verify(items: list[dict], backend, classes: list[str], features: list[str], batch_size: int, max_retries: int, log=None) -> tuple[list[dict], list[dict]]:
    """items: {"key", "payload"}. Returns one record per item (reply, attempts, valid, first_try_valid) and one per batch (seconds, rows)."""
    recs = [{"key": it["key"], "reply": None, "attempts": 0, "valid": False, "first_try_valid": False, "error": None} for it in items]
    batches, pending = [], list(range(len(items)))
    for attempt in range(max_retries + 1):
        for s in range(0, len(pending), batch_size):
            chunk = pending[s:s + batch_size]
            prompts = [prompt(items[i]["payload"], classes, recs[i]["error"] if attempt else None) for i in chunk]
            tic = time.perf_counter()
            texts = backend.generate(prompts)
            sec = time.perf_counter() - tic
            batches.append({"attempt": attempt, "rows": len(chunk), "seconds": sec})
            for i, p, text in zip(chunk, prompts, texts):
                reply, err = parse_reply(text, items[i]["payload"]["edge_label"], classes, features)
                r = recs[i]
                r["attempts"], r["error"] = attempt + 1, err
                if reply is not None:
                    r["reply"], r["valid"], r["first_try_valid"] = reply, True, attempt == 0
                if log is not None:
                    log.write({"key": items[i]["key"], "attempt": attempt, "prompt": p, "output": text, "valid": reply is not None,
                               "error": err, "batch_seconds": round(sec, 4), "batch_rows": len(chunk)})
        pending = [i for i in pending if not recs[i]["valid"]]
        if not pending:
            break
    for i in pending:
        recs[i]["reply"] = fallback(items[i]["payload"]["edge_label"])
    return recs, batches
