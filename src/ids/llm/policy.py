"""M12: the LLM adaptation-policy agent (training-time decisions only). See README, M12, for the design.

evidence -> prompt -> backend text -> validated policy (retry with the error, then the default policy) -> training knobs.
The LLM never sees or changes the model, the data or the features: it picks how the next update is run.
"""
from __future__ import annotations

import json

from .verifier import _json_object

GROUPS = {
    "tcp_flags": ["fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number",
                  "cwr_flag_number", "ack_count", "syn_count", "fin_count", "rst_count"],
    "protocol": ["Protocol_Type", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC"],
    "rate_timing": ["Duration", "Rate", "Srate"],
    "packet_size": ["Header_Length", "Tot_sum", "Min", "Max", "AVG", "Std", "Tot_size", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"],
}
CHOICES = {"update_urgency": ("low", "medium", "high"), "training_style": ("conservative", "standard", "aggressive"),
           "buffer_selection": ("uniform", "boundary"), "update_scope": ("partial", "full")}
KEYS = ("update_urgency", "training_style", "preserve_old_behavior", "feature_focus_groups", "buffer_selection", "update_scope", "reasons")
DEFAULT = {"update_urgency": "medium", "training_style": "standard", "preserve_old_behavior": False, "feature_focus_groups": [],
           "buffer_selection": "uniform", "update_scope": "partial", "reasons": ["default policy"]}

# The knobs each field sets (README, M12). These multipliers are assumptions; the proposal names the direction only.
STYLE = {"conservative": (0.5, 0.5), "standard": (1.0, 1.0), "aggressive": (1.5, 1.5)}    # (learning-rate scale, new-tree scale)
PRESERVE_WEIGHT = 2.0       # weight on rehearsal-buffer rows when preserve_old_behavior is true
FOCUS_WEIGHT = 1.5          # feature weight for the focus groups' features, in both boosters

INSTRUCTIONS = """You decide how one client of a federated network intrusion detector should run its next model update.
You do not train the model and you do not see it. You choose the training policy from the evidence below:
whether ADWIN detected a rise in the model's error on the new rows, the error rates, new classes, class shares,
the per-class F1 on validation rows, feature shift (PSI), and SHAP drift (how differently the model explains the
new rows than its rehearsal buffer). Updating at the wrong time or too aggressively causes forgetting of old attacks.
Feature groups: {groups}.

Evidence:
{evidence}

Answer with one JSON object and nothing else, with exactly these keys:
"update_urgency": "low" (delay: skip this update, keep the rows for later), "medium" or "high" (update now),
"training_style": "conservative" (lower learning rate, fewer new trees), "standard" or "aggressive",
"preserve_old_behavior": true to give the rehearsal buffer's old rows more weight, else false,
"feature_focus_groups": 0 to 2 group names from the feature groups, whose features get more weight,
"buffer_selection": "uniform" (random rows per class) or "boundary" (keep the rows the model is least sure about),
"update_scope": "partial" (add trees to the model) or "full" (refit from scratch on the new rows and the buffer),
"reasons": 1 to 3 short statements, each naming a value from the evidence."""

STRICT = """

Your previous answer was rejected: {error}
Reply with the JSON object only: no code fences, no text before or after it."""


def prompt(evidence: dict, error: str | None = None) -> str:
    text = INSTRUCTIONS.format(groups=", ".join(GROUPS), evidence=json.dumps(evidence, separators=(",", ":")))
    return text + (STRICT.format(error=error) if error else "")


def parse_policy(text: str) -> tuple[dict | None, str | None]:
    """(policy, None) if it meets the contract, else (None, what is wrong)."""
    obj = _json_object(text)
    if not isinstance(obj, dict):
        return None, "no JSON object found"
    missing = [k for k in KEYS if k not in obj]
    if missing:
        return None, f"missing keys {missing}"
    extra = sorted(set(obj) - set(KEYS))
    if extra:
        return None, f"unexpected keys {extra}"
    for k, ok in CHOICES.items():
        if obj[k] not in ok:
            return None, f"{k} must be one of {list(ok)}"
    if not isinstance(obj["preserve_old_behavior"], bool):
        return None, "preserve_old_behavior must be true or false"
    g = obj["feature_focus_groups"]
    if not (isinstance(g, list) and len(g) <= 2 and all(x in GROUPS for x in g) and len(set(g)) == len(g)):
        return None, f"feature_focus_groups must list 0 to 2 distinct names from {list(GROUPS)}"
    r = obj["reasons"]
    if not (isinstance(r, list) and 1 <= len(r) <= 3 and all(isinstance(s, str) and s.strip() for s in r)):
        return None, "reasons must be a list of 1 to 3 non-empty strings"
    return obj, None


def heuristic(ev: dict) -> dict:
    """The fixed rule-based policy, on the same evidence. Thresholds fixed before any M12 result (README, M12)."""
    new_cls, fired = bool(ev["new_classes"]), ev["adwin_fired"]
    rise = ev["error_rate_new_rows"] - ev["error_rate_validation"]
    urgency = "high" if new_cls else "medium" if fired or rise > 0.02 or ev["feature_shift_share"] >= 0.25 else "low"
    style = "aggressive" if new_cls else "standard" if fired else "conservative"
    top = ev["most_shifted_features"][:3]
    focus = [g for g in GROUPS if any(f in GROUPS[g] for f in top)][:2] if ev["feature_shift_share"] >= 0.25 else []
    return {"update_urgency": urgency, "training_style": style, "preserve_old_behavior": new_cls or fired, "feature_focus_groups": focus,
            "buffer_selection": "boundary" if fired else "uniform", "update_scope": "full" if ev["shap_drift"] > 0.5 and not new_cls else "partial",
            "reasons": [f"new_classes {ev['new_classes']}, adwin_fired {fired}, error rise {rise:.3f}, shap_drift {ev['shap_drift']:.2f}"]}


class HeuristicBackend:
    """The fixed policy as reply text, so the LLM path (prompt, parse, log) can be checked without an LLM."""
    name = "heuristic"

    def generate(self, prompts: list[str]) -> list[str]:
        return [json.dumps(heuristic(json.loads(p.split("Evidence:\n", 1)[1].split("\n\nAnswer with", 1)[0]))) for p in prompts]


def decide(evidence: dict, backend, max_retries: int, log=None, key=None) -> dict:
    """One policy: the reply if valid within max_retries retries, else DEFAULT. Returns {policy, valid, attempts, first_try_valid}."""
    err = None
    for attempt in range(max_retries + 1):
        p = prompt(evidence, err)
        text = backend.generate([p])[0]
        pol, err = parse_policy(text)
        if log is not None:
            log.write({"key": key, "attempt": attempt, "prompt": p, "output": text, "valid": pol is not None, "error": err})
        if pol is not None:
            return {"policy": pol, "valid": True, "attempts": attempt + 1, "first_try_valid": attempt == 0}
    return {"policy": dict(DEFAULT), "valid": False, "attempts": max_retries + 1, "first_try_valid": False}
