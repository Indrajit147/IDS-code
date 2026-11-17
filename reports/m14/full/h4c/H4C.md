# H4 (c): SHAP-only against SHAP + LLM explanations (M11 seed 42, 100 rows)

Not preregistered: rubric fixed on 2026-11-17, after M11's reasons were read. The prompt never asked Gemma for operator actions. See the docstring of `scripts/m14_h4c.py` for the rubric.

| item | pairs | SHAP only | SHAP + LLM | difference | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| label_correct | 100 | 0.70 | 0.45 | -0.25 | 0.00102 |
| action_lexicon | 100 | 0.00 | 0.00 | +0.00 | 1 |

Action list over every valid reply of seed 42 (not just the sample): 0 of 14066 (0.0000).

Scorer items done: none yet (fill sheet.csv with y or n and rerun). `action_lexicon` is a keyword proxy, not the rubric.
