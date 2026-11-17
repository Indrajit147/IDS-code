# Released artifacts

The configuration and logs the proposal says to release, written by `python scripts/m15_release.py`. [`MANIFEST.json`](MANIFEST.json) gives each file's SHA-256 and size, and the commit they were written at.

| file | what it is |
|---|---|
| `experiment_config.json` | The full resolved configuration of the headline runs: Kaggle paths, seed 42, 4 threads. Other seeds change `seed` only. |
| `model_hparams.json` | The parts of that configuration that set the models and their training: edge ensemble, incremental updates, distillation, FedAvg, the E4 schedule, drift detection, the gate, the LLM verifier, the policy agent and SHAP. |
| `client_partitions.json` | The partition's settings and outcome: 15 clients, Dirichlet α 0.5, seed, test share, duplicates dropped, device groups, D_pub rows per class, and the clients that drift at t5. The row-by-row assignment isn't released: `scripts/m2_build_partitions.py` rebuilds it deterministically from these settings, and Notebook A's score check would catch a different one. |
| `window_definition.json` | The six windows t0–t5 and the classes each one introduces. |
| `llm_logs/*.jsonl.gz` | Every LLM prompt and reply, one JSON object per attempt: `prompt`, `output`, `attempt` (0 = first try), `valid` (met the schema), `error` (why not). M11 verifier: Gemma 2 2B at seeds 42–44, and Mistral 7B on a subset of seed 42. M12 policy agent: Gemma 2 2B. |
| `dataset/` | The card for the processed data and `SHA256SUMS`, which the notebooks check the attached data against. |

Read a log with `gzip.open(path, "rt")` and one `json.loads` per line.
