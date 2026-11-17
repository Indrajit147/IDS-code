"""M15: the released artifacts the proposal lists, gathered into release/ (tracked in git).

  experiment_config.json   the resolved headline config (Kaggle environment, seed 42, 4 threads)
  model_hparams.json       the config sections that set models and training
  client_partitions.json, window_definition.json   M2's full pool, headline timeline
  llm_logs/*.jsonl.gz      every LLM prompt and reply: M11 seeds 42-44 (Gemma 2 2B), the Mistral subset, M12
  dataset/SHA256SUMS       SHA-256 of the four processed Parquet files (what the public Kaggle dataset must hold)
  MANIFEST.json            SHA-256 and size of every file above, and the commit they came from

Run: python scripts/m15_release.py. Inputs that are gitignored (the M11 logs, data/processed/) must exist locally.
A missing LLM log is skipped and listed in MANIFEST.json, so a later seed can be added by running it again.
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess

from ids.data.ingest import sha256_file
from ids.utils.config import REPO_ROOT, load_config
from ids.utils.jsonl import dumps

OUT = REPO_ROOT / "release"
HPARAM_SECTIONS = ["models", "incremental", "distill", "fedavg", "schedule", "drift", "gate", "llm", "policy", "explain"]
PROCESSED = ["wifi_mqtt_full.parquet", "wifi_mqtt_dev.parquet", "ble.parquet", "wifi_profiling.parquet"]
LOGS = {  # released name -> source under reports/
    "m11_gemma2-2b_seed42.jsonl": "m11/full/llm_log.jsonl",
    "m11_gemma2-2b_seed43.jsonl": "m11/full/seed43/llm_log.jsonl",
    "m11_gemma2-2b_seed44.jsonl": "m11/full/seed44/llm_log.jsonl",
    "m11_mistral-7b_subset_seed42.jsonl": "m11/full/mistral/llm_log.jsonl",
    "m12_gemma2-2b_seed42.jsonl": "m12/full/llm_log_llm.jsonl",
    "m12_gemma2-2b_seed43.jsonl": "m12/full/seed43/llm_log_llm.jsonl",
    "m12_gemma2-2b_seed44.jsonl": "m12/full/seed44/llm_log_llm.jsonl",
}


def gzip_fixed(src, dst) -> None:
    """mtime 0 and no file name in the header, so the same log always gives the same bytes (and SHA-256)."""
    with open(src, "rb") as f, open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as g:
        shutil.copyfileobj(f, g)


def main() -> None:
    cfg = load_config(env="kaggle", overrides=["resources.n_jobs=4", "seed=42"])
    (OUT / "llm_logs").mkdir(parents=True, exist_ok=True)
    (OUT / "dataset").mkdir(parents=True, exist_ok=True)
    (OUT / "experiment_config.json").write_text(dumps(cfg, indent=2), encoding="utf-8", newline="\n")
    (OUT / "model_hparams.json").write_text(dumps({k: cfg[k] for k in HPARAM_SECTIONS}, indent=2), encoding="utf-8", newline="\n")
    for name in ("client_partitions.json", "window_definition.json"):
        text = (REPO_ROOT / "reports" / "m2" / "full" / name).read_text(encoding="utf-8")   # as text: a Windows checkout has CRLF, git stores LF
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")
    missing = []
    for name, src in LOGS.items():
        f = REPO_ROOT / "reports" / src
        if f.exists():
            gzip_fixed(f, OUT / "llm_logs" / f"{name}.gz")
        else:
            missing.append(src)
    proc = REPO_ROOT / "data" / "processed"
    (OUT / "dataset" / "SHA256SUMS").write_text("".join(f"{sha256_file(proc / n)}  {n}\n" for n in PROCESSED), encoding="utf-8", newline="\n")
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in ("MANIFEST.json", "README.md"))
    manifest = {"commit": commit, "missing_logs": missing,
                "files": {p.relative_to(OUT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files}}
    (OUT / "MANIFEST.json").write_text(dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    print(f"release -> {OUT}: {len(files)} files; missing logs: {missing or 'none'}")


if __name__ == "__main__":
    main()
