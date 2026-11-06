import json

import numpy as np
import pytest

from ids.utils import JsonlWriter, Run, derive_seed, load_config, read_jsonl, rng
from ids.utils.config import parse_overrides


def test_config_layers_and_overrides():
    cfg = load_config(env="local", overrides=["gate.tau_conf=0.65", "seed=7"])
    assert cfg["env"] == "local"
    assert cfg["seed"] == 7
    assert cfg["gate"]["tau_conf"] == 0.65
    assert cfg["gate"]["tau_disagree"] == 0.20          # untouched default survives
    assert cfg["resources"]["device"] == "cpu"          # from env/local.yaml


def test_kaggle_profile_paths():
    cfg = load_config(env="kaggle")
    assert cfg["paths"]["data_root"] == "/kaggle/input"
    assert cfg["resources"]["use_dev_sample"] is False


def test_parse_overrides_rejects_bad_input():
    assert parse_overrides(["a.b.c=[1, 2]"]) == {"a": {"b": {"c": [1, 2]}}}
    with pytest.raises(ValueError):
        parse_overrides(["no_equals_sign"])


def test_derived_seeds_stable_and_distinct():
    assert derive_seed(42, "client", 3, "t2") == derive_seed(42, "client", 3, "t2")
    assert derive_seed(42, "client", 3) != derive_seed(42, "client", 4)
    a = rng(42, "w", 0).integers(0, 1_000_000, 5)
    b = rng(42, "w", 0).integers(0, 1_000_000, 5)
    assert np.array_equal(a, b)


def test_jsonl_append_only(tmp_path):
    w = JsonlWriter(tmp_path / "log.jsonl")
    w.write({"x": np.float32(1.5), "arr": np.arange(3)})
    w.write({"x": 2})
    rows = list(read_jsonl(tmp_path / "log.jsonl"))
    assert [r["x"] for r in rows] == [1.5, 2]
    assert rows[0]["arr"] == [0, 1, 2] and "ts" in rows[0]


def test_run_dir_snapshots(tmp_path):
    cfg = load_config(env="local", overrides={"paths": {"results_root": str(tmp_path)}})
    with Run(cfg, "unit") as run:
        run.log("hello", value=1)
    cfg_snap = json.loads((run.dir / "experiment_config.json").read_text(encoding="utf-8"))
    env_snap = json.loads((run.dir / "environment.json").read_text(encoding="utf-8"))
    events = [e["event"] for e in read_jsonl(run.dir / "events.jsonl")]
    assert cfg_snap["seed"] == cfg["seed"]
    assert env_snap["packages"]["numpy"]
    assert events == ["run_start", "hello", "run_end"]


def test_memory_note_reports_this_process():
    from ids.utils.run import memory_note
    note = memory_note()
    assert "rss" in note and "GB" in note
