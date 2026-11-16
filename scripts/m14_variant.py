"""M14 tier 2 (and the D_pub ablation): one sensitivity variant, one seed, as a list of pipeline steps.

Each variant reruns E1 (M3 + M4), E3 with local + rehearsal (M7; its local pass is exactly M6's) and E4
(M9's K sweep and e4_added, so K is chosen on the variant's own validation rows). Reports go to
reports/mX/<pool>/<variant>/seed<seed>/; the headline reports are untouched.

    python scripts/m14_variant.py --variant clients10 --pool dev          # run every step here
    python scripts/m14_variant.py --variant clients10 --pool full --list  # print the steps (the Kaggle notebook runs them)
"""
from __future__ import annotations

import argparse
import shlex
import subprocess

# variant -> (overrides for every step, extra overrides for the partition build only)
VARIANTS = {
    "dpub_benign": ("partition.variant=dpub_benign dpub.benign_only=true", ""),
    "clients10": ("partition.variant=clients10 federation.n_clients=10", ""),
    "clients20": ("partition.variant=clients20 federation.n_clients=20", ""),
    "alpha0.1": ("partition.variant=alpha0.1 federation.dirichlet_alpha=0.1", ""),
    "alpha1.0": ("partition.variant=alpha1.0 federation.dirichlet_alpha=1.0", ""),
    "alt_order": ("timeline.headline=alt_order", ""),                       # M2 already builds this timeline
    "reseed": ("partition.variant=reseed", "seed=7"),                       # same settings, partition drawn at seed 7
    "with_iat": ("'features.exclude_wifi=[]'", ""),                         # D6
    "no_number_weight": ("'features.exclude_wifi=[IAT,Number,Weight]'", ""),  # dataset fact 8
    "no_profiling": ("partition.variant=no_profiling profiling.use=false", ""),  # D7
    "official_split": ("partition.variant=official_split timeline.split=official", ""),
}
TIER2 = [v for v in VARIANTS if v != "dpub_benign"]


def commands(variant: str, pool: str, seed: int, env: str | None) -> list[str]:
    ov, m2_extra = VARIANTS[variant]
    e = f"--env {env} " if env else ""
    tag = f"{variant}/seed{seed}"
    args = f"{e}--pool {pool} --set resources.n_jobs=4 seed={seed} {ov} --tag {tag}"
    py = "python -X faulthandler"
    out = [f"{py} scripts/m2_build_partitions.py {e}--pools {pool} --set {ov} {m2_extra}".rstrip()]
    if "partition.variant" not in ov:            # a headline-partition rebuild redraws the headline M2 figures: put them back
        out.append(f"git checkout -- reports/m2/{pool}")
    out += [f"{py} scripts/m3_baselines.py {args}",
            f"{py} scripts/m4_evaluate.py {args}",         # latest M3 run for the pool: the one just made
            f"{py} scripts/m7_distill.py {args} --only e3 --resume",
            f"{py} scripts/m7_distill.py {args} --report"]
    out += [f"{py} scripts/m9_e4.py {args} --only {p} --resume" for p in ("fixed_k10", "fixed_k20", "fixed_k30", "e4_added")]
    out.append(f"{py} scripts/m9_e4.py {args} --report")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--pool", default="dev")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cmds = commands(a.variant, a.pool, a.seed, a.env)
    if a.list:
        print("\n".join(cmds))
        return
    for c in cmds:
        print("=====", c, flush=True)
        rc = subprocess.run(shlex.split(c)).returncode   # no shell: the same on Windows and Kaggle
        print("exit code:", rc, flush=True)
        if rc:
            raise SystemExit(f"{a.variant}: step failed: {c}")


if __name__ == "__main__":
    main()
