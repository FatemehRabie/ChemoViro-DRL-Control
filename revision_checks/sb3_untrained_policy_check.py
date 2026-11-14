"""Does training change the policies?  (NOT RUN during the review - it needs your RL environment.)

Requires the same packages as ../run_multiseed_drl.py (gymnasium, torch, stable-baselines3, sb3-contrib).

For each of the 80 (model, algorithm, seed) combinations this script calls the repository's own
`train_and_evaluate_job` with `total_steps = 0`. Stable-Baselines3 then builds the network exactly as in
the benchmark (same seed, architecture and hyperparameters), performs no environment step and no gradient
update, and the repository's own evaluation code scores that freshly initialised policy on Tests 1-4.
The returns are compared with the trained agents' returns stored in ../multiseed_results.json.

If the untrained and trained returns agree, the 50,000 training steps did not change the policy's behaviour.

Run (from this folder):
    python sb3_untrained_policy_check.py                 # all 80, one process, roughly 10-20 minutes
    python sb3_untrained_policy_check.py --workers 8     # in parallel
    python sb3_untrained_policy_check.py --models 2 --algos PPO TD3
Out:  results/sb3_untrained_policy_check.json and a summary on screen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import run_multiseed_drl as R  # noqa: E402

TRAINING_SCENARIOS = [
    {"name": "Model 1 (Full, Default, Noise)", "ode_model": "full", "params": "default", "noise": "default"},
    {"name": "Model 2 (Reduced, Default, Noise)", "ode_model": "reduced", "params": "default", "noise": "default"},
    {"name": "Model 3 (Reduced, Modified, Noise)", "ode_model": "reduced", "params": "modified", "noise": "default"},
    {"name": "Model 4 (Reduced, Modified, No Noise)", "ode_model": "reduced", "params": "modified", "noise": "none"},
]


def build_jobs(models, algos):
    jobs = []
    for scen in TRAINING_SCENARIOS:
        if models and not any(scen["name"].startswith(f"Model {m} ") for m in models):
            continue
        for algo in algos:
            for seed in R.SEEDS:
                jobs.append({"scenario_name": scen["name"], "ode_model": scen["ode_model"], "params": scen["params"],
                             "noise": scen["noise"], "algo": algo, "seed": seed, "reward_weights": (1e-2, 1e-1),
                             "total_steps": 0})          # <- the only difference from the benchmark
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--models", type=int, nargs="*", default=[], help="e.g. --models 2 3 (default: all four)")
    ap.add_argument("--algos", nargs="*", default=["PPO", "TRPO", "TD3", "SAC"])
    args = ap.parse_args()

    with open(REPO / "multiseed_results.json") as f:
        stored = {(r["scenario_name"], r["algo"], r["seed"]): r for r in json.load(f)["rl_runs"]}

    jobs = build_jobs(args.models, args.algos)
    print(f"Evaluating {len(jobs)} untrained models (total_steps = 0) ...")
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            outs = pool.map(R.train_and_evaluate_job, jobs)
    else:
        outs = []
        for i, job in enumerate(jobs, 1):
            outs.append(R.train_and_evaluate_job(job))
            print(f"  {i}/{len(jobs)}  {job['scenario_name'].split(' (')[0]} {job['algo']} seed {job['seed']}", flush=True)

    rows, summary = [], {}
    for out in outs:
        key = (out["scenario_name"], out["algo"], out["seed"])
        ref = stored[key]
        rel = {}
        for t, ev in out["test_evals"].items():
            trained = ref["test_evals"][t]["reward_mean"]
            rel[t] = abs(ev["reward_mean"] - trained) / abs(trained)
        worst = max(rel.values())
        rows.append({"model": key[0], "algo": key[1], "seed": key[2], "max_rel_diff_untrained_vs_trained": worst,
                     "untrained": {t: ev["reward_mean"] for t, ev in out["test_evals"].items()},
                     "trained": {t: ref["test_evals"][t]["reward_mean"] for t in out["test_evals"]}})
        summary.setdefault((key[0].split(" (")[0], key[1]), []).append(worst)

    print("\nLargest relative difference between untrained and trained returns over Tests 1-4, per seed:")
    print(f"{'model':8s} {'algo':5s} " + " ".join(f"{'seed ' + str(s):>10s}" for s in R.SEEDS) + "   identical (<1e-6) / within 1%")
    for (m, a), vals in summary.items():
        print(f"{m:8s} {a:5s} " + " ".join(f"{v:10.2e}" for v in vals) + f"   {sum(v < 1e-6 for v in vals)} / {sum(v < 1e-2 for v in vals)} of {len(vals)}")
    allv = np.array([r["max_rel_diff_untrained_vs_trained"] for r in rows])
    print(f"\nOverall: {int((allv < 1e-6).sum())} of {allv.size} runs are identical before and after training; "
          f"{int((allv < 1e-2).sum())} are within 1%.")

    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "sb3_untrained_policy_check.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote results/sb3_untrained_policy_check.json")


if __name__ == "__main__":
    main()
