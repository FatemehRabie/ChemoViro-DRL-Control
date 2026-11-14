"""Audit of the 80 trained agents stored in ../multiseed_results.json.

Question: did the agents learn dosing policies, or are their evaluation returns those of trivial
state-independent policies?

The script
  1. checks that the Gym-free replica environment reproduces the stored open-loop baselines exactly;
  2. evaluates four constant policies - no dosing (0, 0) and the three other corners of the action box -
     on Tests 1-4 with the same episode seeds as the manuscript;
  3. compares every trained agent, seed by seed, with those constants;
  4. summarises the stored training curves (is there any change after the first 10,000 steps?);
  5. summarises the archived seed-42 action sequences.

Run:  python 01_policy_audit.py        (about 1 minute; numpy/scipy only)
Out:  results/01_policy_audit.json, results/01_policy_audit.md
"""
from __future__ import annotations

import numpy as np

from common import (ALGOS, CONSTANTS, MODELS, SEEDS, TESTS, const_policy, eval_on_test, load_multiseed,
                    md_table, rl_runs, save_json, two_stage_policy, RESULTS)

data = load_multiseed()
TEST_NAMES = list(TESTS)
short = lambda t: t.split(" (")[0]

# ---- 1. replica check -------------------------------------------------------------------------
stored_policies = {
    "Constant Half-Dose Baseline (500, 50)": const_policy(500.0, 50.0),
    "Matched Open-Loop Constant (600, 100)": const_policy(600.0, 100.0),
    "Matched Open-Loop 2-Stage": two_stage_policy,
}
max_rel, n_cmp = 0.0, 0
for b in data["baselines"]:
    if b["baseline_name"] not in stored_policies:
        continue
    for t in TEST_NAMES:
        mine = eval_on_test(stored_policies[b["baseline_name"]], t, b["seed"])
        ref = b["test_evals"][t]
        max_rel = max(max_rel, abs(mine["reward_mean"] - ref["reward_mean"]) / abs(ref["reward_mean"]))
        assert abs(mine["steps"] - ref["steps"]) < 1e-9
        n_cmp += 1
print(f"[1] replica vs stored baselines: {n_cmp} values, max relative difference {max_rel:.1e}")
assert max_rel < 1e-9, "replica environment does not reproduce the stored baselines"

# ---- 2. constant policies ---------------------------------------------------------------------
const = {name: {t: {s: eval_on_test(const_policy(*a), t, s)["reward_mean"] for s in SEEDS} for t in TEST_NAMES}
         for name, a in CONSTANTS.items()}
print("[2] constant policies, mean over the 5 seed cohorts")
rows = []
for name in CONSTANTS:
    rows.append([name] + [f"{np.mean(list(const[name][t].values())):.4e}" for t in TEST_NAMES])
    print("    ", rows[-1])
const_table = md_table(["constant policy"] + [short(t) for t in TEST_NAMES], rows)

# ---- 3. every trained agent against the constants ---------------------------------------------
ZERO = "no dosing (0, 0)"
per_run = []
for model in MODELS:
    for algo in ALGOS:
        for r in rl_runs(data, model, algo):
            s = r["seed"]
            entry = {"model": model, "algo": algo, "seed": s, "tests": {}}
            for t in TEST_NAMES:
                v = r["test_evals"][t]["reward_mean"]
                rel = {name: abs(const[name][t][s] - v) / abs(v) for name in CONSTANTS}
                nearest = min(rel, key=rel.get)
                entry["tests"][t] = {"return": v, "rel_diff_to_no_dosing": rel[ZERO], "nearest_constant": nearest,
                                     "rel_diff_to_nearest": rel[nearest], "no_dosing_return": const[ZERO][t][s]}
            names = {entry["tests"][t]["nearest_constant"] for t in TEST_NAMES}
            worst = max(entry["tests"][t]["rel_diff_to_nearest"] for t in TEST_NAMES)
            entry["exact_constant_on_all_tests"] = bool(len(names) == 1 and worst < 1e-9)
            entry["constant_name"] = names.pop() if entry["exact_constant_on_all_tests"] else None
            entry["max_rel_diff_to_no_dosing"] = max(entry["tests"][t]["rel_diff_to_no_dosing"] for t in TEST_NAMES)
            per_run.append(entry)

on_policy = [e for e in per_run if e["algo"] in ("PPO", "TRPO")]
off_policy = [e for e in per_run if e["algo"] in ("TD3", "SAC")]
# Tests 1, 2 and 4 have returns of 1e4-1e6, so a relative difference is meaningful; Test 3 returns are ~ -6.
big_tests = [t for t in TEST_NAMES if "Test 3" not in t]
worst_on = max(e["tests"][t]["rel_diff_to_no_dosing"] for e in on_policy for t in big_tests)
worst_on_abs_t3 = max(abs(e["tests"][t]["return"] - e["tests"][t]["no_dosing_return"]) for e in on_policy for t in TEST_NAMES if "Test 3" in t)
n_exact_zero = sum(e["exact_constant_on_all_tests"] and e["constant_name"] == ZERO for e in on_policy)
print(f"[3] PPO/TRPO (40 runs): largest relative difference to the no-dosing return on Tests 1, 2, 4 = {worst_on:.2e}; "
      f"largest absolute difference on Test 3 = {worst_on_abs_t3:.3f}; runs identical to no dosing to 1e-9: {n_exact_zero}")
rows = []
for model in MODELS:
    for algo in ("PPO", "TRPO"):
        es = [e for e in on_policy if e["model"] == model and e["algo"] == algo]
        rows.append([model.split(" (")[0], algo] +
                    [f"{max(e['tests'][t]['rel_diff_to_no_dosing'] for e in es):.1e}" for t in big_tests] +
                    [f"{max(abs(e['tests'][t]['return'] - e['tests'][t]['no_dosing_return']) for e in es for t in TEST_NAMES if 'Test 3' in t):.3f}"])
on_table = md_table(["trained on", "algorithm"] + [f"{short(t)}: max rel. diff. to no dosing" for t in big_tests] + ["Test 3: max abs. diff."], rows)

rows, n_exact_corner, n_worse_than_zero = [], 0, 0
for e in off_policy:
    n_exact_corner += e["exact_constant_on_all_tests"]
    t2 = "Test 2 (Reduced, Default, No Noise)"
    worse = e["tests"][t2]["return"] < e["tests"][t2]["no_dosing_return"] - 1.0
    n_worse_than_zero += worse
    rows.append([e["model"].split(" (")[0], e["algo"], e["seed"],
                 e["constant_name"] if e["exact_constant_on_all_tests"] else "not one fixed corner",
                 f"{e['tests'][t2]['return']:.0f}", "yes" if worse else "no"])
off_table = md_table(["trained on", "algorithm", "seed", "behaviour on Tests 1-4", "Test 2 return", "worse than no dosing on Test 2"], rows)
m2 = [e for e in off_policy if e["model"].startswith("Model 2")]
print(f"    TD3/SAC (40 runs): exact constant corner policy on all four tests in {n_exact_corner} runs "
      f"({sum(e['exact_constant_on_all_tests'] for e in m2)} of the 10 Model 2 runs); "
      f"worse than no dosing on Test 2 in {n_worse_than_zero} runs")

# identical TD3 rows in Models 3 and 4
same = all(abs(a["test_evals"][t]["reward_mean"] - b["test_evals"][t]["reward_mean"]) < 1e-9
           for a, b in zip(rl_runs(data, "Model 3 (Reduced, Modified, Noise)", "TD3"), rl_runs(data, "Model 4 (Reduced, Modified, No Noise)", "TD3"))
           for t in TEST_NAMES)
print(f"    Model 3 TD3 and Model 4 TD3 give identical returns for every seed and test: {same}")

# ---- 4. training curves -----------------------------------------------------------------------
def at(ts, rw, step):
    i = int(np.searchsorted(ts, step))
    return rw[min(i, len(rw) - 1)]

curve_rows, curves = [], {}
for model in MODELS:
    for algo in ALGOS:
        firsts, r10, r50, drift = [], [], [], []
        for r in rl_runs(data, model, algo):
            ts, rw = np.array(r["curve_timesteps"]), np.array(r["curve_rewards"])
            firsts.append(rw[0]); r10.append(at(ts, rw, 10_000)); r50.append(rw[-1])
            drift.append(abs(rw[-1] - at(ts, rw, 10_000)) / abs(at(ts, rw, 10_000)))
        curves[f"{model}|{algo}"] = {"first_logged": firsts, "at_10k": r10, "final": r50, "rel_change_10k_to_50k": drift}
        curve_rows.append([model.split(" (")[0], algo, f"{np.mean(firsts):.4g}", f"{np.mean(r10):.4g}", f"{np.mean(r50):.4g}",
                           f"{np.median(drift) * 100:.2f}%", f"{np.max(drift) * 100:.1f}%"])
curve_table = md_table(["trained on", "algorithm", "mean return, first logged point (~1k steps)", "at 10k steps", "at 50k steps",
                        "median |change| 10k to 50k across seeds", "max |change|"], curve_rows)
all_drift = np.array([d for v in curves.values() for d in v["rel_change_10k_to_50k"]])
print(f"[4] training curves: relative change of the logged return between 10k and 50k steps is below 1% in "
      f"{int((all_drift < 0.01).sum())} of {all_drift.size} runs and below 5% in {int((all_drift < 0.05).sum())} (median {np.median(all_drift) * 100:.2f}%)")

# ---- 4b. the one cell where the logged return moves; single runs against the half-dose baseline --
T2_NAME = "Test 2 (Reduced, Default, No Noise)"
T3_NAME = next(t for t in TEST_NAMES if "Test 3" in t)
m1sac_rows = []
for r in rl_runs(data, next(m for m in MODELS if m.startswith("Model 1")), "SAC"):
    ts, rw = np.array(r["curve_timesteps"]), np.array(r["curve_rewards"])
    m1sac_rows.append([r["seed"], f"{rw[0]:.0f}", f"{at(ts, rw, 10_000):.0f}", f"{rw[-1]:.0f}",
                       f"{r['test_evals'][T2_NAME]['reward_mean']:.0f}", f"{r['test_evals'][T3_NAME]['reward_mean']:.0f}"])
m1sac_table = md_table(["seed", "training return, first logged point", "at 10k steps", "at 50k steps", "Test 2 return", "Test 3 return"], m1sac_rows)
half_t2 = next(b for b in data["baselines"] if b["baseline_name"].startswith("Constant Half-Dose"))["test_evals"][T2_NAME]["reward_mean"]
n_below_half = int(sum(r["test_evals"][T2_NAME]["reward_mean"] < half_t2 for r in data["rl_runs"]))
print(f"    single runs below the constant half dose on Test 2 ({half_t2:.0f}): {n_below_half} of {len(data['rl_runs'])}")

# ---- 5. archived seed-42 actions --------------------------------------------------------------
act_rows = []
for model in MODELS:
    for algo in ALGOS:
        r = rl_runs(data, model, algo)[0]
        a = np.clip(np.array(r["test_evals"]["Test 1 (Reduced, Default, Noise)"]["actions_sample"], dtype=float), [0, 0], [1000, 100])
        distinct = np.unique(np.round(a, 3), axis=0)
        act_rows.append([model.split(" (")[0], algo, f"{a[:, 0].min():.3g} to {a[:, 0].max():.3g}", f"{a[:, 1].min():.3g} to {a[:, 1].max():.3g}",
                         len(distinct) if len(distinct) > 3 else "; ".join(f"({x:g}, {y:g})" for x, y in distinct)])
act_table = md_table(["trained on", "algorithm", "u1 range (max 1000)", "u2 range (max 100)", "distinct actions on Test 1 (seed 42)"], act_rows)
n_seeds_archived = len({r["seed"] for r in data["rl_runs"] if any("actions_sample" in v for v in r["test_evals"].values())})
# every archived TD3/SAC action, all four tests: is it within 0.1% of the range from a corner of the action box?
n_off_archived = n_off_all_corner = 0
for r in data["rl_runs"]:
    arch = [np.array(v["actions_sample"], dtype=float) for v in r["test_evals"].values() if "actions_sample" in v]
    if r["algo"] not in ("TD3", "SAC") or not arch:
        continue
    a = np.clip(np.concatenate(arch), [0, 0], [1000, 100])
    n_off_archived += 1
    n_off_all_corner += bool(np.all((a[:, 0] <= 1) | (a[:, 0] >= 999)) and np.all((a[:, 1] <= 0.1) | (a[:, 1] >= 99.9)))
print(f"    TD3/SAC runs with archived actions: {n_off_archived}; every archived action at a corner of the action box in {n_off_all_corner}")
print(f"[5] action sequences are archived for {n_seeds_archived} of 5 seeds")

save_json("01_policy_audit.json", {"replica_max_rel_diff": max_rel, "constants": const, "per_run": per_run, "curves": curves,
                                   "summary": {"on_policy_max_rel_diff_to_no_dosing_tests_1_2_4": worst_on,
                                               "on_policy_max_abs_diff_test_3": worst_on_abs_t3,
                                               "on_policy_runs_identical_to_no_dosing": n_exact_zero,
                                               "off_policy_runs_exact_constant_corner": n_exact_corner,
                                               "off_policy_runs_with_archived_actions": n_off_archived,
                                               "off_policy_archived_runs_with_every_action_at_a_corner": n_off_all_corner,
                                               "off_policy_runs_worse_than_no_dosing_test2": n_worse_than_zero,
                                               "model3_model4_td3_identical": same,
                                               "runs_below_constant_half_dose_on_test2": n_below_half,
                                               "runs_with_change_below_1pct_10k_to_50k": int((all_drift < 0.01).sum()),
                                               "runs_with_change_below_5pct_10k_to_50k": int((all_drift < 0.05).sum())}})

(RESULTS / "01_policy_audit.md").write_text(f"""# 01 - Audit of the trained agents

Source: `multiseed_results.json` (80 runs). Replica environment reproduces the stored open-loop baselines
with a maximum relative difference of {max_rel:.1e} over {n_cmp} values.

## Constant policies on the four tests (mean over the five seed cohorts, Regime A)

{const_table}

## PPO and TRPO against the no-dosing policy u = (0, 0)

Largest relative difference between an agent's seed-level return and the no-dosing return on the same episodes.
Test 3 returns are about -6, so the absolute difference is given there.

{on_table}

Across all 40 PPO/TRPO runs the largest relative difference on Tests 1, 2 and 4 is {worst_on:.2e};
{n_exact_zero} runs are identical to no dosing to nine digits on every test.

## TD3 and SAC against the corners of the action box

{off_table}

{n_exact_corner} of the 40 runs are an exact constant corner policy on all four tests; {n_worse_than_zero} runs score below
no dosing on Test 2. Model 3 TD3 and Model 4 TD3 return identical numbers for every seed and test: {same}.

For the other {40 - n_exact_corner} runs the stored returns show only that the policy is not one fixed corner. Actions were saved for
seed 42 alone: in all {n_off_all_corner} of those {n_off_archived} TD3/SAC runs every saved action, on all four tests, lies at a corner of the
action box (one or two distinct corners per run; see the last table).

## Training curves (mean episodic training return stored by the Monitor callback)

{curve_table}

The logged return changes by less than 1% between 10,000 and 50,000 steps in {int((all_drift < 0.01).sum())} of {all_drift.size} runs
(less than 5% in {int((all_drift < 0.05).sum())}).

Model 1 SAC is the one cell where the logged return keeps changing after 10,000 steps (median change 33%; the median in
every other cell is below 5%). No dosing returns about -4.2 in that training model and is its optimum (check 02), so a
return moving toward zero is an agent learning to stop dosing:

{m1sac_table}

Single runs against the simplest baseline: {n_below_half} of the {len(data['rl_runs'])} runs score below the constant half dose on Test 2 ({half_t2:.0f}).

## Archived seed-42 actions on Test 1

{act_table}

Action sequences are archived for {n_seeds_archived} of the 5 seeds only.
""")
print("wrote results/01_policy_audit.json and results/01_policy_audit.md")
