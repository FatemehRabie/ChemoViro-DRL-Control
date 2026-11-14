"""Numerically optimal dosing schedules: the reference the benchmark is missing.

For each environment the script solves the daily piecewise-constant optimal control problem by direct
single shooting (see ocp.py) and scores the result in the exact evaluation environment. It then reports

  a. the optimum next to no dosing, the constant schedules and the two-stage schedule, per environment;
  b. the optimality gap of every method in the manuscript on the deterministic Test 2;
  c. how much any feedback policy could gain over one fixed schedule on Test 1 (5% initial-condition noise)
     and on the manuscript's ten perturbed parameter sets, by re-optimising the schedule for each episode
     with full knowledge of its initial state / parameters (a bound no feedback policy can beat);
  d. that the (U, I, V, C) dynamics of Test 4 are those of the 4D Modified model.

Run:  python 02_optimal_control.py     (about 5 minutes on 2 cores; numpy/scipy only)
Out:  results/02_optimal_control.json, results/02_optimal_control.md
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import minimize

from common import (ALGOS, CONSTANTS, Env, NOISE, NOMINAL_IC, ODE_PARAMS, REGIME_A, REGIME_B, REPO, RESULTS, SEEDS,
                    const_policy, load_multiseed, md_table, rl_runs, run_episode, sampled_parameter_sets, save_json,
                    seed_means, two_stage_policy)
from ocp import default_starts, exact_score, make_objective, schedule_policy, solve

t_start = time.time()
rng = np.random.default_rng(0)
out = {}

REFERENCE_POLICIES = {"no dosing (0, 0)": const_policy(0.0, 0.0), "half dose (500, 50)": const_policy(500.0, 50.0),
                      "best constant in the paper (600, 100)": const_policy(600.0, 100.0),
                      "virus only (1000, 0)": const_policy(1000.0, 0.0), "drug only (0, 100)": const_policy(0.0, 100.0),
                      "full dose (1000, 100)": const_policy(1000.0, 100.0), "two-stage (1000,100) then (0,100) at day 10": two_stage_policy}

# ---- a. optimum per environment ---------------------------------------------------------------
ENVS = [
    ("4D Default (Test 2; training model of Model 2)", "reduced", "default", REGIME_A, 60),
    ("4D Default, Regime B weights", "reduced", "default", REGIME_B, 60),
    ("4D Modified (training model of Models 3 and 4)", "reduced", "modified", REGIME_A, 30),
    ("6D Modified, zero lysis (Test 4)", "full", "modified_no_lysis", REGIME_A, 30),
    ("6D Modified with lysis (Test 3)", "full", "modified", REGIME_A, 6),
    ("6D Default with lysis (training model of Model 1)", "full", "default", REGIME_A, 6),
]
md_sections, schedules = [], {}
for label, model, pkey, w, N in ENVS:
    p = ODE_PARAMS[pkey]
    cost, x, per_start = solve(NOMINAL_IC[model], p, model == "full", w, N, default_starts(N, rng))
    ev = exact_score(x, model, p, w, record=True)
    tum = ev["traj"][:, 0] + ev["traj"][:, 1]
    rows = [["numerical optimum", f"{ev['reward']:.1f}", "0.00%", ev["steps"], f"{ev['tumor_cost']:.0f}", f"{ev['dose_cost']:.0f}", f"{tum.max():.0f}"]]
    refs = {}
    for name, pol in REFERENCE_POLICIES.items():
        r = run_episode(Env(model, p, (0.0, 0.0), w), pol, record=True)
        t_ = r["traj"][:, 0] + r["traj"][:, 1]
        refs[name] = {"return": r["reward"], "steps": r["steps"], "tumor_cost": r["tumor_cost"], "dose_cost": r["dose_cost"], "peak": float(t_.max()),
                      "gap_pct": 100.0 * (ev["reward"] - r["reward"]) / abs(ev["reward"])}
        rows.append([name, f"{r['reward']:.1f}", f"{refs[name]['gap_pct']:.2f}%", r["steps"], f"{r['tumor_cost']:.0f}", f"{r['dose_cost']:.0f}", f"{t_.max():.0f}"])
    spread = (max(per_start.values()) - min(per_start.values())) / abs(min(per_start.values()))
    u1, u2 = x[:N] * 1000.0, x[N:] * 100.0
    out[label] = {"optimum_return": ev["reward"], "surrogate_cost": cost, "steps": ev["steps"], "tumor_cost": ev["tumor_cost"], "dose_cost": ev["dose_cost"],
                  "peak_burden": float(tum.max()), "per_start_surrogate_costs": per_start, "relative_spread_across_starts": spread,
                  "u1": u1, "u2": u2, "references": refs, "weights": w}
    schedules[label] = x
    print(f"[a] {label}: optimum {ev['reward']:.1f} in {ev['steps']} d; no dosing {refs['no dosing (0, 0)']['return']:.1f} "
          f"(gap {refs['no dosing (0, 0)']['gap_pct']:.2f}%); starts agree to {spread:.1e}")
    md_sections.append(f"### {label}\n\nWeights (A1, A2) = {w}. Optimum found from {len(per_start)} starting schedules; their final costs agree to {spread:.1e} (relative).\n\n"
                       + md_table(["schedule", "return", "gap to optimum", "days to U+I<0.1", "tumour cost", "dose cost", "peak U+I"], rows)
                       + f"\n\nOptimal u1 by day (first {min(N, 24)}): " + ", ".join(f"{v:.0f}" for v in u1[:24])
                       + f"\n\nOptimal u2 by day (first {min(N, 32)}): " + ", ".join(f"{v:.0f}" for v in u2[:32]) + "\n")

xA = schedules["4D Default (Test 2; training model of Model 2)"]
opt2 = out["4D Default (Test 2; training model of Model 2)"]["optimum_return"]
np.save(RESULTS / "optimal_schedule_test2_regimeA.npy", xA)

# ---- b. optimality gap of the manuscript's methods on Test 2 ----------------------------------
T2 = "Test 2 (Reduced, Default, No Noise)"
data = load_multiseed()
gap = lambda r: 100.0 * (opt2 - r) / abs(opt2)
entries = [("numerical optimum (this script)", opt2, "reference")]
for name, pol in REFERENCE_POLICIES.items():
    entries.append((name, out["4D Default (Test 2; training model of Model 2)"]["references"][name]["return"], "open-loop"))
with open(REPO / "dp_ablation_results.json") as f:
    raw_dp = json.load(f)
    dp_rows = raw_dp["linear_grid"] if isinstance(raw_dp, dict) else raw_dp
    for d in dp_rows:
        entries.append((f"manuscript DP, {d['grid_points_per_dim']}^4 linear grid", d["test2_reward"], "DP"))
for model in ("Model 2 (Reduced, Default, Noise)",):
    for algo in ALGOS:
        vals = [r["test_evals"][T2]["reward_mean"] for r in rl_runs(data, model, algo)]
        entries.append((f"{algo}, trained on Model 2 (mean of 5 seeds)", float(np.mean(vals)), "DRL"))
best_run = max(data["rl_runs"], key=lambda r: r["test_evals"][T2]["reward_mean"])
entries.append((f"best single DRL run ({best_run['scenario_name'].split(' (')[0]} {best_run['algo']}, seed {best_run['seed']})", float(best_run["test_evals"][T2]["reward_mean"]), "DRL"))
entries.sort(key=lambda e: -e[1])
gap_rows = [[n, f"{r:.0f}", f"{gap(r):.1f}%", k] for n, r, k in entries]
out["test2_gap"] = [{"method": n, "return": r, "gap_pct": gap(r), "kind": k} for n, r, k in entries]
print("[b] Test 2 optimality gaps:", "; ".join(f"{n}: {gap(r):.1f}%" for n, r, k in entries if k == "DRL"))

# ---- c. how much could feedback gain? ---------------------------------------------------------
T1 = "Test 1 (Reduced, Default, Noise)"
p0, N = ODE_PARAMS["default"], 60
nominal = schedule_policy(xA)
r_nom = seed_means(nominal, T1)
r_two = seed_means(two_stage_policy, T1)
r_half = seed_means(const_policy(500.0, 50.0), T1)
r_zero = seed_means(const_policy(0.0, 0.0), T1)
clair = []
for s in SEEDS:
    vals = []
    for ep in range(10):
        env = Env("reduced", p0, NOISE["default"], REGIME_A)
        env.reset(seed=s * 100 + ep)
        res = minimize(make_objective(env.state.copy(), p0, False, REGIME_A, 20), xA, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * (2 * N), options=dict(maxiter=300, ftol=1e-13, gtol=1e-9))
        vals.append(run_episode(Env("reduced", p0, NOISE["default"], REGIME_A), schedule_policy(res.x), seed=s * 100 + ep)["reward"])
    clair.append(np.mean(vals))
r_cl = np.array(clair)
fb_rows = []
for name, a in [("no dosing", r_zero), ("half dose (500, 50)", r_half), ("two-stage", r_two), ("one fixed schedule: the nominal optimum", r_nom),
                ("bound: schedule re-optimised for each episode's true initial state", r_cl)]:
    fb_rows.append([name, f"{a.mean():.0f} ± {a.std(ddof=1):.0f}", f"{100 * (r_cl.mean() - a.mean()) / abs(r_cl.mean()):.3f}%"])
out["test1_feedback_value"] = {"no_dosing": r_zero, "half": r_half, "two_stage": r_two, "nominal_optimum": r_nom, "clairvoyant": r_cl,
                               "headroom_pct": 100 * (r_cl.mean() - r_nom.mean()) / abs(r_cl.mean())}
print(f"[c] Test 1: fixed nominal-optimal schedule {r_nom.mean():.0f}, per-episode bound {r_cl.mean():.0f} -> headroom for feedback "
      f"{out['test1_feedback_value']['headroom_pct']:.4f}%")

sets = sampled_parameter_sets()
def param_eval(pol):
    return np.array([run_episode(Env("reduced", sp, NOISE["default"], REGIME_A), pol, seed=5000 + i)["reward"] for i, sp in enumerate(sets)])
pe = {"no dosing": param_eval(const_policy(0.0, 0.0)), "half dose (500, 50)": param_eval(const_policy(500.0, 50.0)), "two-stage": param_eval(two_stage_policy),
      "one fixed schedule: the nominal optimum": param_eval(nominal)}
cl = []
for i, sp in enumerate(sets):
    env = Env("reduced", sp, NOISE["default"], REGIME_A)
    env.reset(seed=5000 + i)
    res = minimize(make_objective(env.state.copy(), sp, False, REGIME_A, 20), xA, jac=True, method="L-BFGS-B",
                   bounds=[(0.0, 1.0)] * (2 * N), options=dict(maxiter=300, ftol=1e-13, gtol=1e-9))
    cl.append(run_episode(Env("reduced", sp, NOISE["default"], REGIME_A), schedule_policy(res.x), seed=5000 + i)["reward"])
pe["bound: schedule re-optimised for each parameter set"] = np.array(cl)
cm = pe["bound: schedule re-optimised for each parameter set"].mean()
pu_rows = [[n, f"{a.mean():.0f}", f"{a.std(ddof=1):.0f}", f"{100 * (cm - a.mean()) / abs(cm):.2f}%"] for n, a in pe.items()]
out["param_feedback_value"] = {**pe, "headroom_pct": 100 * (cm - pe["one fixed schedule: the nominal optimum"].mean()) / abs(cm)}
print(f"    parameter sets: fixed nominal-optimal schedule {pe['one fixed schedule: the nominal optimum'].mean():.0f}, per-set bound {cm:.0f} -> headroom "
      f"{out['param_feedback_value']['headroom_pct']:.3f}%")

# ---- d. Test 4 is the 4D Modified model in (U, I, V, C) ---------------------------------------
diffs = []
for name, a in CONSTANTS.items():
    r6 = run_episode(Env("full", ODE_PARAMS["modified_no_lysis"], (0.0, 0.0), REGIME_A), const_policy(*a), record=True)
    r4 = run_episode(Env("reduced", ODE_PARAMS["modified"], (0.0, 0.0), REGIME_A), const_policy(*a), record=True)
    n = min(len(r6["traj"]), len(r4["traj"]))
    sub = r6["traj"][:n][:, [0, 1, 2, 5]]
    diffs.append(float(np.max(np.abs(sub - r4["traj"][:n]) / np.maximum(np.abs(r4["traj"][:n]), 1.0))))
    assert r6["steps"] == r4["steps"]
out["test4_vs_4d_modified_max_rel_state_diff"] = max(diffs)
print(f"[d] Test 4 (6D, zero lysis) vs 4D Modified, (U,I,V,C) trajectories under four constant schedules: max relative difference {max(diffs):.1e}")

save_json("02_optimal_control.json", out)
(RESULTS / "02_optimal_control.md").write_text(f"""# 02 - Numerically optimal schedules and what they imply

All returns are computed in the exact evaluation environment (RK45, termination at U+I < 0.1), nominal initial
condition, no noise, unless stated. "Gap to optimum" is (optimum - return) / |optimum|.

## a. Optimum per environment

{chr(10).join(md_sections)}

## b. Optimality gap of every method on Test 2 (Regime A)

{md_table(["method", "Test 2 return", "gap to optimum", "kind"], gap_rows)}

## c. How much could a feedback policy gain over one fixed schedule?

Test 1 (5% initial-condition noise, 1% observation noise; the manuscript's 5 x 10 episodes; mean ± SD over seed cohorts):

{md_table(["policy", "Test 1 return", "gap to the per-episode bound"], fb_rows)}

The manuscript's ten perturbed parameter sets (one noisy episode each):

{md_table(["policy", "mean return", "SD across sets", "gap to the per-set bound"], pu_rows)}

The last row of each table re-optimises the whole schedule with exact knowledge of that episode's initial state
(or parameter set). No feedback policy can do better, so the difference between the last two rows is the largest
improvement closed-loop control can show in these tests.

## d. Test 4 and the 4D Modified model

With nu_U = nu_I = tau = 0 the immune variables no longer act on U, I, V or C. Under four constant schedules the
(U, I, V, C) trajectories of Test 4 and of the 4D Modified model differ by at most {max(diffs):.1e} (relative).

Total run time {time.time() - t_start:.0f} s.
""")
print(f"wrote results/02_optimal_control.json and .md ({time.time() - t_start:.0f} s)")
