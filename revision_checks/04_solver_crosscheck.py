"""Check of the manuscript's integrator statement.

Manuscript (Section 2.3, Table 13, response letter): "the maximum relative state and cumulative-return
discrepancy between the fixed-substep RK4 integrator and adaptive RK45 across nominal and perturbed
trajectories is < 1.2e-4".

This script runs the repository's own RK4 routines (taken verbatim from ../run_multiseed_drl.py: 20 sub-steps
per day in 4D, 40 in 6D) against solve_ivp RK45 (rtol = atol = 1e-6, the evaluation integrator) and against a
tight reference (DOP853, rtol 1e-12), for every model/parameter combination used for training or testing,
four dosing schedules, and the nominal plus two 5%-perturbed initial states.

Relative state differences are reported only for components larger than 1 (a relative difference of a state that
has decayed to 1e-10 is meaningless).

Run:  python 04_solver_crosscheck.py   (about 1 minute; numpy/scipy only)
Out:  results/04_solver_crosscheck.json, results/04_solver_crosscheck.md
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from common import NOMINAL_IC, ODE_PARAMS, REGIME_A, REPO, RESULTS, md_table, save_json

# Take the repository's integrators verbatim (the file itself needs gymnasium/SB3 to import, so the
# pure-numpy block between the two markers is executed on its own).
src = (REPO / "run_multiseed_drl.py").read_text()
block = src[src.index("def reduced_rhs_scalar"): src.index("class ChemoViroEnv")]
ns = {"np": np}
exec(compile(block, "run_multiseed_drl.py (integrators)", "exec"), ns)


def rollout(model, p, policy, y0, integrator, n_sub=None, max_steps=400):
    y = np.array(y0, dtype=float)
    total, states = 0.0, [y.copy()]
    for k in range(max_steps):
        u1, u2 = policy(k)
        if integrator == "rk4":
            if model == "reduced":
                y = ns["rk4_step_reduced"](y, u1, u2, p, 1.0, **({"n_sub": n_sub} if n_sub else {}))
            else:
                y = ns["rk4_step_full"](y, u1, u2, p, 1.0, **({"n_sub": n_sub} if n_sub else {}))
        else:
            rhs = ns["reduced_rhs_scalar"] if model == "reduced" else ns["full_rhs_scalar"]
            kw = dict(method="RK45", rtol=1e-6, atol=1e-6) if integrator == "rk45" else dict(method="DOP853", rtol=1e-12, atol=1e-14)
            y = np.clip(solve_ivp(rhs, (k, k + 1.0), y, args=((u1, u2), p), **kw).y[:, -1], 0.0, 1e8)
        tumor = y[0] + y[1]
        total += -tumor - (0.5 * REGIME_A[0] * u1 * u1 + 0.5 * REGIME_A[1] * u2 * u2)
        states.append(y.copy())
        if tumor < 0.1:
            break
    return total, np.array(states)


POLICIES = {"no dosing": lambda k: (0.0, 0.0), "half dose": lambda k: (500.0, 50.0), "full dose": lambda k: (1000.0, 100.0),
            "two-stage": lambda k: (1000.0, 100.0) if k < 10 else (0.0, 100.0)}
ENVS = {"4D Default (Model 2 training; Tests 1-2)": ("reduced", "default"),
        "4D Modified (Models 3-4 training)": ("reduced", "modified"),
        "6D Default (Model 1 training)": ("full", "default"),
        "6D Modified (Test 3)": ("full", "modified"),
        "6D Modified, zero lysis (Test 4)": ("full", "modified_no_lysis")}

rng = np.random.default_rng(7)
rows, out = [], {}
for env_name, (model, pkey) in ENVS.items():
    p = ODE_PARAMS[pkey]
    worst = {"ret_rk4_vs_rk45": 0.0, "ret_rk45_vs_ref": 0.0, "ret_rk4_vs_ref": 0.0, "state_rk4_vs_rk45": 0.0, "tumor_rk4_vs_rk45": 0.0}
    for pol in POLICIES.values():
        for trial in range(3):
            y0 = np.array(NOMINAL_IC[model]) * (1.0 + (0.05 * rng.standard_normal(len(NOMINAL_IC[model])) if trial else 0.0))
            r4, s4 = rollout(model, p, pol, y0, "rk4")
            r5, s5 = rollout(model, p, pol, y0, "rk45")
            rr, _ = rollout(model, p, pol, y0, "ref")
            n = min(len(s4), len(s5))
            big = np.abs(s5[:n]) > 1.0
            rel = np.abs(s4[:n] - s5[:n]) / np.maximum(np.abs(s5[:n]), 1e-300)
            t4, t5 = s4[:n, 0] + s4[:n, 1], s5[:n, 0] + s5[:n, 1]
            m = t5 > 1.0
            worst["ret_rk4_vs_rk45"] = max(worst["ret_rk4_vs_rk45"], abs(r4 - r5) / abs(r5))
            worst["ret_rk45_vs_ref"] = max(worst["ret_rk45_vs_ref"], abs(r5 - rr) / abs(rr))
            worst["ret_rk4_vs_ref"] = max(worst["ret_rk4_vs_ref"], abs(r4 - rr) / abs(rr))
            worst["state_rk4_vs_rk45"] = max(worst["state_rk4_vs_rk45"], float(rel[big].max()))
            worst["tumor_rk4_vs_rk45"] = max(worst["tumor_rk4_vs_rk45"], float((np.abs(t4 - t5)[m] / t5[m]).max()))
    out[env_name] = worst
    ok = worst["ret_rk4_vs_rk45"] < 1.2e-4 and worst["state_rk4_vs_rk45"] < 1.2e-4
    rows.append([env_name, f"{worst['ret_rk4_vs_rk45']:.1e}", f"{worst['state_rk4_vs_rk45']:.1e}", f"{worst['tumor_rk4_vs_rk45']:.1e}",
                 f"{worst['ret_rk45_vs_ref']:.1e}", "yes" if ok else "NO"])
    print(f"{env_name:44s} return RK4 vs RK45 {worst['ret_rk4_vs_rk45']:.1e} | state {worst['state_rk4_vs_rk45']:.1e} | tumour {worst['tumor_rk4_vs_rk45']:.1e} "
          f"| RK45 vs reference {worst['ret_rk45_vs_ref']:.1e} | within 1.2e-4: {ok}")

# how many sub-steps does the 4D Modified model need?
conv_rows = []
model, p = "reduced", ODE_PARAMS["modified"]
for n_sub in (20, 40, 80, 160):
    w = 0.0
    for pol in POLICIES.values():
        r4, _ = rollout(model, p, pol, NOMINAL_IC[model], "rk4", n_sub=n_sub)
        rr, _ = rollout(model, p, pol, NOMINAL_IC[model], "ref")
        w = max(w, abs(r4 - rr) / abs(rr))
    conv_rows.append([n_sub, f"{w:.1e}"])
    out[f"4D Modified, n_sub={n_sub}"] = w
print("4D Modified, RK4 return error vs reference by sub-steps per day:", conv_rows)

save_json("04_solver_crosscheck.json", out)
(RESULTS / "04_solver_crosscheck.md").write_text(f"""# 04 - Training integrator (RK4) against evaluation integrator (RK45)

Worst case over four dosing schedules and three initial states (nominal and two 5% perturbations), Regime A.

{md_table(["environment", "return: RK4 vs RK45", "states > 1: RK4 vs RK45", "tumour burden > 1: RK4 vs RK45", "return: RK45 vs tight reference", "within the stated 1.2e-4"], rows)}

RK4 return error against the tight reference in the 4D Modified model, nominal initial state, by sub-steps per day:

{md_table(["sub-steps per day", "worst relative return error"], conv_rows)}
""")
print("wrote results/04_solver_crosscheck.json and .md")
