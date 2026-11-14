"""Is the dynamic-programming result in the manuscript a property of DP, or of how the grid was built?

The manuscript's DP (../dp_grid_ablation.py) uses a uniform grid on U, I in [0, 1e6], V in [0, 1e5], C in [0, 100],
five-day stages, and the t = 0 slice of a 100-day finite-horizon policy. This script

  1. shows where the benchmark trajectories actually live relative to that grid;
  2. solves the same problem (same 9 actions) on a grid that is uniform in log10(1 + x), covers the reachable
     range of V, uses daily stages with the evaluation reward and an absorbing state at U+I < 0.1, and
     iterates to a stationary policy;
  3. re-scores the manuscript's own 20^4 DP table on the same 5 x 10 Test 1 episodes used for every other row;
  4. queries 4D DP tables from the 6D tests through the projection (U, I, V, E_V, E_T, C) -> (U, I, V, C),
     the same operator the manuscript applies to the DRL policies.

Run:  python 03_dp_grid_design.py      (about 2 minutes; numpy/scipy only; needs ../dp_grid_ablation.py)
Out:  results/03_dp_grid_design.json, results/03_dp_grid_design.md
"""
from __future__ import annotations

import itertools
import sys
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.interpolate import interpn

from common import (Env, NOISE, ODE_PARAMS, REGIME_A, REPO, RESULTS, SEEDS, TESTS, const_policy, eval_on_test, md_table,
                    rk4_day_batch, run_episode, save_json, seed_means)

sys.path.insert(0, str(REPO))
import dp_grid_ablation as repo_dp  # noqa: E402  (the manuscript's DP code, used unchanged)

ACTIONS = [(a, b) for a in (0.0, 500.0, 1000.0) for b in (0.0, 50.0, 100.0)]  # same 9 actions as the manuscript
T1, T2, T3, T4 = list(TESTS)
out = {}


# ---- log-grid DP ------------------------------------------------------------------------------
def build_log_dp(n, p, weights=REGIME_A, zmax=(6.2, 6.2, 8.0), cmax=30.0, sweeps=80, n_sub=20):
    """Value iteration on a grid uniform in log10(1+x) for U, I, V and linear in C. Returns the stationary policy."""
    t0 = time.time()
    grids = (np.linspace(0, zmax[0], n), np.linspace(0, zmax[1], n), np.linspace(0, zmax[2], n), np.linspace(0, cmax, n))
    ZU, ZI, ZV, CC = np.meshgrid(*grids, indexing="ij")
    Y0 = np.array([10 ** ZU.ravel() - 1, 10 ** ZI.ravel() - 1, 10 ** ZV.ravel() - 1, CC.ravel()])
    S = n ** 4
    A1, A2 = weights
    idxs, wts, stage, term = [], [], [], []
    for u1, u2 in ACTIONS:
        Y = rk4_day_batch(Y0.copy(), u1, u2, p, False, n_sub)
        tumor = Y[0] + Y[1]
        stage.append(tumor + 0.5 * A1 * u1 ** 2 + 0.5 * A2 * u2 ** 2)   # the evaluation reward, sign flipped
        term.append(tumor < 0.1)                                         # absorbing: episode ends
        coords = [np.log10(1 + Y[0]), np.log10(1 + Y[1]), np.log10(1 + Y[2]), Y[3]]
        lo, fr = [], []
        for c, g in zip(coords, grids):
            c = np.clip(c, g[0], g[-1])
            step = g[1] - g[0]
            i = np.minimum((c - g[0]) // step, n - 2).astype(np.int32)
            lo.append(i)
            fr.append((c - g[i]) / step)
        idx = np.empty((16, S), np.int32)
        w = np.empty((16, S), np.float32)
        for k, corner in enumerate(itertools.product((0, 1), repeat=4)):
            ii = [lo[d] + corner[d] for d in range(4)]
            idx[k] = ((ii[0] * n + ii[1]) * n + ii[2]) * n + ii[3]
            ww = np.ones(S)
            for d in range(4):
                ww *= fr[d] if corner[d] else (1.0 - fr[d])
            w[k] = ww
        idxs.append(idx)
        wts.append(w)
    V = np.zeros(S)
    pol = np.zeros(S, np.int8)
    for sweep in range(sweeps):
        best = np.full(S, np.inf)
        for a in range(len(ACTIONS)):
            fut = np.einsum("cs,cs->s", V[idxs[a]], wts[a])
            fut[term[a]] = 0.0
            q = stage[a] + fut
            m = q < best
            best[m] = q[m]
            pol[m] = a
        delta = np.max(np.abs(best - V))
        V = best
        if delta < 1e-6 * max(1.0, np.max(np.abs(V))):
            break
    return {"grids": grids, "pol": pol.reshape(n, n, n, n), "n": n, "seconds": time.time() - t0, "sweeps": sweep + 1}


def log_dp_policy(dp):
    g, n, pol = dp["grids"], dp["n"], dp["pol"]
    acts = np.array(ACTIONS)

    def f(t, obs):
        c = [np.log10(1 + max(obs[0], 0)), np.log10(1 + max(obs[1], 0)), np.log10(1 + max(obs[2], 0)), max(obs[3], 0)]
        ii = [int(np.clip(round((ci - gi[0]) / (gi[1] - gi[0])), 0, n - 1)) for ci, gi in zip(c, g)]
        return tuple(acts[pol[tuple(ii)]])
    return f


def repo_dp_policy(tup):
    """Same lookup rule as eval_dp() in ../dp_grid_ablation.py."""
    p1, p2, grids, acts = tup

    def f(t, obs):
        oc = np.clip(obs, [0, 0, 0, 0], [1e6, 1e6, 1e5, 100.0])
        u1 = interpn(grids, p1, oc, bounds_error=False, fill_value=None)[0]
        u2 = interpn(grids, p2, oc, bounds_error=False, fill_value=None)[0]
        return tuple(acts[np.argmin(np.linalg.norm(acts - np.array([u1, u2]), axis=1))])
    return f


def project(pol4):
    return lambda t, obs: pol4(t, np.asarray(obs)[[0, 1, 2, 5]] if len(obs) == 6 else obs)


def ms(a):
    return f"{np.mean(a):.0f} ± {np.std(a, ddof=1):.0f}"


# ---- 1. where do the trajectories live? --------------------------------------------------------
p0 = ODE_PARAMS["default"]
traj_rows = []
for name, pol in [("no dosing", const_policy(0.0, 0.0)), ("two-stage", lambda t, o: (1000.0, 100.0) if t < 10 else (0.0, 100.0))]:
    r = run_episode(Env("reduced", p0, (0.0, 0.0), REGIME_A), pol, record=True)
    tr = r["traj"]
    over = int((tr[:, 2] > 1e5).sum())
    traj_rows.append([name, f"{tr[:, 0].max():.3g}", f"{tr[:, 1].max():.3g}", f"{tr[:, 2].max():.3g}", f"{over} of {len(tr)}"])
    out[f"trajectory_{name}"] = {"max_U": tr[:, 0].max(), "max_I": tr[:, 1].max(), "max_V": tr[:, 2].max(), "days_V_above_1e5": over, "days": len(tr)}
print("[1] reachable range on Test 2:", traj_rows)

# ---- 2. log-grid DP on the 4D Default model ----------------------------------------------------
log_rows = []
for n in (10, 15, 20):
    dp = build_log_dp(n, p0)
    f = log_dp_policy(dp)
    r2 = run_episode(Env("reduced", p0, (0.0, 0.0), REGIME_A), f)
    r1 = seed_means(f, T1)
    out[f"log_dp_{n}"] = {"states": n ** 4, "seconds": dp["seconds"], "sweeps": dp["sweeps"], "test2": r2["reward"], "test2_days": r2["steps"],
                          "test1_seed_means": r1, "policy_table_bytes_int8": int(dp["pol"].nbytes)}
    log_rows.append([f"{n}^4 = {n ** 4:,}", f"{dp['seconds']:.1f}", f"{dp['pol'].nbytes / 1024:.0f} kB", f"{r2['reward']:.0f}", ms(r1)])
    print(f"[2] log-grid DP {n}^4: {dp['seconds']:.1f} s, Test 2 {r2['reward']:.0f}, Test 1 {ms(r1)}")
    if n == 20:
        dp_default_20 = dp

# ---- 3. manuscript DP at 20^4, same episodes as every other row --------------------------------
tup20 = repo_dp.run_dp_grid(20)[0]
f_repo = repo_dp_policy(tup20)
repo_t1_shared = seed_means(f_repo, T1)
repo_t1_script = np.array([repo_dp.eval_dp(tup20, noise=True, seed=s)[0] for s in SEEDS])
repo_t2 = run_episode(Env("reduced", p0, (0.0, 0.0), REGIME_A), f_repo)["reward"]
out["repo_dp_20"] = {"test1_shared_5x10_episodes": repo_t1_shared, "test1_as_in_dp_grid_ablation_py": repo_t1_script, "test2": repo_t2}
print(f"[3] manuscript DP 20^4 on Test 1: {ms(repo_t1_script)} with its own 5 single episodes; {ms(repo_t1_shared)} on the shared 5 x 10 episodes")

# ---- 4. 4D tables queried from the 6D tests through the projection -----------------------------
dp_modified_20 = build_log_dp(20, ODE_PARAMS["modified"], zmax=(8.2, 8.2, 10.0))
proj_rows = []
for name, pol in [("manuscript DP, 20^4 linear grid (Default model)", project(f_repo)),
                  ("log-grid DP, 20^4, built on the 4D Default model", project(log_dp_policy(dp_default_20))),
                  ("log-grid DP, 20^4, built on the 4D Modified model", project(log_dp_policy(dp_modified_20))),
                  ("no dosing", const_policy(0.0, 0.0)), ("half dose (500, 50)", const_policy(500.0, 50.0))]:
    r3, r4 = seed_means(pol, T3), seed_means(pol, T4)
    out[f"projection_{name}"] = {"test3": r3, "test4": r4}
    proj_rows.append([name, f"{np.mean(r3):.1f} ± {np.std(r3, ddof=1):.1f}", ms(r4)])
    print(f"[4] {name}: Test 3 {np.mean(r3):.1f}, Test 4 {ms(r4)}")

save_json("03_dp_grid_design.json", out)
(RESULTS / "03_dp_grid_design.md").write_text(f"""# 03 - Dynamic programming: grid design, evaluation episodes, and the 6D tests

## 1. Where the Test 2 trajectories live

{md_table(["schedule", "max U", "max I", "max V", "days with V above 1e5"], traj_rows)}

The manuscript's DP grid spans V in [0, 1e5] and clips V at 1e5 inside its own transition model, so for most of every
episode it plans with a virus load one to two orders of magnitude too small. Its U and I axes are uniform on [0, 1e6]:
at 20 points per axis the spacing is 52,632 cells/mm^3 and at 50 points 20,408, so the initial state (U, I) = (10,000, 100)
and the termination threshold 0.1 both sit inside the first cell.

## 2. Same problem, same nine actions, grid uniform in log10(1 + x), V up to 1e8

{md_table(["states", "build time (s)", "policy table (int8)", "Test 2 return", "Test 1 return (5 x 10 shared episodes)"], log_rows)}

For comparison (Test 2): numerical optimum -1,149,610; two-stage schedule -1,165,143; manuscript DP -1,319,462 at 20^4 and
-1,211,746 at 50^4 (881 s).

## 3. The manuscript's 20^4 table on the same episodes as the other rows

`dp_grid_ablation.py` scores Test 1 with one noisy episode per seed from its own random stream:
{ms(repo_t1_script)}. On the 5 x 10 episodes used for the agents and open-loop baselines the same table scores
{ms(repo_t1_shared)}.

## 4. 4D tables on the 6D tests, through the projection used for the DRL policies

{md_table(["policy", "Test 3 return", "Test 4 return"], proj_rows)}

Manuscript values for comparison: PPO/TRPO -5.85 ± 0.48 (Test 3) and -8.11e4 ± 2.59e3 (Test 4); half dose -2.76e3 and -1.075e5.
""")
print("wrote results/03_dp_grid_design.json and .md")
