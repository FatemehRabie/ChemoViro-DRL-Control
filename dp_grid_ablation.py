import os
import gc
import time
import json
import tracemalloc
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interpn, RegularGridInterpolator

import run_multiseed_drl as R

DEFAULT_PARAMS = R.ODE_PARAMS["default"]
SEEDS = R.SEEDS
ACTIONS_LIST = [(a, b) for a in (0.0, 500.0, 1000.0) for b in (0.0, 50.0, 100.0)]


def reduced_rhs_vec(U, I, V, C, u1, u2, p):
    U = np.maximum(0.0, U)
    I = np.maximum(0.0, I)
    V = np.maximum(0.0, V)
    C = np.maximum(0.0, C)
    inf = (p["beta"] * U * V) / (p["Ku"] + U)
    kc_c = p["Kc"] + C
    dU = p["alpha"] * U * (1.0 - (U + I) / p["K"]) - inf - (p["delta_U"] * U * C) / kc_c
    dI = inf - p["delta"] * I - (p["delta_I"] * I * C) / kc_c
    dV = p["b"] * p["delta"] * I - inf - p["gamma"] * V + u1
    dC = u2 - p["psi"] * C
    return dU, dI, dV, dC


def rk4_integrate_vec(U, I, V, C, u1, u2, dt, n_sub, p):
    h = dt / n_sub
    for _ in range(n_sub):
        k1U, k1I, k1V, k1C = reduced_rhs_vec(U, I, V, C, u1, u2, p)
        k2U, k2I, k2V, k2C = reduced_rhs_vec(U + 0.5 * h * k1U, I + 0.5 * h * k1I, V + 0.5 * h * k1V, C + 0.5 * h * k1C, u1, u2, p)
        k3U, k3I, k3V, k3C = reduced_rhs_vec(U + 0.5 * h * k2U, I + 0.5 * h * k2I, V + 0.5 * h * k2V, C + 0.5 * h * k2C, u1, u2, p)
        k4U, k4I, k4V, k4C = reduced_rhs_vec(U + h * k3U, I + h * k3I, V + h * k3V, C + h * k3C, u1, u2, p)
        U = np.clip(U + (h / 6.0) * (k1U + 2 * k2U + 2 * k3U + k4U), 0.0, 1e6)
        I = np.clip(I + (h / 6.0) * (k1I + 2 * k2I + 2 * k3I + k4I), 0.0, 1e6)
        V = np.clip(V + (h / 6.0) * (k1V + 2 * k2V + 2 * k3V + k4V), 0.0, 1e5)
        C = np.clip(C + (h / 6.0) * (k1C + 2 * k2C + 2 * k3C + k4C), 0.0, 100.0)
    return U, I, V, C


def reduced_rhs_scalar(t, y, action, p):
    U = max(0.0, y[0]); I = max(0.0, y[1]); V = max(0.0, y[2]); C = max(0.0, y[3])
    u1, u2 = float(action[0]), float(action[1])
    inf = (p["beta"] * U * V) / (p["Ku"] + U)
    kc_c = p["Kc"] + C
    return [
        p["alpha"] * U * (1.0 - (U + I) / p["K"]) - inf - (p["delta_U"] * U * C) / kc_c,
        inf - p["delta"] * I - (p["delta_I"] * I * C) / kc_c,
        p["b"] * p["delta"] * I - inf - p["gamma"] * V + u1,
        u2 - p["psi"] * C
    ]


def run_dp_grid(n_pts, weights=(1e-2, 1e-1), params=DEFAULT_PARAMS):
    """Original uniform linear-grid 4D DP (U, I in [0, 1e6], V in [0, 1e5], C in [0, 100], 5-day stages)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    state_grids = (
        np.linspace(0, 1e6, n_pts),
        np.linspace(0, 1e6, n_pts),
        np.linspace(0, 1e5, n_pts),
        np.linspace(0, 100, n_pts)
    )
    actions = np.array(np.meshgrid([0.0, 500.0, 1000.0], [0.0, 50.0, 100.0])).T.reshape(-1, 2)
    T_final, dt = 100.0, 5.0
    N = len(np.arange(0, T_final + dt, dt))
    total_states = n_pts ** 4
    # Theoretical memory for storing full time-indexed float64 value + 2 policy tables across N stages:
    mem_sweep_mb = (3 * N * total_states * 8) / (1024 ** 2)

    U_g, I_g, V_g, C_g = np.meshgrid(*state_grids, indexing='ij')
    stage_base = U_g + I_g
    A1, A2 = weights

    next_pts_per_action = []
    act_costs = []
    for a_idx, (u1, u2) in enumerate(actions):
        nU, nI, nV, nC = rk4_integrate_vec(U_g, I_g, V_g, C_g, u1, u2, dt, 25, params)
        pts = np.stack([nU, nI, nV, nC], axis=-1).reshape(-1, 4)
        next_pts_per_action.append(pts)
        act_costs.append((0.5 * A1 * u1 ** 2 + 0.5 * A2 * u2 ** 2) * dt)

    V_next = np.zeros((n_pts, n_pts, n_pts, n_pts), dtype=np.float64)
    p_u1_0 = np.zeros((n_pts, n_pts, n_pts, n_pts), dtype=np.float64)
    p_u2_0 = np.zeros((n_pts, n_pts, n_pts, n_pts), dtype=np.float64)

    for k in range(N - 2, -1, -1):
        min_cost = np.full((n_pts, n_pts, n_pts, n_pts), np.inf, dtype=np.float64)
        best_u1 = np.zeros_like(min_cost)
        best_u2 = np.zeros_like(min_cost)
        rgi_v = RegularGridInterpolator(state_grids, V_next, bounds_error=False, fill_value=np.inf)
        for a_idx, (u1, u2) in enumerate(actions):
            fut = rgi_v(next_pts_per_action[a_idx]).reshape(n_pts, n_pts, n_pts, n_pts)
            tot = stage_base * dt + act_costs[a_idx] + fut
            mask = tot < min_cost
            min_cost[mask] = tot[mask]
            best_u1[mask] = u1
            best_u2[mask] = u2
        V_next = min_cost
        if k == 0:
            p_u1_0 = best_u1
            p_u2_0 = best_u2

    vec_runtime = time.time() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_bytes / (1024 ** 2)
    policy_nbytes_mb = (p_u1_0.nbytes + p_u2_0.nbytes) / (1024 ** 2)
    serial_est_s = (N - 1) * total_states * len(actions) * (7393.80 / (20 * 12000 * 9))
    return (p_u1_0, p_u2_0, state_grids, actions), vec_runtime, serial_est_s, peak_ram_mb, policy_nbytes_mb, mem_sweep_mb


def eval_dp(dp_tuple, weights=(1e-2, 1e-1), noise=False, seed=42):
    """Legacy single-episode evaluator retained for backward compatibility with revision_checks."""
    p_u1_0, p_u2_0, state_grids, actions = dp_tuple
    rng = np.random.default_rng(seed)
    base = np.array([10000.0, 100.0, 500.0, 100.0])
    y = base * (1.0 + 0.05 * rng.standard_normal(4)) if noise else base.copy()
    A1, A2 = weights
    total_r = 0.0
    for step in range(400):
        obs = y * (1.0 + 0.01 * rng.standard_normal(4)) if noise else y.copy()
        obs_c = np.clip(obs, [0, 0, 0, 0], [1e6, 1e6, 1e5, 100.0])
        u1 = interpn(state_grids, p_u1_0, obs_c, bounds_error=False, fill_value=None)[0]
        u2 = interpn(state_grids, p_u2_0, obs_c, bounds_error=False, fill_value=None)[0]
        act = actions[np.argmin(np.linalg.norm(actions - np.array([u1, u2]), axis=1))]
        sol = solve_ivp(reduced_rhs_scalar, (0.0, 1.0), y, args=(act, DEFAULT_PARAMS), method='RK45', rtol=1e-6, atol=1e-6)
        y = np.clip(sol.y[:, -1], 0.0, 1e8)
        tumor = float(y[0] + y[1])
        total_r += -tumor - (0.5 * A1 * act[0] ** 2 + 0.5 * A2 * act[1] ** 2)
        if tumor < 0.1:
            return total_r, tumor, step + 1
    return total_r, float(y[0] + y[1]), 400


def make_linear_dp_callable(dp_tuple):
    p_u1_0, p_u2_0, state_grids, actions = dp_tuple
    rgi_u1 = RegularGridInterpolator(state_grids, p_u1_0, bounds_error=False, fill_value=None)
    rgi_u2 = RegularGridInterpolator(state_grids, p_u2_0, bounds_error=False, fill_value=None)
    def fn(t, obs):
        obs_c = np.clip(obs[:4], [0, 0, 0, 0], [1e6, 1e6, 1e5, 100.0])
        u1 = rgi_u1(obs_c)[0]
        u2 = rgi_u2(obs_c)[0]
        act = actions[np.argmin(np.linalg.norm(actions - np.array([u1, u2]), axis=1))]
        return float(act[0]), float(act[1])
    return fn


def make_log_dp_callable(dp_obj):
    g, n, pol, acts = dp_obj["grids"], dp_obj["n"], dp_obj["pol"], np.array(dp_obj["actions"])
    def fn(t, obs):
        c = [np.log10(1.0 + max(float(obs[0]), 0.0)), np.log10(1.0 + max(float(obs[1]), 0.0)),
             np.log10(1.0 + max(float(obs[2]), 0.0)), max(float(obs[3]), 0.0)]
        ii = [int(np.clip(round((ci - gi[0]) / (gi[1] - gi[0])), 0, n - 1)) for ci, gi in zip(c, g)]
        return tuple(acts[pol[tuple(ii)]])
    return fn


def evaluate_dp_across_tests(fn, weights=(1e-2, 1e-1)):
    """Evaluates any 4D DP policy on the EXACT shared 5 x 10 episodes across Tests 1-6 (using Pi_{6->4} on 6D tests)."""
    adapter = R.CallablePolicyAdapter(fn)
    per_test = {}
    for t_cfg in R.TESTING_SCENARIOS:
        if not t_cfg["noise"]:
            ev = R.eval_model_on_env(adapter, 4, t_cfg, weights, n_episodes=1, base_seed=SEEDS[0] * 100)
            seed_rewards = [ev["reward_mean"]] * len(SEEDS)
            seed_burdens = [ev["terminal_burden"]] * len(SEEDS)
            seed_steps = [ev["steps"]] * len(SEEDS)
        else:
            seed_rewards, seed_burdens, seed_steps = [], [], []
            for s in SEEDS:
                ev = R.eval_model_on_env(adapter, 4, t_cfg, weights, n_episodes=10, base_seed=s * 100)
                seed_rewards.append(ev["reward_mean"])
                seed_burdens.append(ev["terminal_burden"])
                seed_steps.append(ev["steps"])
        per_test[t_cfg["name"]] = {
            "seed_rewards": seed_rewards,
            "reward_mean": float(np.mean(seed_rewards)),
            "reward_sd": float(np.std(seed_rewards, ddof=1)),
            "burden_mean": float(np.mean(seed_burdens)),
            "steps_mean": float(np.mean(seed_steps)),
        }
    return per_test


def plot_dp_ablation(linear_results, log_results, opt_t2, twostage_t2, half_t2, out_png):
    x_lin = np.array([r["grid_points_per_dim"] for r in linear_results], dtype=float)
    x_log = np.array([r["grid_points_per_dim"] for r in log_results], dtype=float)
    grids_lin = [f"${r['grid_points_per_dim']}^4$" for r in linear_results]

    lin_t1_m = [r["test1_reward_mean"] / 1e6 for r in linear_results]
    lin_t1_sd = [r["test1_reward_sd"] / 1e6 for r in linear_results]
    lin_t2 = [r["test2_reward"] / 1e6 for r in linear_results]

    log_t1_m = [r["test1_reward_mean"] / 1e6 for r in log_results]
    log_t1_sd = [r["test1_reward_sd"] / 1e6 for r in log_results]
    log_t2 = [r["test2_reward"] / 1e6 for r in log_results]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    # Panel (a): Return vs Grid Resolution (Linear vs Log-Spaced)
    axes[0].axhline(opt_t2 / 1e6, color='black', linestyle='-', linewidth=1.8, label=f"Numerical Optimum ({opt_t2/1e6:.3f}M)")
    axes[0].axhline(twostage_t2 / 1e6, color='tab:red', linestyle='-.', linewidth=1.6, label=f"2-Stage Open-Loop ({twostage_t2/1e6:.3f}M)")
    axes[0].axhline(half_t2 / 1e6, color='tab:green', linestyle=':', linewidth=1.6, label=f"Constant Half-Dose ({half_t2/1e6:.3f}M)")
    axes[0].errorbar(x_log, log_t1_m, yerr=log_t1_sd, fmt='-D', color='tab:cyan', markersize=6.5, capsize=4, linewidth=2.1, label="Log-Grid DP Test 1 (Shared 5×10)")
    axes[0].plot(x_log, log_t2, '--d', color='teal', markersize=6.0, linewidth=1.8, label="Log-Grid DP Test 2 (Determ.)")
    axes[0].errorbar(x_lin, lin_t1_m, yerr=lin_t1_sd, fmt='-o', color='tab:blue', markersize=6.0, capsize=4, linewidth=2.0, label="Linear-Grid DP Test 1 (Shared 5×10)")
    axes[0].plot(x_lin, lin_t2, '--s', color='tab:orange', markersize=6.0, linewidth=1.8, label="Linear-Grid DP Test 2 (Determ.)")
    axes[0].set_xticks(x_lin); axes[0].set_xticklabels(grids_lin)
    axes[0].set_xlabel("Discretized Grid Resolution ($N_g^4$)", fontsize=10.5)
    axes[0].set_ylabel("Cumulative Return ($\\times 10^6$)", fontsize=10.5)
    axes[0].set_title("(a) Policy Return: Log-Spaced vs. Linear Grid", fontweight="bold", fontsize=11.2)
    axes[0].legend(fontsize=7.8, loc="lower right")
    axes[0].grid(True, linestyle='--', alpha=0.45)

    # Panel (b): Runtime Scaling
    axes[1].plot(x_lin, [r["vectorized_runtime_s"] for r in linear_results], '-o', color='tab:blue', markersize=6.5, linewidth=2.0, label="Linear-Grid 20-Stage Sweep (s)")
    axes[1].plot(x_log, [r["vectorized_runtime_s"] for r in log_results], '-D', color='teal', markersize=6.5, linewidth=2.0, label="Log-Grid Daily Value Iter. (s)")
    axes[1].plot(x_lin, [r["serial_solve_ivp_est_s"] for r in linear_results], '--^', color='tab:brown', markersize=6.0, linewidth=1.8, label="Serial SciPy solve_ivp (Est., s)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x_lin); axes[1].set_xticklabels(grids_lin)
    axes[1].set_xlabel("Discretized Grid Resolution ($N_g^4$)", fontsize=10.5)
    axes[1].set_ylabel("Wall-Clock Runtime (seconds, log scale)", fontsize=10.5)
    axes[1].set_title("(b) Computation Time Scaling ($O(N_g^4)$)", fontweight="bold", fontsize=11.2)
    axes[1].legend(fontsize=8.2, loc="upper left")
    axes[1].grid(True, which="both", linestyle='--', alpha=0.35)

    # Panel (c): Measured Peak RAM vs Saved Policy Table Size
    axes[2].plot(x_lin, [r["peak_measured_ram_mb"] for r in linear_results], '-o', color='tab:pink', markersize=6.5, linewidth=2.0, label="Linear-Grid Peak RAM (Measured, MB)")
    axes[2].plot(x_log, [r["peak_measured_ram_mb"] for r in log_results], '--s', color='tab:purple', markersize=6.0, linewidth=1.8, label="Log-Grid Peak RAM (Measured, MB)")
    axes[2].plot(x_log, [r["policy_int8_kb"] / 1024.0 for r in log_results], '-D', color='teal', markersize=6.5, linewidth=2.0, label="Saved Log-Grid int8 Policy (MB)")
    for xi, mi in zip(x_lin, [r["peak_measured_ram_mb"] for r in linear_results]):
        axes[2].annotate(f"{mi:.1f} MB", (xi, mi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.8)
    for xi, ki in zip(x_log, [r["policy_int8_kb"] for r in log_results]):
        axes[2].annotate(f"{ki:.0f} kB", (xi, ki / 1024.0), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.8, color="teal")
    axes[2].set_yscale("log")
    axes[2].set_ylim(2e-3, 1e4)
    axes[2].set_xticks(x_lin); axes[2].set_xticklabels(grids_lin)
    axes[2].set_xlabel("Discretized Grid Resolution ($N_g^4$)", fontsize=10.5)
    axes[2].set_ylabel("Measured Memory (MB, log scale)", fontsize=10.5)
    axes[2].set_title("(c) Measured Peak RAM & Saved Policy Size", fontweight="bold", fontsize=11.2)
    axes[2].legend(fontsize=8.2, loc="lower right")
    axes[2].grid(True, which="both", linestyle='--', alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights = (1e-2, 1e-1)

    # Reference values on Test 2
    x_opt = R.solve_daily_ocp(R.INITIAL_CONDITIONS["nominal_4d"], DEFAULT_PARAMS, False, weights=weights, N=65, n_sub=40)
    opt_pol = R.CallablePolicyAdapter(lambda t, o: (x_opt[t] * 1000.0, x_opt[65 + t] * 100.0) if t < 65 else (0.0, 0.0))
    opt_t2 = R.eval_model_on_env(opt_pol, 4, R.TESTING_SCENARIOS[1], weights, n_episodes=1)["reward_mean"]
    twostage_t2 = R.eval_model_on_env(R.CallablePolicyAdapter(lambda t, o: (1000.0, 100.0) if t < 10 else (0.0, 100.0)), 4, R.TESTING_SCENARIOS[1], weights, n_episodes=1)["reward_mean"]
    half_t2 = R.eval_model_on_env(R.CallablePolicyAdapter(lambda t, o: (500.0, 50.0)), 4, R.TESTING_SCENARIOS[1], weights, n_episodes=1)["reward_mean"]

    linear_results = []
    for n_g in [10, 15, 20, 30, 50]:
        print(f"Running Linear-Grid 4D DP for {n_g}^4 = {n_g ** 4} states...")
        dp_tup, v_rt, s_rt, peak_ram_mb, pol_mb, mem_mb = run_dp_grid(n_g, weights=weights)
        fn_lin = make_linear_dp_callable(dp_tup)
        per_test = evaluate_dp_across_tests(fn_lin, weights=weights)
        t1 = per_test["Test 1 (Reduced, Default, Noise)"]
        t2 = per_test["Test 2 (Reduced, Default, No Noise)"]
        res = {
            "grid_type": "linear",
            "grid_points_per_dim": n_g,
            "total_states": n_g ** 4,
            "peak_measured_ram_mb": round(peak_ram_mb, 2),
            "memory_mb": round(peak_ram_mb, 2),
            "memory_sweep_formula_mb": round(mem_mb, 2),
            "policy_float64_pair_mb": round(pol_mb, 3),
            "policy_int8_kb": round((n_g ** 4) / 1024.0, 2),
            "vectorized_runtime_s": round(v_rt, 2),
            "serial_solve_ivp_est_s": round(s_rt, 1),
            "test1_reward_mean": t1["reward_mean"],
            "test1_reward_sd": t1["reward_sd"],
            "test1_burden_mean": t1["burden_mean"],
            "test2_reward": t2["reward_mean"],
            "test2_burden": t2["burden_mean"],
            "test2_steps": int(round(t2["steps_mean"])),
            "test2_shortfall_pct": round((abs(t2["reward_mean"]) - abs(opt_t2)) / abs(opt_t2) * 100.0, 2),
            "all_tests": per_test
        }
        print(f"  Linear {n_g}^4: rt={v_rt:.2f}s | peakRAM={peak_ram_mb:.2f}MB | T1={t1['reward_mean']:.0f}+/-{t1['reward_sd']:.0f} | T2={t2['reward_mean']:.0f} ({res['test2_shortfall_pct']}%)")
        linear_results.append(res)

    log_results = []
    for n_g in [10, 15, 20, 30]:
        print(f"Running Log-Spaced 4D DP for {n_g}^4 = {n_g ** 4} states...")
        gc.collect()
        tracemalloc.start()
        dp_log = R.build_log_dp_4d(n_g, DEFAULT_PARAMS, weights=weights)
        _, log_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        log_peak_mb = log_peak_bytes / (1024 ** 2)
        fn_log = make_log_dp_callable(dp_log)
        per_test = evaluate_dp_across_tests(fn_log, weights=weights)
        t1 = per_test["Test 1 (Reduced, Default, Noise)"]
        t2 = per_test["Test 2 (Reduced, Default, No Noise)"]
        res = {
            "grid_type": "log_spaced",
            "grid_points_per_dim": n_g,
            "total_states": n_g ** 4,
            "sweeps": dp_log["sweeps"],
            "peak_measured_ram_mb": round(log_peak_mb, 2),
            "policy_int8_kb": round(dp_log["policy_bytes"] / 1024.0, 2),
            "value_float64_mb": round(((n_g ** 4) * 8) / (1024 ** 2), 3),
            "vectorized_runtime_s": round(dp_log["seconds"], 2),
            "test1_reward_mean": t1["reward_mean"],
            "test1_reward_sd": t1["reward_sd"],
            "test1_burden_mean": t1["burden_mean"],
            "test2_reward": t2["reward_mean"],
            "test2_burden": t2["burden_mean"],
            "test2_steps": int(round(t2["steps_mean"])),
            "test2_shortfall_pct": round((abs(t2["reward_mean"]) - abs(opt_t2)) / abs(opt_t2) * 100.0, 2),
            "all_tests": per_test
        }
        print(f"  Log-Grid {n_g}^4: rt={dp_log['seconds']:.2f}s ({dp_log['sweeps']} sweeps) | peakRAM={log_peak_mb:.2f}MB | T1={t1['reward_mean']:.0f}+/-{t1['reward_sd']:.0f} | T2={t2['reward_mean']:.0f} ({res['test2_shortfall_pct']}%)")
        log_results.append(res)

    # Also build Log-Spaced 4D DP (20^4) on the 4D Modified parameters to test matched-parameter 6D projection on Tests 3 & 4
    dp_log_mod20 = R.build_log_dp_4d(20, R.ODE_PARAMS["modified"], weights=weights, zmax=(8.2, 8.2, 10.0))
    mod20_tests = evaluate_dp_across_tests(make_log_dp_callable(dp_log_mod20), weights=weights)

    out_payload = {
        "linear_grid": linear_results,
        "log_grid": log_results,
        "log_grid_modified_20": {
            "seconds": round(dp_log_mod20["seconds"], 2),
            "sweeps": dp_log_mod20["sweeps"],
            "policy_int8_kb": round(dp_log_mod20["policy_bytes"] / 1024.0, 2),
            "all_tests": mod20_tests
        },
        "references_test2": {
            "numerical_optimum": opt_t2,
            "two_stage_open_loop": twostage_t2,
            "constant_half_dose": half_t2
        }
    }
    out_json = os.path.join(base_dir, "dp_ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(out_payload, f, indent=2)

    out_png = os.path.join(base_dir, "dp_ablation.png")
    plot_dp_ablation(linear_results, log_results, opt_t2, twostage_t2, half_t2, out_png)
    parent_png = os.path.join(os.path.dirname(base_dir), "dp_ablation.png")
    if os.path.isdir(os.path.dirname(base_dir)):
        plot_dp_ablation(linear_results, log_results, opt_t2, twostage_t2, half_t2, parent_png)
    print("Saved complete DP ablation results and figure to", out_json, "and", out_png)
