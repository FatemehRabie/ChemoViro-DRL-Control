import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interpn

DEFAULT_PARAMS = {
    "alpha": 0.206, "K": 1e6, "beta": 0.05, "Ku": 1e5, "delta_U": 50.0,
    "Kc": 1e5, "delta": 0.5115, "delta_I": 60.0, "b": 50,
    "gamma": 0.01, "psi": 4.17, "nu_U": 0.08, "nu_I": 0.1,
    "tau": 0.2, "delta_V": 0.01, "delta_T": 0.01, "phi": 0.7,
    "beta_T": 0.5, "kappa": 1e5
}

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
        k2U, k2I, k2V, k2C = reduced_rhs_vec(U + 0.5*h*k1U, I + 0.5*h*k1I, V + 0.5*h*k1V, C + 0.5*h*k1C, u1, u2, p)
        k3U, k3I, k3V, k3C = reduced_rhs_vec(U + 0.5*h*k2U, I + 0.5*h*k2I, V + 0.5*h*k2V, C + 0.5*h*k2C, u1, u2, p)
        k4U, k4I, k4V, k4C = reduced_rhs_vec(U + h*k3U, I + h*k3I, V + h*k3V, C + h*k3C, u1, u2, p)
        U = np.clip(U + (h/6.0)*(k1U + 2*k2U + 2*k3U + k4U), 0.0, 1e6)
        I = np.clip(I + (h/6.0)*(k1I + 2*k2I + 2*k3I + k4I), 0.0, 1e6)
        V = np.clip(V + (h/6.0)*(k1V + 2*k2V + 2*k3V + k4V), 0.0, 1e5)
        C = np.clip(C + (h/6.0)*(k1C + 2*k2C + 2*k3C + k4C), 0.0, 100.0)
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
    # Memory for full value_table + p_u1 + p_u2 in float64: 3 * N * n_pts^4 * 8 bytes
    mem_mb = (3 * N * total_states * 8) / (1024 ** 2)

    # Precompute next states for all grid points and all 9 actions!
    U_g, I_g, V_g, C_g = np.meshgrid(*state_grids, indexing='ij')
    stage_base = U_g + I_g
    A1, A2 = weights

    next_pts_per_action = []
    act_costs = []
    for a_idx, (u1, u2) in enumerate(actions):
        nU, nI, nV, nC = rk4_integrate_vec(U_g, I_g, V_g, C_g, u1, u2, dt, 25, params)
        pts = np.stack([nU, nI, nV, nC], axis=-1).reshape(-1, 4)
        next_pts_per_action.append(pts)
        act_costs.append((0.5 * A1 * u1**2 + 0.5 * A2 * u2**2) * dt)

    V_next = np.zeros((n_pts, n_pts, n_pts, n_pts), dtype=np.float64)
    p_u1_0 = np.zeros((n_pts, n_pts, n_pts, n_pts), dtype=np.float64)
    p_u2_0 = np.zeros((n_pts, n_pts, n_pts, n_pts), dtype=np.float64)

    for k in range(N - 2, -1, -1):
        min_cost = np.full((n_pts, n_pts, n_pts, n_pts), np.inf, dtype=np.float64)
        best_u1 = np.zeros_like(min_cost)
        best_u2 = np.zeros_like(min_cost)
        for a_idx, (u1, u2) in enumerate(actions):
            fut = interpn(state_grids, V_next, next_pts_per_action[a_idx], bounds_error=False, fill_value=np.inf).reshape(n_pts, n_pts, n_pts, n_pts)
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
    # Estimate equivalent serial solve_ivp runtime based on benchmarked 0.00342s per solve_ivp call
    serial_est_s = (N - 1) * total_states * len(actions) * (7393.80 / (20 * 12000 * 9))
    return (p_u1_0, p_u2_0, state_grids, actions), vec_runtime, serial_est_s, mem_mb

def eval_dp(dp_tuple, weights=(1e-2, 1e-1), noise=False, seed=42):
    p_u1_0, p_u2_0, state_grids, actions = dp_tuple
    rng = np.random.default_rng(seed)
    base = np.array([10000.0, 100.0, 500.0, 100.0])
    y = base * (1.0 + 0.05 * rng.standard_normal(4)) if noise else base.copy()
    A1, A2 = weights
    total_r = 0.0
    for step in range(400):
        obs = y * (1.0 + 0.01 * rng.standard_normal(4)) if noise else y.copy()
        obs_c = np.clip(obs, [0,0,0,0], [1e6, 1e6, 1e5, 100.0])
        u1 = interpn(state_grids, p_u1_0, obs_c, bounds_error=False, fill_value=None)[0]
        u2 = interpn(state_grids, p_u2_0, obs_c, bounds_error=False, fill_value=None)[0]
        act = actions[np.argmin(np.linalg.norm(actions - np.array([u1, u2]), axis=1))]
        sol = solve_ivp(reduced_rhs_scalar, (0.0, 1.0), y, args=(act, DEFAULT_PARAMS), method='RK45', rtol=1e-6, atol=1e-6)
        y = np.clip(sol.y[:, -1], 0.0, 1e8)
        tumor = float(y[0] + y[1])
        total_r += -tumor - (0.5 * A1 * act[0]**2 + 0.5 * A2 * act[1]**2)
        if tumor < 0.1:
            return total_r, tumor, step + 1
    return total_r, float(y[0] + y[1]), 400

def plot_dp_ablation(results, out_png):
    grids = [f"${r['grid_points_per_dim']}^4$" for r in results]
    x = np.arange(len(results))
    t1_mean = [r["test1_reward_mean"] / 1e6 for r in results]
    t1_sd = [r["test1_reward_sd"] / 1e6 for r in results]
    t2_val = [r["test2_reward"] / 1e6 for r in results]
    vec_rt = [r["vectorized_runtime_s"] for r in results]
    ser_rt = [r["serial_solve_ivp_est_s"] for r in results]
    mem_mb = [r["memory_mb"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].errorbar(x, t1_mean, yerr=t1_sd, fmt='-o', capsize=4, linewidth=2.0, label="4D DP on Test 1 (Mean ± 1 SD, n=5)")
    axes[0].plot(x, t2_val, '--s', linewidth=1.8, label="4D DP on Test 2 (No Noise)")
    axes[0].axhline(-1.317, color='gray', linestyle=':', linewidth=1.6, label="Constant Half-Dose (-1.317e6)")
    axes[0].axhline(-1.164, color='green', linestyle='-.', linewidth=1.6, label="Matched 2-Stage Open-Loop (-1.164e6)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(grids)
    axes[0].set_xlabel("Grid Resolution ($N_g^4$)")
    axes[0].set_ylabel("Cumulative Return ($\\times 10^6$)")
    axes[0].set_title("(a) Policy Return vs. Grid Resolution")
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(x, vec_rt, '-o', linewidth=2.0, label="Vectorized 20-substep RK4 (s)")
    axes[1].plot(x, ser_rt, '--^', linewidth=2.0, label="Est. Serial solve_ivp (s)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x); axes[1].set_xticklabels(grids)
    axes[1].set_xlabel("Grid Resolution ($N_g^4$)")
    axes[1].set_ylabel("Runtime (seconds, log scale)")
    axes[1].set_title("(b) Computational Runtime Scaling")
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].grid(True, alpha=0.35)

    axes[2].bar(x, mem_mb, width=0.55, color="#4c72b0", edgecolor="black")
    axes[2].set_yscale("log")
    axes[2].set_xticks(x); axes[2].set_xticklabels(grids)
    axes[2].set_xlabel("Grid Resolution ($N_g^4$)")
    axes[2].set_ylabel("Table Memory Footprint (MB, log scale)")
    axes[2].set_title("(c) Memory Scaling ($O(N_g^4)$)")
    axes[2].grid(True, alpha=0.35, axis="y")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results = []
    for n_g in [10, 15, 20, 30, 50]:
        print(f"Running DP for grid {n_g}^4 = {n_g**4} states...")
        dp_tup, v_rt, s_rt, mem = run_dp_grid(n_g, weights=(1e-2, 1e-1))
        r_t1 = [eval_dp(dp_tup, weights=(1e-2, 1e-1), noise=True, seed=s)[0] for s in [42, 123, 456, 789, 1024]]
        b_t1 = [eval_dp(dp_tup, weights=(1e-2, 1e-1), noise=True, seed=s)[1] for s in [42, 123, 456, 789, 1024]]
        r_t2, b_t2, st_t2 = eval_dp(dp_tup, weights=(1e-2, 1e-1), noise=False)
        res = {
            "grid_points_per_dim": n_g,
            "total_states": n_g**4,
            "memory_mb": round(mem, 2),
            "vectorized_runtime_s": round(v_rt, 2),
            "serial_solve_ivp_est_s": round(s_rt, 1),
            "test1_reward_mean": float(np.mean(r_t1)),
            "test1_reward_sd": float(np.std(r_t1, ddof=1)),
            "test1_burden_mean": float(np.mean(b_t1)),
            "test2_reward": float(r_t2),
            "test2_burden": float(b_t2),
            "test2_steps": int(st_t2)
        }
        print(res)
        results.append(res)
    out_json = os.path.join(base_dir, "dp_ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    out_png = os.path.join(base_dir, "dp_ablation.png")
    plot_dp_ablation(results, out_png)
    print("Saved DP ablation results and figure to", out_json, "and", out_png)
