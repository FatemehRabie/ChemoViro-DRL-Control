import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

CONFIG_PARAMS = {
    "default": {
        "alpha": 0.206, "K": 1e6, "beta": 0.05, "Ku": 1e5, "delta_U": 50.0,
        "Kc": 1e5, "delta": 0.5115, "delta_I": 60.0, "b": 50,
        "gamma": 0.01, "psi": 4.17, "nu_U": 0.08, "nu_I": 0.1,
        "tau": 0.2, "delta_V": 0.01, "delta_T": 0.01, "phi": 0.7,
        "beta_T": 0.5, "kappa": 1e5
    },
    "modified": {
        "alpha": 0.5, "K": 1e8, "beta": 0.1, "Ku": 1000.0, "delta_U": 25.0,
        "Kc": 1e7, "delta": 0.7117, "delta_I": 70.0, "b": 100,
        "gamma": 0.03, "psi": 7.17, "nu_U": 0.08, "nu_I": 0.1, "tau": 0.2,
        "delta_V": 0.01, "delta_T": 0.01, "phi": 0.7, "beta_T": 0.5,
        "kappa": 1e5
    },
    "modified_no_lysis": {
        "alpha": 0.5, "K": 1e8, "beta": 0.1, "Ku": 1000.0, "delta_U": 25.0,
        "Kc": 1e7, "delta": 0.7117, "delta_I": 70.0, "b": 100,
        "gamma": 0.03, "psi": 7.17, "nu_U": 0.0, "nu_I": 0.0, "tau": 0.0,
        "delta_V": 0.01, "delta_T": 0.01, "phi": 0.7, "beta_T": 0.5,
        "kappa": 1e5
    }
}

def reduced_rhs(t, y, action, p):
    U = y[0] if y[0] > 0.0 else 0.0
    I = y[1] if y[1] > 0.0 else 0.0
    V = y[2] if y[2] > 0.0 else 0.0
    C = y[3] if y[3] > 0.0 else 0.0
    u1, u2 = float(action[0]), float(action[1])
    inf = (p["beta"] * U * V) / (p["Ku"] + U)
    kc_c = p["Kc"] + C
    dUdt = p["alpha"] * U * (1.0 - (U + I) / p["K"]) - inf - (p["delta_U"] * U * C) / kc_c
    dIdt = inf - p["delta"] * I - (p["delta_I"] * I * C) / kc_c
    dVdt = p["b"] * p["delta"] * I - inf - p["gamma"] * V + u1
    dCdt = u2 - p["psi"] * C
    return [dUdt, dIdt, dVdt, dCdt]

def full_rhs(t, y, action, p):
    U = y[0] if y[0] > 0.0 else 0.0
    I = y[1] if y[1] > 0.0 else 0.0
    V = y[2] if y[2] > 0.0 else 0.0
    E_V = y[3] if y[3] > 0.0 else 0.0
    E_T = y[4] if y[4] > 0.0 else 0.0
    C = y[5] if y[5] > 0.0 else 0.0
    u1, u2 = float(action[0]), float(action[1])
    inf = (p["beta"] * U * V) / (p["Ku"] + U)
    kc_c = p["Kc"] + C
    dUdt = p["alpha"] * U * (1.0 - (U + I) / p["K"]) - inf - p["nu_U"] * U * E_T - (p["delta_U"] * U * C) / kc_c
    dIdt = inf - p["delta"] * I - p["nu_I"] * E_T * I - p["tau"] * E_V * I - (p["delta_I"] * I * C) / kc_c
    dVdt = p["b"] * p["delta"] * I - inf - p["gamma"] * V + u1
    dE_Vdt = p["phi"] * I - p["delta_V"] * E_V
    dE_Tdt = p["beta_T"] * (U + I) / (p["kappa"] + U + I) - p["delta_T"] * E_T
    dCdt = u2 - p["psi"] * C
    return [dUdt, dIdt, dVdt, dE_Vdt, dE_Tdt, dCdt]

def eval_schedule(schedule_fn, model_type, params, weights, seed=None, noise=False, max_steps=400):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng(42)
    base = np.array([10000.0, 100.0, 500.0, 100.0]) if model_type == 'reduced' else np.array([10000.0, 100.0, 500.0, 100.0, 100.0, 100.0])
    if noise:
        y = base * (1.0 + 0.05 * rng.standard_normal(len(base)))
    else:
        y = base.copy()
    rhs = reduced_rhs if model_type == 'reduced' else full_rhs
    A1, A2 = weights
    total_r = 0.0
    for step in range(max_steps):
        obs = y * (1.0 + 0.01 * rng.standard_normal(len(y))) if noise else y.copy()
        u1, u2 = schedule_fn(step, obs)
        u1 = float(np.clip(u1, 0.0, 1000.0))
        u2 = float(np.clip(u2, 0.0, 100.0))
        sol = solve_ivp(rhs, (0.0, 1.0), y, args=((u1, u2), params), method='RK45', rtol=1e-6, atol=1e-6)
        y = np.clip(sol.y[:, -1], 0.0, 1e8)
        tumor = float(y[0] + y[1])
        r = -tumor - (0.5 * A1 * u1 * u1 + 0.5 * A2 * u2 * u2)
        total_r += r
        if tumor < 0.1:
            return total_r, tumor, step + 1
    return total_r, float(y[0] + y[1]), max_steps

# 1. Grid-search matched open-loop constant and piecewise-constant policies on Reduced Default (No Noise)
for w_label, weights in [("High-Penalty (A1=1e-2, A2=1e-1)", (1e-2, 1e-1)), ("Low-Penalty (A1=1e-7, A2=1e-5)", (1e-7, 1e-5))]:
    print(f"\n==================== {w_label} ====================")
    best_r, best_u = -np.inf, None
    for u1 in np.linspace(0, 1000, 21):
        for u2 in np.linspace(0, 100, 21):
            r, tf_b, st = eval_schedule(lambda t, o: (u1, u2), 'reduced', CONFIG_PARAMS['default'], weights, noise=False)
            if r > best_r:
                best_r, best_u = r, (u1, u2)
    print(f"Best constant open-loop on Default 4D: u*={best_u}, R={best_r:.4e}")

    # Two-stage piecewise constant search: stage 1 for t < t_sw, stage 2 for t >= t_sw
    best_pw_r, best_pw = -np.inf, None
    for t_sw in [5, 10, 15, 20, 25, 30, 35]:
        for u1_a in [0, 250, 500, 750, 1000]:
            for u2_a in [0, 25, 50, 75, 100]:
                for u1_b in [0, 100, 250, 500, 1000]:
                    for u2_b in [0, 25, 50, 100]:
                        r, tf_b, st = eval_schedule(lambda t, o: (u1_a, u2_a) if t < t_sw else (u1_b, u2_b), 'reduced', CONFIG_PARAMS['default'], weights, noise=False)
                        if r > best_pw_r:
                            best_pw_r, best_pw = r, (t_sw, u1_a, u2_a, u1_b, u2_b)
    print(f"Best 2-stage open-loop on Default 4D: params={best_pw}, R={best_pw_r:.4e}")
