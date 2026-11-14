"""Shared helpers for the revision checks.

Everything here needs only numpy / scipy / matplotlib, so the checks run without Gymnasium,
PyTorch or Stable-Baselines3.

`Env` is a Gym-free replica of `ChemoViroEnv(use_solve_ivp=True)` in ../run_multiseed_drl.py:
same right-hand sides, same solve_ivp call (RK45, rtol = atol = 1e-6), same clipping, same reward,
same termination rule, and the same random stream (Gymnasium's `reset(seed=s)` builds
`np.random.Generator(PCG64(s))`, which is what `np.random.default_rng(s)` returns).
`01_policy_audit.py` verifies that it reproduces the stored baseline numbers in
../multiseed_results.json to machine precision before it is used for anything else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
U_MAX = (1000.0, 100.0)

ODE_PARAMS = {
    "default": {
        "alpha": 0.206, "K": 1e6, "beta": 0.05, "Ku": 1e5, "delta_U": 50.0, "Kc": 1e5,
        "delta": 0.5115, "delta_I": 60.0, "b": 50, "gamma": 0.01, "psi": 4.17,
        "nu_U": 0.08, "nu_I": 0.1, "tau": 0.2, "delta_V": 0.01, "delta_T": 0.01,
        "phi": 0.7, "beta_T": 0.5, "kappa": 1e5,
    },
    "modified": {
        "alpha": 0.5, "K": 1e8, "beta": 0.1, "Ku": 1000.0, "delta_U": 25.0, "Kc": 1e7,
        "delta": 0.7117, "delta_I": 70.0, "b": 100, "gamma": 0.03, "psi": 7.17,
        "nu_U": 0.08, "nu_I": 0.1, "tau": 0.2, "delta_V": 0.01, "delta_T": 0.01,
        "phi": 0.7, "beta_T": 0.5, "kappa": 1e5,
    },
}
ODE_PARAMS["modified_no_lysis"] = dict(ODE_PARAMS["modified"], nu_U=0.0, nu_I=0.0, tau=0.0)

NOISE = {"none": (0.0, 0.0), "default": (0.05, 0.01)}  # (initial-condition sigma, observation sigma)

# name -> (ode model, parameter key, noise key); identical to the manuscript's Tests 1-4
TESTS = {
    "Test 1 (Reduced, Default, Noise)": ("reduced", "default", "default"),
    "Test 2 (Reduced, Default, No Noise)": ("reduced", "default", "none"),
    "Test 3 (Full, Modified, Noise)": ("full", "modified", "default"),
    "Test 4 (Full, Modified, Zero-Lysis Control)": ("full", "modified_no_lysis", "default"),
}
MODELS = {
    "Model 1 (Full, Default, Noise)": ("full", "default", "default"),
    "Model 2 (Reduced, Default, Noise)": ("reduced", "default", "default"),
    "Model 3 (Reduced, Modified, Noise)": ("reduced", "modified", "default"),
    "Model 4 (Reduced, Modified, No Noise)": ("reduced", "modified", "none"),
}
ALGOS = ["PPO", "TRPO", "TD3", "SAC"]
REGIME_A = (1e-2, 1e-1)
REGIME_B = (1e-7, 1e-5)
NOMINAL_IC = {"reduced": [1e4, 100.0, 500.0, 100.0], "full": [1e4, 100.0, 500.0, 100.0, 100.0, 100.0]}


# ----------------------------------------------------------------------------------------------
# Scalar right-hand sides (same arithmetic as run_multiseed_drl.py)
# ----------------------------------------------------------------------------------------------
def reduced_rhs(t, y, action, p):
    U = y[0] if y[0] > 0.0 else 0.0
    I = y[1] if y[1] > 0.0 else 0.0
    V = y[2] if y[2] > 0.0 else 0.0
    C = y[3] if y[3] > 0.0 else 0.0
    u1, u2 = float(action[0]), float(action[1])
    inf = (p["beta"] * U * V) / (p["Ku"] + U)
    kc_c = p["Kc"] + C
    return [
        p["alpha"] * U * (1.0 - (U + I) / p["K"]) - inf - (p["delta_U"] * U * C) / kc_c,
        inf - p["delta"] * I - (p["delta_I"] * I * C) / kc_c,
        p["b"] * p["delta"] * I - inf - p["gamma"] * V + u1,
        u2 - p["psi"] * C,
    ]


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
    return [
        p["alpha"] * U * (1.0 - (U + I) / p["K"]) - inf - p["nu_U"] * U * E_T - (p["delta_U"] * U * C) / kc_c,
        inf - p["delta"] * I - p["nu_I"] * E_T * I - p["tau"] * E_V * I - (p["delta_I"] * I * C) / kc_c,
        p["b"] * p["delta"] * I - inf - p["gamma"] * V + u1,
        p["phi"] * I - p["delta_V"] * E_V,
        p["beta_T"] * (U + I) / (p["kappa"] + U + I) - p["delta_T"] * E_T,
        u2 - p["psi"] * C,
    ]


# ----------------------------------------------------------------------------------------------
# Evaluation environment (Gym-free replica)
# ----------------------------------------------------------------------------------------------
class Env:
    def __init__(self, ode_model="reduced", params=None, noise=(0.0, 0.0), weights=REGIME_A,
                 max_time=400.0, dt=1.0, base=None, threshold=0.1):
        self.ode_model = ode_model
        self.p = params if params is not None else ODE_PARAMS["default"]
        self.ic_noise, self.obs_noise = noise
        self.A1, self.A2 = weights
        self.max_time, self.dt, self.threshold = max_time, dt, threshold
        self.dim = 4 if ode_model == "reduced" else 6
        self.base = np.array(base if base is not None else NOMINAL_IC[ode_model], dtype=float)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        if self.ic_noise > 0.0:
            self.state = self.base * (1.0 + self.ic_noise * self.rng.standard_normal(self.dim))
        else:
            self.state = self.base.copy()
        self.state = np.clip(self.state, 0.0, 1e8)
        return self._obs()

    def _obs(self):
        obs = self.state.copy()
        if self.obs_noise > 0.0:
            obs = obs * (1.0 + self.obs_noise * self.rng.standard_normal(self.dim))
        return np.clip(obs, 0.0, 1e8).astype(np.float32)

    def step(self, action):
        u1 = float(np.clip(action[0], 0.0, U_MAX[0]))
        u2 = float(np.clip(action[1], 0.0, U_MAX[1]))
        rhs = reduced_rhs if self.ode_model == "reduced" else full_rhs
        sol = solve_ivp(rhs, (self.time, self.time + self.dt), self.state, args=((u1, u2), self.p),
                        method="RK45", rtol=1e-6, atol=1e-6)
        self.state = np.clip(sol.y[:, -1], 0.0, 1e8)
        self.time += self.dt
        tumor = float(self.state[0] + self.state[1])
        pen = 0.5 * self.A1 * u1 * u1 + 0.5 * self.A2 * u2 * u2
        done = bool(self.time >= self.max_time or tumor < self.threshold)
        return self._obs(), -tumor - pen, done, {"tumor": tumor, "pen": pen}


def run_episode(env, policy, seed=None, record=False):
    """policy(step_index, observation) -> (u1, u2). Returns the episode summary."""
    obs = env.reset(seed=seed)
    init = float(env.state[0] + env.state[1])
    total = tumor_cost = dose_cost = 0.0
    steps = 0
    traj, acts = ([env.state.copy()], []) if record else (None, None)
    done = False
    while not done:
        a = policy(steps, obs)
        obs, r, done, info = env.step(a)
        total += r
        tumor_cost += info["tumor"]
        dose_cost += info["pen"]
        steps += 1
        if record:
            traj.append(env.state.copy())
            acts.append((float(a[0]), float(a[1])))
    out = {"reward": total, "tumor_cost": tumor_cost, "dose_cost": dose_cost, "steps": steps,
           "terminal": float(env.state[0] + env.state[1]), "init": init}
    if record:
        out["traj"] = np.array(traj)
        out["acts"] = np.array(acts)
    return out


def eval_on_test(policy, test_name, seed, weights=REGIME_A, n_ep=10):
    """Same protocol as the manuscript: episode seeds are 100*seed + ep; 10 episodes if noisy, 1 if not."""
    model, pkey, nkey = TESTS[test_name]
    env = Env(model, ODE_PARAMS[pkey], NOISE[nkey], weights)
    n = n_ep if nkey != "none" else 1
    eps = [run_episode(env, policy, seed=seed * 100 + ep) for ep in range(n)]
    return {"reward_mean": float(np.mean([e["reward"] for e in eps])),
            "terminal_burden": float(np.mean([e["terminal"] for e in eps])),
            "steps": float(np.mean([e["steps"] for e in eps]))}


def seed_means(policy, test_name, weights=REGIME_A):
    return np.array([eval_on_test(policy, test_name, s, weights)["reward_mean"] for s in SEEDS])


def sampled_parameter_sets(n=10, rng_seed=999):
    """The manuscript's M = 10 perturbed parameter sets (Eq. 17), same generator and draw order."""
    rng = np.random.default_rng(rng_seed)
    base = ODE_PARAMS["default"]
    sets = []
    for _ in range(n):
        sp = base.copy()
        sp["alpha"] = float(rng.uniform(0.8 * base["alpha"], 1.2 * base["alpha"]))
        sp["beta"] = float(max(0.005, rng.normal(base["beta"], 0.1 * base["beta"])))
        sp["delta_U"] = float(rng.uniform(0.85 * base["delta_U"], 1.15 * base["delta_U"]))
        sp["delta_I"] = float(rng.uniform(0.85 * base["delta_I"], 1.15 * base["delta_I"]))
        sp["psi"] = float(max(0.5, rng.normal(base["psi"], 0.1 * base["psi"])))
        sp["gamma"] = float(max(0.001, rng.normal(base["gamma"], 0.1 * base["gamma"])))
        sets.append(sp)
    return sets


# ----------------------------------------------------------------------------------------------
# Open-loop reference schedules used in the manuscript
# ----------------------------------------------------------------------------------------------
def const_policy(u1, u2):
    return lambda t, o: (u1, u2)


def two_stage_policy(t, o):
    return (1000.0, 100.0) if t < 10 else (0.0, 100.0)


CONSTANTS = {
    "no dosing (0, 0)": (0.0, 0.0),
    "virus only (1000, 0)": (1000.0, 0.0),
    "drug only (0, 100)": (0.0, 100.0),
    "full dose (1000, 100)": (1000.0, 100.0),
}


# ----------------------------------------------------------------------------------------------
# Vectorised model (batch over the last axis) for optimisation and dynamic programming
# ----------------------------------------------------------------------------------------------
def rhs_batch(Y, u1, u2, p, full):
    Y = np.maximum(Y, 0.0)
    if full:
        U, I, V, EV, ET, C = Y
    else:
        U, I, V, C = Y
    inf = p["beta"] * U * V / (p["Ku"] + U)
    kc = p["Kc"] + C
    dU = p["alpha"] * U * (1.0 - (U + I) / p["K"]) - inf - p["delta_U"] * U * C / kc
    dI = inf - p["delta"] * I - p["delta_I"] * I * C / kc
    dV = p["b"] * p["delta"] * I - inf - p["gamma"] * V + u1
    dC = u2 - p["psi"] * C
    if full:
        dU = dU - p["nu_U"] * U * ET
        dI = dI - p["nu_I"] * ET * I - p["tau"] * EV * I
        dEV = p["phi"] * I - p["delta_V"] * EV
        dET = p["beta_T"] * (U + I) / (p["kappa"] + U + I) - p["delta_T"] * ET
        return np.array([dU, dI, dV, dEV, dET, dC])
    return np.array([dU, dI, dV, dC])


def rk4_day_batch(Y, u1, u2, p, full, n_sub=20):
    """Advance a batch of states by one day with classical RK4 (n_sub equal sub-steps)."""
    h = 1.0 / n_sub
    for _ in range(n_sub):
        k1 = rhs_batch(Y, u1, u2, p, full)
        k2 = rhs_batch(Y + 0.5 * h * k1, u1, u2, p, full)
        k3 = rhs_batch(Y + 0.5 * h * k2, u1, u2, p, full)
        k4 = rhs_batch(Y + h * k3, u1, u2, p, full)
        Y = np.maximum(Y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0)
    return Y


# ----------------------------------------------------------------------------------------------
# Stored results and small output helpers
# ----------------------------------------------------------------------------------------------
def load_multiseed():
    with open(REPO / "multiseed_results.json") as f:
        return json.load(f)


def rl_runs(data, model_name, algo):
    runs = [r for r in data["rl_runs"] if r["scenario_name"] == model_name and r["algo"] == algo]
    runs.sort(key=lambda r: SEEDS.index(r["seed"]))
    assert [r["seed"] for r in runs] == SEEDS, (model_name, algo)
    return runs


def save_json(name, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    path = RESULTS / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
    return path


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------
# Figure style (validated categorical palette, recessive axes, text in ink colours)
# ----------------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"PPO": "#2a78d6", "TRPO": "#eb6834", "TD3": "#1baf7a", "SAC": "#eda100"}  # fixed order
ACCENT = "#2a78d6"
DEEMPH = "#b9b8b1"


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def new_figure(nrows=1, ncols=1, figsize=(8, 4.5), **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.labelsize": 10, "legend.frameon": False,
                         "figure.dpi": 100, "savefig.dpi": 200, "axes.titlelocation": "left"})
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes
