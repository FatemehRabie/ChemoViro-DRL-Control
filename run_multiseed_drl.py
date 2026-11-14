import os
import time
import json
import itertools
import numpy as np
import scipy.stats as stats
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import gymnasium as gym
import torch
from stable_baselines3 import TD3, SAC, PPO
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from multiprocessing import Pool

torch.set_num_threads(1)

SEEDS = [42, 123, 456, 789, 1024]
U_MAX = np.array([1000.0, 100.0], dtype=np.float64)
OBS_SCALE = 8.0        # log10(1 + 1e8) = 8.0 maps [0, 1e8] -> [0, 1]
REWARD_SCALE = 1e4     # scales training rewards only; all logged/evaluated returns are in exact physical units

ODE_PARAMS = {
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
    },
    "coupled_6d": {
        "alpha": 0.206, "K": 1e6, "beta": 0.05, "Ku": 1e5, "delta_U": 50.0,
        "Kc": 1e5, "delta": 0.5115, "delta_I": 60.0, "b": 50,
        "gamma": 0.01, "psi": 4.17, "nu_U": 0.03, "nu_I": 0.04,
        "tau": 0.04, "delta_V": 0.01, "delta_T": 0.01, "phi": 0.001,
        "beta_T": 0.05, "kappa": 1e5
    }
}

INITIAL_CONDITIONS = {
    "nominal_4d": [10000.0, 100.0, 500.0, 100.0],
    "nominal_6d": [10000.0, 100.0, 500.0, 100.0, 100.0, 100.0],
    "virus_free_4d": [10000.0, 0.0, 0.0, 0.0],
    "virus_free_6d": [10000.0, 0.0, 0.0, 0.0, 0.5, 0.0],
}

NOISE_CONFIG = {
    "none": {"initial_conditions_noise": 0.0, "observations_noise": 0.0},
    "default": {"initial_conditions_noise": 0.05, "observations_noise": 0.01}
}

# Standardized normalized continuous-control hyperparameters across all scenarios
ALGO_HYPERPARAMS = {
    "PPO":  {"lr": 3e-4, "gamma": 0.995, "net_arch_size": 128, "n_steps": 1024, "batch_size": 128, "n_epochs": 10, "clip_range": 0.2, "train_steps": 250000},
    "TRPO": {"lr": 1e-3, "gamma": 0.995, "net_arch_size": 128, "n_steps": 1024, "batch_size": 128, "train_steps": 250000},
    "TD3":  {"lr": 1e-3, "gamma": 0.995, "net_arch_size": 128, "batch_size": 256, "learning_starts": 2000, "train_freq": 2, "gradient_steps": 1, "action_noise_sigma": 0.2, "train_steps": 80000},
    "SAC":  {"lr": 3e-4, "gamma": 0.995, "net_arch_size": 128, "batch_size": 256, "learning_starts": 2000, "train_freq": 2, "gradient_steps": 1, "train_steps": 80000},
}

TRAINING_SCENARIOS = [
    {"name": "Model 1 (Full, Default, Noise)", "ode_model": "full", "params": "default", "ic": "nominal_6d", "noise": "default", "domain_rand": False},
    {"name": "Model 2 (Reduced, Default, Noise)", "ode_model": "reduced", "params": "default", "ic": "nominal_4d", "noise": "default", "domain_rand": False},
    {"name": "Model 3 (Reduced, Modified, Noise)", "ode_model": "reduced", "params": "modified", "ic": "nominal_4d", "noise": "default", "domain_rand": False},
    {"name": "Model 4 (Reduced, Modified, No Noise)", "ode_model": "reduced", "params": "modified", "ic": "nominal_4d", "noise": "none", "domain_rand": False},
    {"name": "Model 5 (Full Coupled, Virus-Free, Noise)", "ode_model": "full", "params": "coupled_6d", "ic": "virus_free_6d", "noise": "default", "domain_rand": False},
    {"name": "Model 6 (Reduced DR, Virus-Free, Noise)", "ode_model": "reduced", "params": "default", "ic": "virus_free_4d", "noise": "default", "domain_rand": True},
]

TESTING_SCENARIOS = [
    {"name": "Test 1 (Reduced, Default, Noise)", "ode_model": "reduced", "params": "default", "ic": "nominal_4d", "noise": "default"},
    {"name": "Test 2 (Reduced, Default, No Noise)", "ode_model": "reduced", "params": "default", "ic": "nominal_4d", "noise": "none"},
    {"name": "Test 3 (Full, Modified, Noise)", "ode_model": "full", "params": "modified", "ic": "nominal_6d", "noise": "default"},
    {"name": "Test 4 (Full, Modified, Zero-Lysis Control)", "ode_model": "full", "params": "modified_no_lysis", "ic": "nominal_6d", "noise": "default"},
    {"name": "Test 5 (Reduced, Default, Virus-Free, Noise)", "ode_model": "reduced", "params": "default", "ic": "virus_free_4d", "noise": "default"},
    {"name": "Test 6 (Full Coupled, Virus-Free, Noise)", "ode_model": "full", "params": "coupled_6d", "ic": "virus_free_6d", "noise": "default"},
]

def reduced_rhs_scalar(t, y, action, p):
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
        u2 - p["psi"] * C
    ]

def full_rhs_scalar(t, y, action, p):
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
        u2 - p["psi"] * C
    ]

def rk4_step_reduced(state, u1, u2, p, dt=1.0, n_sub=40):
    h = dt / n_sub
    U, I, V, C = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    alpha, K, beta, Ku, delta_U, Kc, delta, delta_I, b, gamma, psi = (
        p["alpha"], p["K"], p["beta"], p["Ku"], p["delta_U"], p["Kc"],
        p["delta"], p["delta_I"], p["b"], p["gamma"], p["psi"]
    )
    bd = b * delta
    for _ in range(n_sub):
        u_pos = U if U > 0.0 else 0.0
        i_pos = I if I > 0.0 else 0.0
        v_pos = V if V > 0.0 else 0.0
        c_pos = C if C > 0.0 else 0.0
        inf1 = (beta * u_pos * v_pos) / (Ku + u_pos)
        kcc1 = Kc + c_pos
        k1U = alpha * u_pos * (1.0 - (u_pos + i_pos) / K) - inf1 - (delta_U * u_pos * c_pos) / kcc1
        k1I = inf1 - delta * i_pos - (delta_I * i_pos * c_pos) / kcc1
        k1V = bd * i_pos - inf1 - gamma * v_pos + u1
        k1C = u2 - psi * c_pos

        u_pos = max(0.0, U + 0.5 * h * k1U)
        i_pos = max(0.0, I + 0.5 * h * k1I)
        v_pos = max(0.0, V + 0.5 * h * k1V)
        c_pos = max(0.0, C + 0.5 * h * k1C)
        inf2 = (beta * u_pos * v_pos) / (Ku + u_pos)
        kcc2 = Kc + c_pos
        k2U = alpha * u_pos * (1.0 - (u_pos + i_pos) / K) - inf2 - (delta_U * u_pos * c_pos) / kcc2
        k2I = inf2 - delta * i_pos - (delta_I * i_pos * c_pos) / kcc2
        k2V = bd * i_pos - inf2 - gamma * v_pos + u1
        k2C = u2 - psi * c_pos

        u_pos = max(0.0, U + 0.5 * h * k2U)
        i_pos = max(0.0, I + 0.5 * h * k2I)
        v_pos = max(0.0, V + 0.5 * h * k2V)
        c_pos = max(0.0, C + 0.5 * h * k2C)
        inf3 = (beta * u_pos * v_pos) / (Ku + u_pos)
        kcc3 = Kc + c_pos
        k3U = alpha * u_pos * (1.0 - (u_pos + i_pos) / K) - inf3 - (delta_U * u_pos * c_pos) / kcc3
        k3I = inf3 - delta * i_pos - (delta_I * i_pos * c_pos) / kcc3
        k3V = bd * i_pos - inf3 - gamma * v_pos + u1
        k3C = u2 - psi * c_pos

        u_pos = max(0.0, U + h * k3U)
        i_pos = max(0.0, I + h * k3I)
        v_pos = max(0.0, V + h * k3V)
        c_pos = max(0.0, C + h * k3C)
        inf4 = (beta * u_pos * v_pos) / (Ku + u_pos)
        kcc4 = Kc + c_pos
        k4U = alpha * u_pos * (1.0 - (u_pos + i_pos) / K) - inf4 - (delta_U * u_pos * c_pos) / kcc4
        k4I = inf4 - delta * i_pos - (delta_I * i_pos * c_pos) / kcc4
        k4V = bd * i_pos - inf4 - gamma * v_pos + u1
        k4C = u2 - psi * c_pos

        U = min(1e8, max(0.0, U + (h / 6.0) * (k1U + 2.0 * k2U + 2.0 * k3U + k4U)))
        I = min(1e8, max(0.0, I + (h / 6.0) * (k1I + 2.0 * k2I + 2.0 * k3I + k4I)))
        V = min(1e8, max(0.0, V + (h / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)))
        C = min(1e8, max(0.0, C + (h / 6.0) * (k1C + 2.0 * k2C + 2.0 * k3C + k4C)))
    return np.array([U, I, V, C], dtype=np.float64)

def rk4_step_full(state, u1, u2, p, dt=1.0, n_sub=40):
    h = dt / n_sub
    y = np.array(state, dtype=np.float64)
    for _ in range(n_sub):
        k1 = np.array(full_rhs_scalar(0.0, y, (u1, u2), p))
        k2 = np.array(full_rhs_scalar(0.0, np.maximum(0.0, y + 0.5 * h * k1), (u1, u2), p))
        k3 = np.array(full_rhs_scalar(0.0, np.maximum(0.0, y + 0.5 * h * k2), (u1, u2), p))
        k4 = np.array(full_rhs_scalar(0.0, np.maximum(0.0, y + h * k3), (u1, u2), p))
        y = np.clip(y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1e8)
    return y

class ChemoViroEnv(gym.Env):
    def __init__(self, ode_model='reduced', ode_params=None, noise_config=None, max_time=400.0, dt=1.0,
                 reward_weights=(1e-2, 1e-1), use_solve_ivp=False, initial_state=None, domain_rand=False):
        super().__init__()
        self.ode_model = ode_model
        self.base_ode_params = (ode_params or ODE_PARAMS["default"]).copy()
        self.ode_params = self.base_ode_params.copy()
        self.noise_config = noise_config or NOISE_CONFIG["none"]
        self.max_time = max_time
        self.dt = dt
        self.A1, self.A2 = reward_weights
        self.use_solve_ivp = use_solve_ivp
        self.domain_rand = domain_rand
        if self.ode_model == 'reduced':
            self.state_dim = 4
            default_ic = INITIAL_CONDITIONS["nominal_4d"]
            high_obs = np.array([1e8] * 4, dtype=np.float32)
        else:
            self.state_dim = 6
            default_ic = INITIAL_CONDITIONS["nominal_6d"]
            high_obs = np.array([1e8] * 6, dtype=np.float32)
        self.base_ic = np.array(initial_state if initial_state is not None else default_ic, dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=np.zeros(self.state_dim, dtype=np.float32), high=high_obs, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32), high=np.array([1000.0, 100.0], dtype=np.float32), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0.0
        if self.domain_rand and self.ode_model == 'reduced':
            sp = self.base_ode_params.copy()
            sp["alpha"] = float(self.np_random.uniform(0.16, 0.26))
            sp["beta"] = float(10 ** self.np_random.uniform(np.log10(0.035), np.log10(0.14)))
            sp["b"] = float(self.np_random.uniform(40.0, 120.0))
            sp["gamma"] = float(10 ** self.np_random.uniform(np.log10(0.01), np.log10(0.35)))
            sp["Ku"] = float(10 ** self.np_random.uniform(np.log10(1.5e4), np.log10(1.2e5)))
            sp["delta_U"] = float(self.np_random.uniform(35.0, 65.0))
            sp["delta_I"] = float(self.np_random.uniform(45.0, 75.0))
            U0 = float(10 ** self.np_random.uniform(np.log10(2000.0), np.log10(15000.0)))
            self.ode_params = sp
            self.state = np.array([U0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.ode_params = self.base_ode_params.copy()
            base = self.base_ic.copy()
            noise = self.noise_config.get("initial_conditions_noise", 0.0)
            if noise > 0.0:
                self.state = base * (1.0 + noise * self.np_random.standard_normal(self.state_dim))
            else:
                self.state = base
        self.state = np.clip(self.state, 0.0, 1e8)
        return self._get_obs(), {}

    def step(self, action):
        u1 = float(np.clip(action[0], 0.0, 1000.0))
        u2 = float(np.clip(action[1], 0.0, 100.0))
        if self.use_solve_ivp:
            rhs = reduced_rhs_scalar if self.ode_model == 'reduced' else full_rhs_scalar
            sol = solve_ivp(rhs, (self.time, self.time + self.dt), self.state, args=((u1, u2), self.ode_params), method='RK45', rtol=1e-6, atol=1e-6)
            self.state = np.clip(sol.y[:, -1], 0.0, 1e8)
        else:
            if self.ode_model == 'reduced':
                self.state = rk4_step_reduced(self.state, u1, u2, self.ode_params, self.dt, n_sub=40)
            else:
                self.state = rk4_step_full(self.state, u1, u2, self.ode_params, self.dt, n_sub=40)
        self.time += self.dt
        tumor = float(self.state[0] + self.state[1])
        reward = -tumor - (0.5 * self.A1 * u1 * u1 + 0.5 * self.A2 * u2 * u2)
        terminated = bool(self.time >= self.max_time or tumor < 0.1)
        return self._get_obs(), float(reward), terminated, False, {"tumor_burden": tumor}

    def _get_obs(self):
        obs = self.state.copy()
        noise = self.noise_config.get("observations_noise", 0.0)
        if noise > 0.0:
            obs = obs * (1.0 + noise * self.np_random.standard_normal(self.state_dim))
        return np.clip(obs, 0.0, 1e8).astype(np.float32)

def normalize_obs(obs):
    return (np.log10(1.0 + np.maximum(np.asarray(obs, dtype=np.float64), 0.0)) / OBS_SCALE).astype(np.float32)

def to_dose(action):
    return (np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) + 1.0) * 0.5 * U_MAX

class NormalizedEnv(gym.Wrapper):
    """Agent-facing wrapper: actions in [-1, 1]^2, log-scaled O(1) observations, scaled training rewards."""
    def __init__(self, env, reward_scale=REWARD_SCALE):
        super().__init__(env)
        n = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.5, shape=(n,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.reward_scale = float(reward_scale)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return normalize_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(to_dose(action))
        return normalize_obs(obs), float(reward / self.reward_scale), terminated, truncated, info

class PhysicalUnitsPolicy:
    """Adapter taking raw physical observations and returning physical doses (u1, u2)."""
    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(normalize_obs(obs), deterministic=deterministic)
        return to_dose(action).astype(np.float32), None

class RawReturnCurveCallback(BaseCallback):
    """Records rolling mean return in exact physical units from the underlying Monitor wrapper."""
    def __init__(self, monitor, eval_freq=1000):
        super().__init__(verbose=0)
        self.monitor = monitor
        self.eval_freq = eval_freq
        self.timesteps = []
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rets = self.monitor.get_episode_rewards()
            if len(rets) > 0:
                self.rewards.append(float(np.mean(rets[-100:])))
                self.timesteps.append(int(self.num_timesteps))
        return True

def predict_with_projection(model, obs, model_dim, env_dim):
    if model_dim == env_dim:
        action, _ = model.predict(obs, deterministic=True)
    elif model_dim == 6 and env_dim == 4:
        padded = np.insert(obs, 3, [0.0, 0.0])
        action, _ = model.predict(padded, deterministic=True)
    elif model_dim == 4 and env_dim == 6:
        reduced = np.delete(obs, [3, 4])
        action, _ = model.predict(reduced, deterministic=True)
    else:
        raise ValueError("Dimension mismatch")
    return action

def get_narrow_param_sets(n=10, rng_seed=999):
    rng = np.random.default_rng(rng_seed)
    base_p = ODE_PARAMS["default"]
    sets = []
    for _ in range(n):
        sp = base_p.copy()
        sp["alpha"] = float(rng.uniform(0.8 * base_p["alpha"], 1.2 * base_p["alpha"]))
        sp["beta"] = float(max(0.005, rng.normal(base_p["beta"], 0.1 * base_p["beta"])))
        sp["delta_U"] = float(rng.uniform(0.85 * base_p["delta_U"], 1.15 * base_p["delta_U"]))
        sp["delta_I"] = float(rng.uniform(0.85 * base_p["delta_I"], 1.15 * base_p["delta_I"]))
        sp["psi"] = float(max(0.5, rng.normal(base_p["psi"], 0.1 * base_p["psi"])))
        sp["gamma"] = float(max(0.001, rng.normal(base_p["gamma"], 0.1 * base_p["gamma"])))
        sets.append(sp)
    return sets

def get_wide_heterogeneous_cohort(n=10, rng_seed=2026):
    rng = np.random.default_rng(rng_seed)
    base_p = ODE_PARAMS["default"]
    cohort = []
    for _ in range(n):
        sp = base_p.copy()
        sp["alpha"] = float(rng.uniform(0.16, 0.26))
        sp["beta"] = float(10 ** rng.uniform(np.log10(0.035), np.log10(0.14)))
        sp["b"] = float(rng.uniform(40.0, 120.0))
        sp["gamma"] = float(10 ** rng.uniform(np.log10(0.01), np.log10(0.35)))
        sp["Ku"] = float(10 ** rng.uniform(np.log10(1.5e4), np.log10(1.2e5)))
        sp["delta_U"] = float(rng.uniform(35.0, 65.0))
        sp["delta_I"] = float(rng.uniform(45.0, 75.0))
        U0 = float(10 ** rng.uniform(np.log10(2000.0), np.log10(15000.0)))
        cohort.append((np.array([U0, 0.0, 0.0, 0.0], dtype=np.float64), sp))
    return cohort

def eval_model_on_env(model, model_dim, test_cfg, reward_weights, n_episodes=10, base_seed=10000):
    is_noisy = test_cfg["noise"] != "none"
    n_ep = n_episodes if is_noisy else 1
    ic_key = test_cfg.get("ic", "nominal_4d" if test_cfg["ode_model"] == "reduced" else "nominal_6d")
    env = ChemoViroEnv(
        ode_model=test_cfg["ode_model"],
        ode_params=ODE_PARAMS[test_cfg["params"]],
        noise_config=NOISE_CONFIG[test_cfg["noise"]],
        reward_weights=reward_weights,
        use_solve_ivp=True,
        initial_state=INITIAL_CONDITIONS[ic_key]
    )
    ep_rewards, ep_burdens, ep_lengths, ep_fracs = [], [], [], []
    first_actions = None
    for ep in range(n_ep):
        if hasattr(model, "reset_ep"):
            model.reset_ep()
        obs, _ = env.reset(seed=base_seed + ep)
        init_burden = float(env.state[0] + env.state[1])
        done = False
        tot_r = 0.0
        steps = 0
        acts = []
        while not done:
            act = predict_with_projection(model, obs, model_dim, env.state_dim)
            obs, r, done, _, _ = env.step(act)
            tot_r += r
            steps += 1
            if ep == 0:
                acts.append([float(act[0]), float(act[1])])
        term_burden = float(env.state[0] + env.state[1])
        ep_rewards.append(tot_r)
        ep_burdens.append(term_burden)
        ep_lengths.append(steps)
        ep_fracs.append(term_burden / init_burden)
        if ep == 0:
            first_actions = acts
    return {
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std_ep": float(np.std(ep_rewards)),
        "terminal_burden": float(np.mean(ep_burdens)),
        "residual_fraction": float(np.mean(ep_fracs)),
        "steps": float(np.mean(ep_lengths)),
        "actions_sample": first_actions if first_actions is not None else []
    }

def evaluate_cohorts(policy, model_dim, reward_weights):
    # 1. Original narrow M=10 perturbed parameter sets (seed 5000 + p_idx)
    narrow_sets = get_narrow_param_sets(10, 999)
    param_rewards, param_burdens = [], []
    for p_idx, sp in enumerate(narrow_sets):
        if hasattr(policy, "reset_ep"):
            policy.reset_ep()
        p_env = ChemoViroEnv(ode_model="reduced", ode_params=sp, noise_config=NOISE_CONFIG["default"],
                             reward_weights=reward_weights, use_solve_ivp=True, initial_state=INITIAL_CONDITIONS["nominal_4d"])
        obs, _ = p_env.reset(seed=5000 + p_idx)
        done, tot_r = False, 0.0
        while not done:
            act = predict_with_projection(policy, obs, model_dim, 4)
            obs, r, done, _, _ = p_env.step(act)
            tot_r += r
        param_rewards.append(float(tot_r))
        param_burdens.append(float(p_env.state[0] + p_env.state[1]))

    # 2. Redesigned wide heterogeneous M=10 virus-free cohort (deterministic & noisy)
    wide_cohort = get_wide_heterogeneous_cohort(10, 2026)
    wide_rewards_det, wide_burdens_det, wide_steps_det = [], [], []
    wide_rewards_noisy = []
    for p_idx, (y0, sp) in enumerate(wide_cohort):
        if hasattr(policy, "reset_ep"):
            policy.reset_ep()
        w_env = ChemoViroEnv(ode_model="reduced", ode_params=sp, noise_config=NOISE_CONFIG["none"],
                             reward_weights=reward_weights, use_solve_ivp=True, initial_state=y0)
        obs, _ = w_env.reset(seed=6000 + p_idx)
        done, tot_r, st = False, 0.0, 0
        while not done:
            act = predict_with_projection(policy, obs, model_dim, 4)
            obs, r, done, _, _ = w_env.step(act)
            tot_r += r
            st += 1
        wide_rewards_det.append(float(tot_r))
        wide_burdens_det.append(float(w_env.state[0] + w_env.state[1]))
        wide_steps_det.append(int(st))

        if hasattr(policy, "reset_ep"):
            policy.reset_ep()
        wn_env = ChemoViroEnv(ode_model="reduced", ode_params=sp, noise_config=NOISE_CONFIG["default"],
                              reward_weights=reward_weights, use_solve_ivp=True, initial_state=y0)
        obs, _ = wn_env.reset(seed=7000 + p_idx)
        done, tot_rn = False, 0.0
        while not done:
            act = predict_with_projection(policy, obs, model_dim, 4)
            obs, r, done, _, _ = wn_env.step(act)
            tot_rn += r
        wide_rewards_noisy.append(float(tot_rn))

    return {
        "param_unc_reward_mean": float(np.mean(param_rewards)),
        "param_unc_reward_std": float(np.std(param_rewards, ddof=1)),
        "param_unc_burden_mean": float(np.mean(param_burdens)),
        "param_unc_patient_rewards": param_rewards,
        "wide_cohort_reward_mean": float(np.mean(wide_rewards_det)),
        "wide_cohort_reward_std": float(np.std(wide_rewards_det, ddof=1)),
        "wide_cohort_burden_mean": float(np.mean(wide_burdens_det)),
        "wide_cohort_steps_mean": float(np.mean(wide_steps_det)),
        "wide_cohort_patient_rewards": wide_rewards_det,
        "wide_cohort_noisy_reward_mean": float(np.mean(wide_rewards_noisy)),
        "wide_cohort_noisy_patient_rewards": wide_rewards_noisy,
    }

def train_and_evaluate_job(job):
    torch.set_num_threads(1)
    scen_name = job["scenario_name"]
    ode_model = job["ode_model"]
    params_key = job["params"]
    noise_key = job["noise"]
    ic_key = job.get("ic", "nominal_4d" if ode_model == "reduced" else "nominal_6d")
    domain_rand = job.get("domain_rand", False)
    algo = job["algo"]
    seed = job["seed"]
    reward_weights = job["reward_weights"]

    hp = ALGO_HYPERPARAMS[algo]
    total_steps = job["total_steps"] if "total_steps" in job else hp["train_steps"]
    save_model = job.get("save_model", total_steps > 0)

    raw_env = ChemoViroEnv(
        ode_model=ode_model,
        ode_params=ODE_PARAMS[params_key],
        noise_config=NOISE_CONFIG[noise_key],
        reward_weights=reward_weights,
        use_solve_ivp=False,
        initial_state=INITIAL_CONDITIONS[ic_key],
        domain_rand=domain_rand
    )
    monitor = Monitor(raw_env)
    env = NormalizedEnv(monitor, reward_scale=REWARD_SCALE)

    net_sz = hp["net_arch_size"]
    if algo in ("TD3", "SAC"):
        policy_kwargs = dict(net_arch=dict(pi=[net_sz, net_sz], qf=[net_sz, net_sz]))
    else:
        policy_kwargs = dict(net_arch=dict(pi=[net_sz, net_sz], vf=[net_sz, net_sz]))

    common_kw = dict(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=seed,
        device="cpu",
        gamma=hp["gamma"],
        learning_rate=hp["lr"],
        policy_kwargs=policy_kwargs
    )
    if algo == "TRPO":
        model = TRPO(**common_kw, n_steps=hp["n_steps"], batch_size=hp["batch_size"])
    elif algo == "PPO":
        model = PPO(**common_kw, n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], clip_range=hp["clip_range"])
    elif algo == "TD3":
        model = TD3(
            **common_kw,
            batch_size=hp["batch_size"],
            learning_starts=hp["learning_starts"],
            train_freq=(hp["train_freq"], "step"),
            gradient_steps=hp["gradient_steps"],
            action_noise=NormalActionNoise(mean=np.zeros(2), sigma=hp["action_noise_sigma"] * np.ones(2))
        )
    elif algo == "SAC":
        model = SAC(
            **common_kw,
            batch_size=hp["batch_size"],
            learning_starts=hp["learning_starts"],
            train_freq=(hp["train_freq"], "step"),
            gradient_steps=hp["gradient_steps"]
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    cb = RawReturnCurveCallback(monitor, eval_freq=1000)
    t0 = time.time()
    if total_steps > 0:
        model.learn(total_timesteps=total_steps, callback=cb)
    train_time = time.time() - t0

    if save_model:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models_normalized")
        os.makedirs(models_dir, exist_ok=True)
        tag = f"{scen_name.split(' (')[0].replace(' ', '')}_{algo}_seed{seed}"
        model.save(os.path.join(models_dir, tag))

    policy = PhysicalUnitsPolicy(model)
    model_dim = 4 if ode_model == "reduced" else 6
    test_evals = {}
    for t_cfg in TESTING_SCENARIOS:
        # Archive actions_sample for EVERY seed (fixing seed-42-only limitation)
        test_evals[t_cfg["name"]] = eval_model_on_env(policy, model_dim, t_cfg, reward_weights, n_episodes=10, base_seed=seed * 100)

    cohort_res = evaluate_cohorts(policy, model_dim, reward_weights)
    out = {
        "scenario_name": scen_name,
        "algo": algo,
        "seed": seed,
        "train_time": train_time,
        "train_steps": total_steps,
        "curve_timesteps": cb.timesteps,
        "curve_rewards": cb.rewards,
        "test_evals": test_evals,
    }
    out.update(cohort_res)
    return out


# ----------------------------------------------------------------------------------------------
# Direct single-shooting optimal control & log-spaced 4D DP helpers for reference rows
# ----------------------------------------------------------------------------------------------
def _rhs_batch(Y, u1, u2, p, full):
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

def _rk4_day_batch(Y, u1, u2, p, full, n_sub=40):
    h = 1.0 / n_sub
    for _ in range(n_sub):
        k1 = _rhs_batch(Y, u1, u2, p, full)
        k2 = _rhs_batch(Y + 0.5 * h * k1, u1, u2, p, full)
        k3 = _rhs_batch(Y + 0.5 * h * k2, u1, u2, p, full)
        k4 = _rhs_batch(Y + h * k3, u1, u2, p, full)
        Y = np.clip(Y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1e8)
    return Y

def solve_daily_ocp(y0, p, full, weights=(1e-2, 1e-1), N=65, n_sub=40, maxiter=250):
    A1, A2 = weights
    def fg(x, eps=1e-4):
        n = x.size
        X = np.repeat(x[None, :], 2 * n + 1, axis=0)
        hi = np.minimum(x + eps, 1.0)
        lo = np.maximum(x - eps, 0.0)
        idx = np.arange(n)
        X[1 + idx, idx] = hi
        X[1 + n + idx, idx] = lo
        B = X.shape[0]
        u1 = X[:, :N] * 1000.0
        u2 = X[:, N:] * 100.0
        Y = np.repeat(np.asarray(y0, dtype=float)[:, None], B, axis=1)
        c = np.zeros(B)
        for k in range(N):
            Y = _rk4_day_batch(Y, u1[:, k], u2[:, k], p, full, n_sub)
            c += Y[0] + Y[1] + 0.5 * A1 * u1[:, k] ** 2 + 0.5 * A2 * u2[:, k] ** 2
        return c[0], (c[1:1 + n] - c[1 + n:]) / (hi - lo)

    two10 = np.zeros(2 * N); two10[:min(10, N)] = 1.0; two10[N:] = 1.0
    two20 = np.zeros(2 * N); two20[:min(20, N)] = 1.0; two20[N:] = 1.0
    starts = [np.zeros(2 * N), np.full(2 * N, 0.5), two10, two20, np.ones(2 * N)]
    best = None
    for x0 in starts:
        res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * (2 * N),
                       options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-13, gtol=1e-9))
        if best is None or res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best[1]

def build_log_dp_4d(n, p, weights=(1e-2, 1e-1), zmax=(6.2, 6.2, 8.0), cmax=30.0, sweeps=90, n_sub=40):
    t0 = time.time()
    actions = [(a, b) for a in (0.0, 500.0, 1000.0) for b in (0.0, 50.0, 100.0)]
    grids = (np.linspace(0, zmax[0], n), np.linspace(0, zmax[1], n), np.linspace(0, zmax[2], n), np.linspace(0, cmax, n))
    ZU, ZI, ZV, CC = np.meshgrid(*grids, indexing="ij")
    Y0 = np.array([10 ** ZU.ravel() - 1.0, 10 ** ZI.ravel() - 1.0, 10 ** ZV.ravel() - 1.0, CC.ravel()])
    S = n ** 4
    A1, A2 = weights
    idxs, wts, stage, term = [], [], [], []
    for u1, u2 in actions:
        Y = _rk4_day_batch(Y0.copy(), u1, u2, p, False, n_sub)
        tumor = Y[0] + Y[1]
        stage.append(tumor + 0.5 * A1 * u1 ** 2 + 0.5 * A2 * u2 ** 2)
        term.append(tumor < 0.1)
        coords = [np.log10(1.0 + Y[0]), np.log10(1.0 + Y[1]), np.log10(1.0 + Y[2]), Y[3]]
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
        for a in range(len(actions)):
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
    return {
        "grids": grids, "pol": pol.reshape(n, n, n, n), "n": n, "actions": actions,
        "seconds": time.time() - t0, "sweeps": sweep + 1, "policy_bytes": int(pol.nbytes)
    }

class CallablePolicyAdapter:
    def __init__(self, fn):
        self.fn = fn
        self.step_idx = 0

    def reset_ep(self):
        self.step_idx = 0

    def predict(self, obs, deterministic=True):
        act = self.fn(self.step_idx, obs)
        self.step_idx += 1
        return np.array(act, dtype=np.float32), None


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reward_weights = (1e-2, 1e-1)

    jobs = []
    # Schedule off-policy and 6D jobs first so longest jobs start first on the worker pool
    for algo in ["SAC", "TD3", "PPO", "TRPO"]:
        for scen in TRAINING_SCENARIOS:
            for seed in SEEDS:
                jobs.append({
                    "scenario_name": scen["name"],
                    "ode_model": scen["ode_model"],
                    "params": scen["params"],
                    "ic": scen["ic"],
                    "noise": scen["noise"],
                    "domain_rand": scen["domain_rand"],
                    "algo": algo,
                    "seed": seed,
                    "reward_weights": reward_weights,
                    "save_model": True
                })

    print(f"Launching {len(jobs)} normalized multi-seed DRL jobs across 16 CPU cores...")
    t_start = time.time()
    with Pool(processes=16) as pool:
        all_out = pool.map(train_and_evaluate_job, jobs)
    print(f"Completed all {len(jobs)} DRL jobs in {time.time() - t_start:.1f}s!")

    # 1. Solve direct numerical optimal schedules (L-BFGS-B) for each test & cohort
    print("Computing direct numerical optimal control schedules (L-BFGS-B + RK45 verification)...")
    ocp_schedules = {}
    for t_cfg in TESTING_SCENARIOS:
        full = t_cfg["ode_model"] == "full"
        y0 = INITIAL_CONDITIONS[t_cfg["ic"]]
        p = ODE_PARAMS[t_cfg["params"]]
        N_hor = 15 if t_cfg["params"] == "modified" and full else 65
        x_opt = solve_daily_ocp(y0, p, full, weights=reward_weights, N=N_hor, n_sub=40, maxiter=250)
        ocp_schedules[t_cfg["name"]] = (x_opt, N_hor)

    # Clairvoyant per-patient optima on Narrow (M=10) and Wide (M=10) cohorts, plus Cohort-Optimal Open-Loop
    narrow_sets = get_narrow_param_sets(10, 999)
    narrow_clairvoyant_noisy = []
    for p_idx, sp in enumerate(narrow_sets):
        p_env = ChemoViroEnv(ode_model="reduced", ode_params=sp, noise_config=NOISE_CONFIG["default"],
                             reward_weights=reward_weights, use_solve_ivp=True, initial_state=INITIAL_CONDITIONS["nominal_4d"])
        _, _ = p_env.reset(seed=5000 + p_idx)
        y0_true = p_env.state.copy()
        xp = solve_daily_ocp(y0_true, sp, False, weights=reward_weights, N=60, n_sub=40, maxiter=180)
        pol = CallablePolicyAdapter(lambda t, o, x=xp: (x[t] * 1000.0, x[60 + t] * 100.0) if t < 60 else (0.0, 0.0))
        _, _ = p_env.reset(seed=5000 + p_idx)
        done, tr = False, 0.0
        while not done:
            act, _ = pol.predict(None)
            _, r, done, _, _ = p_env.step(act)
            tr += r
        narrow_clairvoyant_noisy.append(float(tr))

    wide_cohort = get_wide_heterogeneous_cohort(10, 2026)
    wide_clairvoyant_det, wide_clairvoyant_steps = [], []
    for p_idx, (y0, sp) in enumerate(wide_cohort):
        xp = solve_daily_ocp(y0, sp, False, weights=reward_weights, N=65, n_sub=40, maxiter=200)
        w_env = ChemoViroEnv(ode_model="reduced", ode_params=sp, noise_config=NOISE_CONFIG["none"],
                             reward_weights=reward_weights, use_solve_ivp=True, initial_state=y0)
        w_env.reset(seed=6000 + p_idx)
        done, tr, st = False, 0.0, 0
        while not done:
            act = (xp[st] * 1000.0, xp[65 + st] * 100.0) if st < 65 else (0.0, 0.0)
            _, r, done, _, _ = w_env.step(act)
            tr += r
            st += 1
        wide_clairvoyant_det.append(float(tr))
        wide_clairvoyant_steps.append(int(st))

    # Optimize Cohort-Optimal Single Open-Loop Schedule across the 10 Wide Cohort patients
    N_coh = 65
    def cohort_fg(x, eps=1e-4):
        n = x.size
        X = np.repeat(x[None, :], 2 * n + 1, axis=0)
        hi = np.minimum(x + eps, 1.0)
        lo = np.maximum(x - eps, 0.0)
        idx = np.arange(n)
        X[1 + idx, idx] = hi
        X[1 + n + idx, idx] = lo
        B = X.shape[0]
        u1 = X[:, :N_coh] * 1000.0
        u2 = X[:, N_coh:] * 100.0
        tot_c = np.zeros(B)
        A1, A2 = reward_weights
        for y0, sp in wide_cohort:
            Y = np.repeat(y0[:, None], B, axis=1)
            c = np.zeros(B)
            for k in range(N_coh):
                Y = _rk4_day_batch(Y, u1[:, k], u2[:, k], sp, False, 20)
                c += Y[0] + Y[1] + 0.5 * A1 * u1[:, k] ** 2 + 0.5 * A2 * u2[:, k] ** 2
            tot_c += c / len(wide_cohort)
        return tot_c[0], (tot_c[1:1 + n] - tot_c[1 + n:]) / (hi - lo)

    two10 = np.zeros(2 * N_coh); two10[:10] = 1.0; two10[N_coh:] = 1.0
    res_coh = minimize(cohort_fg, two10, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * (2 * N_coh), options=dict(maxiter=180))
    x_cohort_ol = res_coh.x.copy()

    # 2. Build Log-Spaced 4D DP tables (10^4, 15^4, 20^4) on Default 4D & Modified 4D
    print("Building log-spaced 4D DP policies (10^4, 15^4, 20^4)...")
    log_dp_tables = {}
    for n_g in (10, 15, 20):
        dp_d = build_log_dp_4d(n_g, ODE_PARAMS["default"], weights=reward_weights)
        log_dp_tables[f"Log-Grid 4D DP ({n_g}^4)"] = dp_d

    def make_log_dp_fn(dp_obj):
        g, n, pol, acts = dp_obj["grids"], dp_obj["n"], dp_obj["pol"], np.array(dp_obj["actions"])
        def f(t, obs):
            c = [np.log10(1.0 + max(float(obs[0]), 0.0)), np.log10(1.0 + max(float(obs[1]), 0.0)),
                 np.log10(1.0 + max(float(obs[2]), 0.0)), max(float(obs[3]), 0.0)]
            ii = [int(np.clip(round((ci - gi[0]) / (gi[1] - gi[0])), 0, n - 1)) for ci, gi in zip(c, g)]
            return tuple(acts[pol[tuple(ii)]])
        return f

    # 3. Evaluate all open-loop baselines, numerical optimal schedules, and log-spaced 4D DP (with 6->4 projection!)
    static_baselines = {
        "No Dosing (0, 0)": (lambda t, o: (0.0, 0.0), 4),
        "Constant Half-Dose Baseline (500, 50)": (lambda t, o: (500.0, 50.0), 4),
        "Matched Open-Loop Constant (600, 100)": (lambda t, o: (600.0, 100.0), 4),
        "Matched Open-Loop 2-Stage": (lambda t, o: (1000.0, 100.0) if t < 10 else (0.0, 100.0), 4),
    }
    for dp_name, dp_obj in log_dp_tables.items():
        static_baselines[dp_name] = (make_log_dp_fn(dp_obj), 4)

    baseline_out = []
    for b_name, (b_fn, b_dim) in static_baselines.items():
        cohort_stats = evaluate_cohorts(CallablePolicyAdapter(b_fn), b_dim, reward_weights)
        for seed in SEEDS:
            b_evals = {}
            for t_cfg in TESTING_SCENARIOS:
                res = eval_model_on_env(CallablePolicyAdapter(b_fn), b_dim, t_cfg, reward_weights, n_episodes=10, base_seed=seed * 100)
                b_evals[t_cfg["name"]] = res
            entry = {"baseline_name": b_name, "seed": seed, "test_evals": b_evals}
            entry.update(cohort_stats)
            baseline_out.append(entry)

    # Also evaluate the test-specific Numerical Optimal Open-Loop schedule across all seeds and tests
    opt_cohort_pol = CallablePolicyAdapter(lambda t, o: (x_cohort_ol[t] * 1000.0, x_cohort_ol[N_coh + t] * 100.0) if t < N_coh else (0.0, 0.0))
    # On narrow cohort use Test 2 nominal optimal schedule; on wide cohort use Cohort-Optimal Open-Loop schedule
    x_nom_def, N_nom_def = ocp_schedules["Test 2 (Reduced, Default, No Noise)"]
    nom_opt_pol = CallablePolicyAdapter(lambda t, o: (x_nom_def[t] * 1000.0, x_nom_def[N_nom_def + t] * 100.0) if t < N_nom_def else (0.0, 0.0))
    nom_cohort_stats = evaluate_cohorts(nom_opt_pol, 4, reward_weights)
    coh_ol_stats = evaluate_cohorts(opt_cohort_pol, 4, reward_weights)
    nom_cohort_stats["wide_cohort_reward_mean"] = coh_ol_stats["wide_cohort_reward_mean"]
    nom_cohort_stats["wide_cohort_reward_std"] = coh_ol_stats["wide_cohort_reward_std"]
    nom_cohort_stats["wide_cohort_burden_mean"] = coh_ol_stats["wide_cohort_burden_mean"]
    nom_cohort_stats["wide_cohort_steps_mean"] = coh_ol_stats["wide_cohort_steps_mean"]
    nom_cohort_stats["wide_cohort_patient_rewards"] = coh_ol_stats["wide_cohort_patient_rewards"]
    nom_cohort_stats["wide_cohort_noisy_reward_mean"] = coh_ol_stats["wide_cohort_noisy_reward_mean"]
    nom_cohort_stats["wide_cohort_noisy_patient_rewards"] = coh_ol_stats["wide_cohort_noisy_patient_rewards"]

    for seed in SEEDS:
        opt_evals = {}
        for t_cfg in TESTING_SCENARIOS:
            xt, Nt = ocp_schedules[t_cfg["name"]]
            t_dim = 4 if t_cfg["ode_model"] == "reduced" else 6
            pol_t = CallablePolicyAdapter(lambda t, o, x=xt, N=Nt: (x[t] * 1000.0, x[N + t] * 100.0) if t < N else (0.0, 0.0))
            opt_evals[t_cfg["name"]] = eval_model_on_env(pol_t, t_dim, t_cfg, reward_weights, n_episodes=10, base_seed=seed * 100)
        entry = {"baseline_name": "Numerical Optimum Open-Loop (L-BFGS-B)", "seed": seed, "test_evals": opt_evals}
        entry.update(nom_cohort_stats)
        baseline_out.append(entry)

    cohort_references = {
        "narrow_clairvoyant_noisy_rewards": narrow_clairvoyant_noisy,
        "narrow_clairvoyant_noisy_mean": float(np.mean(narrow_clairvoyant_noisy)),
        "narrow_clairvoyant_noisy_std": float(np.std(narrow_clairvoyant_noisy, ddof=1)),
        "wide_clairvoyant_det_rewards": wide_clairvoyant_det,
        "wide_clairvoyant_det_mean": float(np.mean(wide_clairvoyant_det)),
        "wide_clairvoyant_det_std": float(np.std(wide_clairvoyant_det, ddof=1)),
        "wide_clairvoyant_det_steps": wide_clairvoyant_steps,
        "log_dp_metadata": {k: {"states": v["n"] ** 4, "seconds": v["seconds"], "sweeps": v["sweeps"], "policy_bytes": v["policy_bytes"]}
                            for k, v in log_dp_tables.items()}
    }

    out_data = {
        "rl_runs": all_out,
        "baselines": baseline_out,
        "cohort_references": cohort_references,
        "hyperparams": ALGO_HYPERPARAMS
    }
    out_path = os.path.join(base_dir, "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    parent_out_path = os.path.join(os.path.dirname(base_dir), "multiseed_results.json")
    if os.path.isdir(os.path.dirname(base_dir)):
        with open(parent_out_path, "w") as f:
            json.dump(out_data, f, indent=2)
    print("Saved complete normalized multi-seed results to", out_path)
