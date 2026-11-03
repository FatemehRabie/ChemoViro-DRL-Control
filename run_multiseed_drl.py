import os
import time
import json
import numpy as np
import scipy.stats as stats
from scipy.integrate import solve_ivp
import gymnasium as gym
import torch
from stable_baselines3 import TD3, SAC, PPO
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Pool

torch.set_num_threads(1)

SEEDS = [42, 123, 456, 789, 1024]

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
    }
}

NOISE_CONFIG = {
    "none": {"initial_conditions_noise": 0.0, "observations_noise": 0.0},
    "default": {"initial_conditions_noise": 0.05, "observations_noise": 0.01}
}

# Exact Optuna-selected hyperparameters for all 16 combinations
SELECTED_HYPERPARAMS = {
    "Model 1 (Full, Default, Noise)": {
        "TRPO": {"lr": 4.63e-5, "gamma": 0.9633, "net_arch_size": 512},
        "PPO":  {"lr": 6.09e-5, "gamma": 0.9601, "net_arch_size": 128, "n_steps": 512, "batch_size": 128, "n_epochs": 5, "clip_range": 0.289},
        "TD3":  {"lr": 1.08e-4, "gamma": 0.9241, "net_arch_size": 128},
        "SAC":  {"lr": 2.78e-5, "gamma": 0.9052, "net_arch_size": 512},
    },
    "Model 2 (Reduced, Default, Noise)": {
        "TRPO": {"lr": 4.49e-5, "gamma": 0.9014, "net_arch_size": 128},
        "PPO":  {"lr": 9.95e-4, "gamma": 0.9321, "net_arch_size": 256, "n_steps": 256, "batch_size": 128, "n_epochs": 5, "clip_range": 0.279},
        "TD3":  {"lr": 3.67e-4, "gamma": 0.9987, "net_arch_size": 512},
        "SAC":  {"lr": 2.48e-5, "gamma": 0.9001, "net_arch_size": 512},
    },
    "Model 3 (Reduced, Modified, Noise)": {
        "TRPO": {"lr": 1.12e-4, "gamma": 0.9718, "net_arch_size": 256},
        "PPO":  {"lr": 6.03e-5, "gamma": 0.9783, "net_arch_size": 128, "n_steps": 256, "batch_size": 128, "n_epochs": 10, "clip_range": 0.175},
        "TD3":  {"lr": 1.82e-5, "gamma": 0.9883, "net_arch_size": 128},
        "SAC":  {"lr": 7.56e-5, "gamma": 0.9003, "net_arch_size": 256},
    },
    "Model 4 (Reduced, Modified, No Noise)": {
        "TRPO": {"lr": 2.30e-4, "gamma": 0.9987, "net_arch_size": 256},
        "PPO":  {"lr": 8.80e-4, "gamma": 0.9995, "net_arch_size": 256, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "clip_range": 0.376},
        "TD3":  {"lr": 2.78e-4, "gamma": 0.9448, "net_arch_size": 128},
        "SAC":  {"lr": 6.13e-4, "gamma": 0.9872, "net_arch_size": 512},
    }
}

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

def rk4_step_reduced(state, u1, u2, p, dt=1.0, n_sub=20):
    h = dt / n_sub
    U, I, V, C = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    alpha, K, beta, Ku, delta_U, Kc, delta, delta_I, b, gamma, psi = (
        p["alpha"], p["K"], p["beta"], p["Ku"], p["delta_U"], p["Kc"],
        p["delta"], p["delta_I"], p["b"], p["gamma"], p["psi"]
    )
    bd = b * delta
    for _ in range(n_sub):
        # k1
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

        # k2
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

        # k3
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

        # k4
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
    def __init__(self, ode_model='reduced', ode_params=None, noise_config=None, max_time=400.0, dt=1.0, reward_weights=(1e-2, 1e-1), use_solve_ivp=False):
        super().__init__()
        self.ode_model = ode_model
        self.ode_params = ode_params or ODE_PARAMS["default"]
        self.noise_config = noise_config or NOISE_CONFIG["none"]
        self.max_time = max_time
        self.dt = dt
        self.A1, self.A2 = reward_weights
        self.use_solve_ivp = use_solve_ivp
        if self.ode_model == 'reduced':
            self.state_dim = 4
            high_obs = np.array([1e6, 1e6, 1e5, 1e5], dtype=np.float32)
        else:
            self.state_dim = 6
            high_obs = np.array([1e6, 1e6, 1e5, 1e5, 1e5, 1e5], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.zeros(self.state_dim, dtype=np.float32), high=high_obs, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32), high=np.array([1000.0, 100.0], dtype=np.float32), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0.0
        base = np.array([10000.0, 100.0, 500.0, 100.0]) if self.ode_model == 'reduced' else np.array([10000.0, 100.0, 500.0, 100.0, 100.0, 100.0])
        noise = self.noise_config.get("initial_conditions_noise", 0.0)
        if noise > 0.0:
            self.state = base * (1.0 + noise * self.np_random.standard_normal(self.state_dim))
        else:
            self.state = base.copy()
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
                self.state = rk4_step_reduced(self.state, u1, u2, self.ode_params, self.dt)
            else:
                self.state = rk4_step_full(self.state, u1, u2, self.ode_params, self.dt)
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

class LearningCurveCallback(BaseCallback):
    def __init__(self, eval_freq=1000):
        super().__init__(verbose=0)
        self.eval_freq = eval_freq
        self.timesteps = []
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                self.rewards.append(float(np.mean([ep['r'] for ep in self.model.ep_info_buffer])))
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

def eval_model_on_env(model, model_dim, test_cfg, reward_weights, n_episodes=20, base_seed=10000):
    is_noisy = test_cfg["noise"] != "none"
    n_ep = n_episodes if is_noisy else 1
    env = ChemoViroEnv(
        ode_model=test_cfg["ode_model"],
        ode_params=ODE_PARAMS[test_cfg["params"]],
        noise_config=NOISE_CONFIG[test_cfg["noise"]],
        reward_weights=reward_weights,
        use_solve_ivp=True
    )
    ep_rewards, ep_burdens, ep_lengths, ep_fracs = [], [], [], []
    first_actions = None
    for ep in range(n_ep):
        obs, _ = env.reset(seed=base_seed + ep)
        init_burden = float(env.state[0] + env.state[1])
        done = False
        tot_r = 0.0
        steps = 0
        acts = []
        while not done:
            act = predict_with_projection(model, obs, model_dim, env.state_dim)
            obs, r, done, _, info = env.step(act)
            tot_r += r
            steps += 1
            if ep == 0:
                acts.append(act)
        term_burden = float(env.state[0] + env.state[1])
        ep_rewards.append(tot_r)
        ep_burdens.append(term_burden)
        ep_lengths.append(steps)
        ep_fracs.append(term_burden / init_burden)
        if ep == 0:
            first_actions = np.array(acts)
    return {
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std_ep": float(np.std(ep_rewards)),
        "terminal_burden": float(np.mean(ep_burdens)),
        "residual_fraction": float(np.mean(ep_fracs)),
        "steps": float(np.mean(ep_lengths)),
        "actions_sample": first_actions.tolist() if first_actions is not None else []
    }

def train_and_evaluate_job(job):
    torch.set_num_threads(1)
    scen_name = job["scenario_name"]
    ode_model = job["ode_model"]
    params_key = job["params"]
    noise_key = job["noise"]
    algo = job["algo"]
    seed = job["seed"]
    reward_weights = job["reward_weights"]
    total_steps = job["total_steps"]

    hp = SELECTED_HYPERPARAMS[scen_name][algo]
    env = Monitor(ChemoViroEnv(
        ode_model=ode_model,
        ode_params=ODE_PARAMS[params_key],
        noise_config=NOISE_CONFIG[noise_key],
        reward_weights=reward_weights,
        use_solve_ivp=False
    ))
    net_sz = hp["net_arch_size"]
    if algo in ["TD3", "SAC"]:
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
        model = TRPO(**common_kw)
    elif algo == "PPO":
        model = PPO(**common_kw, n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], clip_range=hp["clip_range"])
    elif algo == "TD3":
        model = TD3(**common_kw, train_freq=(10, "step"), gradient_steps=2, learning_starts=1000)
    elif algo == "SAC":
        model = SAC(**common_kw, train_freq=(10, "step"), gradient_steps=2, learning_starts=1000)

    cb = LearningCurveCallback(eval_freq=1000)
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb)
    train_time = time.time() - t0

    model_dim = 4 if ode_model == "reduced" else 6
    testing_scenarios = [
        {"name": "Test 1 (Reduced, Default, Noise)", "ode_model": "reduced", "params": "default", "noise": "default"},
        {"name": "Test 2 (Reduced, Default, No Noise)", "ode_model": "reduced", "params": "default", "noise": "none"},
        {"name": "Test 3 (Full, Modified, Noise)", "ode_model": "full", "params": "modified", "noise": "default"},
        {"name": "Test 4 (Full, Modified, Zero-Lysis Control)", "ode_model": "full", "params": "modified_no_lysis", "noise": "default"},
    ]
    test_evals = {}
    for t_cfg in testing_scenarios:
        res = eval_model_on_env(model, model_dim, t_cfg, reward_weights, n_episodes=10, base_seed=seed * 100)
        if seed != SEEDS[0]:
            res.pop("actions_sample", None)
        test_evals[t_cfg["name"]] = res

    # Also evaluate parametric uncertainty across 10 sampled parameter sets on Reduced Default
    rng = np.random.default_rng(999)
    param_rewards, param_burdens = [], []
    base_p = ODE_PARAMS["default"]
    for p_idx in range(10):
        sampled_p = base_p.copy()
        sampled_p["alpha"] = float(rng.uniform(0.8 * base_p["alpha"], 1.2 * base_p["alpha"]))
        sampled_p["beta"] = float(max(0.005, rng.normal(base_p["beta"], 0.1 * base_p["beta"])))
        sampled_p["delta_U"] = float(rng.uniform(0.85 * base_p["delta_U"], 1.15 * base_p["delta_U"]))
        sampled_p["delta_I"] = float(rng.uniform(0.85 * base_p["delta_I"], 1.15 * base_p["delta_I"]))
        sampled_p["psi"] = float(max(0.5, rng.normal(base_p["psi"], 0.1 * base_p["psi"])))
        sampled_p["gamma"] = float(max(0.001, rng.normal(base_p["gamma"], 0.1 * base_p["gamma"])))
        p_env = ChemoViroEnv(ode_model="reduced", ode_params=sampled_p, noise_config=NOISE_CONFIG["default"], reward_weights=reward_weights, use_solve_ivp=True)
        obs, _ = p_env.reset(seed=5000 + p_idx)
        done = False
        tot_r = 0.0
        while not done:
            act = predict_with_projection(model, obs, model_dim, 4)
            obs, r, done, _, _ = p_env.step(act)
            tot_r += r
        param_rewards.append(tot_r)
        param_burdens.append(float(p_env.state[0] + p_env.state[1]))

    return {
        "scenario_name": scen_name,
        "algo": algo,
        "seed": seed,
        "train_time": train_time,
        "curve_timesteps": cb.timesteps,
        "curve_rewards": cb.rewards,
        "test_evals": test_evals,
        "param_unc_reward_mean": float(np.mean(param_rewards)),
        "param_unc_reward_std": float(np.std(param_rewards, ddof=1)),
        "param_unc_burden_mean": float(np.mean(param_burdens))
    }

if __name__ == "__main__":
    training_scenarios = [
        {"name": "Model 1 (Full, Default, Noise)", "ode_model": "full", "params": "default", "noise": "default"},
        {"name": "Model 2 (Reduced, Default, Noise)", "ode_model": "reduced", "params": "default", "noise": "default"},
        {"name": "Model 3 (Reduced, Modified, Noise)", "ode_model": "reduced", "params": "modified", "noise": "default"},
        {"name": "Model 4 (Reduced, Modified, No Noise)", "ode_model": "reduced", "params": "modified", "noise": "none"},
    ]
    jobs = []
    # Schedule SAC and TD3 first so longest jobs start immediately
    for algo in ["SAC", "TD3", "PPO", "TRPO"]:
        for scen in training_scenarios:
            for seed in SEEDS:
                jobs.append({
                    "scenario_name": scen["name"],
                    "ode_model": scen["ode_model"],
                    "params": scen["params"],
                    "noise": scen["noise"],
                    "algo": algo,
                    "seed": seed,
                    "reward_weights": (1e-2, 1e-1),
                    "total_steps": 50000
                })

    print(f"Launching {len(jobs)} multi-seed jobs across 16 CPU cores...")
    t_start = time.time()
    with Pool(processes=16) as pool:
        all_out = pool.map(train_and_evaluate_job, jobs)
    print(f"Completed all {len(jobs)} jobs in {time.time() - t_start:.1f}s!")

    # Evaluate Baselines on the same 5 seed cohorts and 10 parametric uncertainty sets
    class CallableBaseline:
        def __init__(self, fn):
            self.fn = fn
            self.step_idx = 0
        def reset_ep(self):
            self.step_idx = 0
        def predict(self, obs, deterministic=True):
            act = self.fn(self.step_idx, obs)
            self.step_idx += 1
            return np.array(act, dtype=np.float32), None

    baselines = {
        "Constant Half-Dose Baseline (500, 50)": lambda t, o: (500.0, 50.0),
        "Matched Open-Loop Constant (600, 100)": lambda t, o: (600.0, 100.0),
        "Matched Open-Loop 2-Stage": lambda t, o: (1000.0, 100.0) if t < 10 else (0.0, 100.0),
    }
    testing_scenarios = [
        {"name": "Test 1 (Reduced, Default, Noise)", "ode_model": "reduced", "params": "default", "noise": "default"},
        {"name": "Test 2 (Reduced, Default, No Noise)", "ode_model": "reduced", "params": "default", "noise": "none"},
        {"name": "Test 3 (Full, Modified, Noise)", "ode_model": "full", "params": "modified", "noise": "default"},
        {"name": "Test 4 (Full, Modified, Zero-Lysis Control)", "ode_model": "full", "params": "modified_no_lysis", "noise": "default"},
    ]
    baseline_out = []
    for b_name, b_fn in baselines.items():
        for seed in SEEDS:
            b_evals = {}
            for t_cfg in testing_scenarios:
                is_noisy = t_cfg["noise"] != "none"
                n_ep = 10 if is_noisy else 1
                env = ChemoViroEnv(
                    ode_model=t_cfg["ode_model"],
                    ode_params=ODE_PARAMS[t_cfg["params"]],
                    noise_config=NOISE_CONFIG[t_cfg["noise"]],
                    reward_weights=(1e-2, 1e-1),
                    use_solve_ivp=True
                )
                ep_r, ep_b, ep_s, ep_f = [], [], [], []
                for ep in range(n_ep):
                    obs, _ = env.reset(seed=seed * 100 + ep)
                    init_b = float(env.state[0] + env.state[1])
                    done = False
                    tot_r = 0.0
                    st = 0
                    while not done:
                        act = b_fn(st, obs)
                        obs, r, done, _, _ = env.step(act)
                        tot_r += r
                        st += 1
                    term_b = float(env.state[0] + env.state[1])
                    ep_r.append(tot_r); ep_b.append(term_b); ep_s.append(st); ep_f.append(term_b / init_b)
                b_evals[t_cfg["name"]] = {
                    "reward_mean": float(np.mean(ep_r)),
                    "terminal_burden": float(np.mean(ep_b)),
                    "residual_fraction": float(np.mean(ep_f)),
                    "steps": float(np.mean(ep_s))
                }
            # Parametric uncertainty
            rng = np.random.default_rng(999)
            param_rewards, param_burdens = [], []
            base_p = ODE_PARAMS["default"]
            for p_idx in range(10):
                sampled_p = base_p.copy()
                sampled_p["alpha"] = float(rng.uniform(0.8 * base_p["alpha"], 1.2 * base_p["alpha"]))
                sampled_p["beta"] = float(max(0.005, rng.normal(base_p["beta"], 0.1 * base_p["beta"])))
                sampled_p["delta_U"] = float(rng.uniform(0.85 * base_p["delta_U"], 1.15 * base_p["delta_U"]))
                sampled_p["delta_I"] = float(rng.uniform(0.85 * base_p["delta_I"], 1.15 * base_p["delta_I"]))
                sampled_p["psi"] = float(max(0.5, rng.normal(base_p["psi"], 0.1 * base_p["psi"])))
                sampled_p["gamma"] = float(max(0.001, rng.normal(base_p["gamma"], 0.1 * base_p["gamma"])))
                p_env = ChemoViroEnv(ode_model="reduced", ode_params=sampled_p, noise_config=NOISE_CONFIG["default"], reward_weights=(1e-2, 1e-1), use_solve_ivp=True)
                obs, _ = p_env.reset(seed=5000 + p_idx)
                done = False; tot_r = 0.0; st = 0
                while not done:
                    act = b_fn(st, obs)
                    obs, r, done, _, _ = p_env.step(act)
                    tot_r += r; st += 1
                param_rewards.append(tot_r)
                param_burdens.append(float(p_env.state[0] + p_env.state[1]))
            baseline_out.append({
                "baseline_name": b_name,
                "seed": seed,
                "test_evals": b_evals,
                "param_unc_reward_mean": float(np.mean(param_rewards)),
                "param_unc_reward_std": float(np.std(param_rewards, ddof=1)),
                "param_unc_burden_mean": float(np.mean(param_burdens))
            })

    out_path = r"C:\Users\USER\.gemini\antigravity\brain\7ff53ad6-4cd2-4b51-a639-b826042b038e\scratch\multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump({"rl_runs": all_out, "baselines": baseline_out}, f, indent=2)
    print("Saved results to", out_path)
