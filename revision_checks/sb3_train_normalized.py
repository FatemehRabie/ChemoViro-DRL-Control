"""Proposed corrected training set-up.  (NOT RUN during the review - it needs your RL environment.)

Requires the same packages as ../run_multiseed_drl.py (gymnasium, torch, stable-baselines3, sb3-contrib).

Why: in ../run_multiseed_drl.py the agents see raw states (1e2 ... 1e7) and must output raw doses in
[0, 1000] x [0, 100]. With Stable-Baselines3 defaults that leaves
  - PPO/TRPO sampling doses from a Gaussian centred near 0 with standard deviation 1, i.e. they only ever try
    doses of 0-3 units out of 1000;
  - TD3/SAC with tanh output layers driven far into saturation by the raw inputs, where the gradient is zero;
  - TD3 with no exploration noise at all (action_noise defaults to None);
  - per-step rewards of order -1e4 to -1e5.
The audit in 01_policy_audit.py shows the resulting policies are "no dosing" or fixed corners of the action box.

What this script changes (and nothing else about the benchmark):
  1. actions:       the agent acts in [-1, 1]^2, mapped affinely to [0, 1000] x [0, 100];
  2. observations:  log10(1 + x) / 8, so every state component is O(1);
  3. rewards:       divided by --reward-scale during training only; all reported returns are in the original units;
  4. TD3:           Gaussian exploration noise;
  5. discount:      one fixed gamma close to 1 (the objective is an undiscounted sum over ~45 daily steps);
  6. integrator:    40 RK4 sub-steps per day in the 4D training environment (see 04_solver_crosscheck.py);
  7. bookkeeping:   models are saved, actions are archived for every seed, and each run is compared with the
                    no-dosing and corner policies so a degenerate policy is flagged immediately.

Evaluation is done by the repository's own `eval_model_on_env` (raw observations in, physical doses out) through a
thin adapter, on the same tests and episode seeds, so the numbers are directly comparable with the manuscript's tables.

The hyperparameters below are sensible starting values, not tuned ones. Check the learning curves
(results/multiseed_results_normalized.json) rise above the no-dosing level and compare the final returns with the
numerical optimum from 02_optimal_control.py (Test 2: -1.1496e6) before trusting any run.

Run (from this folder):
    python sb3_train_normalized.py --models 2 --algos SAC --seeds 42            # a quick first look
    python sb3_train_normalized.py --workers 16                                 # full 4 x 4 x 5 benchmark
Out:  results/multiseed_results_normalized.json (same schema as ../multiseed_results.json), results/models_normalized/*.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import TRPO  # noqa: E402
from stable_baselines3 import PPO, SAC, TD3  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402

import run_multiseed_drl as R  # noqa: E402

U_MAX = np.array([1000.0, 100.0])
OBS_SCALE = 8.0  # log10(1 + 1e8) = 8, the largest value the environment allows

TRAINING_SCENARIOS = [
    {"name": "Model 1 (Full, Default, Noise)", "ode_model": "full", "params": "default", "noise": "default"},
    {"name": "Model 2 (Reduced, Default, Noise)", "ode_model": "reduced", "params": "default", "noise": "default"},
    {"name": "Model 3 (Reduced, Modified, Noise)", "ode_model": "reduced", "params": "modified", "noise": "default"},
    {"name": "Model 4 (Reduced, Modified, No Noise)", "ode_model": "reduced", "params": "modified", "noise": "none"},
]
TESTING_SCENARIOS = [
    {"name": "Test 1 (Reduced, Default, Noise)", "ode_model": "reduced", "params": "default", "noise": "default"},
    {"name": "Test 2 (Reduced, Default, No Noise)", "ode_model": "reduced", "params": "default", "noise": "none"},
    {"name": "Test 3 (Full, Modified, Noise)", "ode_model": "full", "params": "modified", "noise": "default"},
    {"name": "Test 4 (Full, Modified, Zero-Lysis Control)", "ode_model": "full", "params": "modified_no_lysis", "noise": "default"},
]


def normalize_obs(obs):
    return (np.log10(1.0 + np.maximum(np.asarray(obs, dtype=np.float64), 0.0)) / OBS_SCALE).astype(np.float32)


def to_dose(action):
    """[-1, 1]^2 -> [0, 1000] x [0, 100]."""
    return (np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) + 1.0) * 0.5 * U_MAX


class TrainEnv(R.ChemoViroEnv):
    """The benchmark's training environment with 40 instead of 20 RK4 sub-steps per day in the 4D model."""

    def step(self, action):
        if self.use_solve_ivp or self.ode_model != "reduced":
            return super().step(action)
        u1 = float(np.clip(action[0], 0.0, 1000.0))
        u2 = float(np.clip(action[1], 0.0, 100.0))
        self.state = R.rk4_step_reduced(self.state, u1, u2, self.ode_params, self.dt, n_sub=40)
        self.time += self.dt
        tumor = float(self.state[0] + self.state[1])
        reward = -tumor - (0.5 * self.A1 * u1 * u1 + 0.5 * self.A2 * u2 * u2)
        terminated = bool(self.time >= self.max_time or tumor < 0.1)
        return self._get_obs(), float(reward), terminated, False, {"tumor_burden": tumor}


class NormalizedEnv(gym.Wrapper):
    """Agent-facing view: actions in [-1, 1]^2, log-scaled observations, scaled rewards."""

    def __init__(self, env, reward_scale):
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
        return normalize_obs(obs), reward / self.reward_scale, terminated, truncated, info


class PhysicalUnitsPolicy:
    """Lets the repository's evaluation code (raw observations in, physical doses out) drive a model trained on NormalizedEnv."""

    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(normalize_obs(obs), deterministic=deterministic)
        return to_dose(action).astype(np.float32), None


class RawReturnCurve(BaseCallback):
    """Mean of the last 100 training-episode returns in original reward units, every `every` steps."""

    def __init__(self, monitor, every=1000):
        super().__init__(verbose=0)
        self.monitor, self.every = monitor, every
        self.timesteps, self.rewards = [], []

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0:
            rets = self.monitor.get_episode_rewards()
            if len(rets) > 0:
                self.rewards.append(float(np.mean(rets[-100:])))
                self.timesteps.append(int(self.num_timesteps))
        return True


def train_and_evaluate_job(job):
    torch.set_num_threads(1)
    algo, seed, weights = job["algo"], job["seed"], job["reward_weights"]
    monitor = Monitor(TrainEnv(ode_model=job["ode_model"], ode_params=R.ODE_PARAMS[job["params"]],
                               noise_config=R.NOISE_CONFIG[job["noise"]], reward_weights=weights, use_solve_ivp=False))
    env = NormalizedEnv(monitor, job["reward_scale"])
    net = [job["net"], job["net"]]
    policy_kwargs = dict(net_arch=dict(pi=net, qf=net)) if algo in ("TD3", "SAC") else dict(net_arch=dict(pi=net, vf=net))
    common = dict(policy="MlpPolicy", env=env, verbose=0, seed=seed, device="cpu", gamma=job["gamma"], policy_kwargs=policy_kwargs)
    if algo == "PPO":
        model = PPO(**common, learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=10)
        steps = job["steps_on_policy"]
    elif algo == "TRPO":
        model = TRPO(**common, learning_rate=1e-3, n_steps=1024, batch_size=128)
        steps = job["steps_on_policy"]
    elif algo == "TD3":
        model = TD3(**common, learning_rate=1e-3, learning_starts=2000,
                    action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.2 * np.ones(2)))
        steps = job["steps_off_policy"]
    elif algo == "SAC":
        model = SAC(**common, learning_rate=3e-4, learning_starts=2000)
        steps = job["steps_off_policy"]
    else:
        raise ValueError(algo)

    cb = RawReturnCurve(monitor)
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb)
    train_time = time.time() - t0

    out_dir = HERE / "results" / "models_normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{job['scenario_name'].split(' (')[0].replace(' ', '')}_{algo}_seed{seed}"
    model.save(str(out_dir / tag))

    policy = PhysicalUnitsPolicy(model)
    model_dim = 4 if job["ode_model"] == "reduced" else 6
    test_evals = {t["name"]: R.eval_model_on_env(policy, model_dim, t, weights, n_episodes=10, base_seed=seed * 100)
                  for t in TESTING_SCENARIOS}                      # actions_sample is kept for every seed

    rng = np.random.default_rng(999)                               # the manuscript's ten perturbed parameter sets
    base_p = R.ODE_PARAMS["default"]
    param_rewards, param_burdens = [], []
    for p_idx in range(10):
        sp = base_p.copy()
        sp["alpha"] = float(rng.uniform(0.8 * base_p["alpha"], 1.2 * base_p["alpha"]))
        sp["beta"] = float(max(0.005, rng.normal(base_p["beta"], 0.1 * base_p["beta"])))
        sp["delta_U"] = float(rng.uniform(0.85 * base_p["delta_U"], 1.15 * base_p["delta_U"]))
        sp["delta_I"] = float(rng.uniform(0.85 * base_p["delta_I"], 1.15 * base_p["delta_I"]))
        sp["psi"] = float(max(0.5, rng.normal(base_p["psi"], 0.1 * base_p["psi"])))
        sp["gamma"] = float(max(0.001, rng.normal(base_p["gamma"], 0.1 * base_p["gamma"])))
        p_env = R.ChemoViroEnv(ode_model="reduced", ode_params=sp, noise_config=R.NOISE_CONFIG["default"],
                               reward_weights=weights, use_solve_ivp=True)
        obs, _ = p_env.reset(seed=5000 + p_idx)
        done, tot = False, 0.0
        while not done:
            obs, r, done, _, _ = p_env.step(R.predict_with_projection(policy, obs, model_dim, 4))
            tot += r
        param_rewards.append(tot)
        param_burdens.append(float(p_env.state[0] + p_env.state[1]))

    return {"scenario_name": job["scenario_name"], "algo": algo, "seed": seed, "train_time": train_time, "train_steps": steps,
            "curve_timesteps": cb.timesteps, "curve_rewards": cb.rewards, "test_evals": test_evals,
            "param_unc_reward_mean": float(np.mean(param_rewards)), "param_unc_reward_std": float(np.std(param_rewards, ddof=1)),
            "param_unc_burden_mean": float(np.mean(param_burdens))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--algos", nargs="*", default=["SAC", "TD3", "PPO", "TRPO"])
    ap.add_argument("--seeds", type=int, nargs="*", default=R.SEEDS)
    ap.add_argument("--steps-on-policy", type=int, default=300_000, help="PPO / TRPO environment steps")
    ap.add_argument("--steps-off-policy", type=int, default=60_000, help="TD3 / SAC environment steps")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--reward-scale", type=float, default=1e4)
    ap.add_argument("--net", type=int, default=128, help="hidden units per layer (two layers)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    jobs = []
    for algo in args.algos:
        for scen in TRAINING_SCENARIOS:
            if not any(scen["name"].startswith(f"Model {m} ") for m in args.models):
                continue
            for seed in args.seeds:
                jobs.append({"scenario_name": scen["name"], "ode_model": scen["ode_model"], "params": scen["params"], "noise": scen["noise"],
                             "algo": algo, "seed": seed, "reward_weights": (1e-2, 1e-1), "steps_on_policy": args.steps_on_policy,
                             "steps_off_policy": args.steps_off_policy, "gamma": args.gamma, "reward_scale": args.reward_scale, "net": args.net})
    print(f"Launching {len(jobs)} runs on {args.workers} worker(s) ...")
    t0 = time.time()
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            outs = pool.map(train_and_evaluate_job, jobs)
    else:
        outs = [train_and_evaluate_job(j) for j in jobs]
    print(f"Finished in {time.time() - t0:.0f} s")

    with open(REPO / "multiseed_results.json") as f:
        baselines = json.load(f)["baselines"]                      # same evaluation protocol, so they carry over unchanged
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "multiseed_results_normalized.json", "w") as f:
        json.dump({"rl_runs": outs, "baselines": baselines, "settings": vars(args)}, f, indent=1)

    # Degeneracy check against constant policies (uses the audit's numbers when they are available).
    consts = None
    audit = HERE / "results" / "01_policy_audit.json"
    if audit.exists():
        with open(audit) as f:
            consts = json.load(f)["constants"]
    t2 = "Test 2 (Reduced, Default, No Noise)"
    print("\nTest 2 (deterministic) returns. References: no dosing -1.5033e6, half dose -1.3187e6, two-stage -1.1651e6, numerical optimum -1.1496e6")
    for o in outs:
        r = o["test_evals"][t2]["reward_mean"]
        flag = ""
        if consts is not None:
            for name, per_test in consts.items():
                if abs(per_test[t2][str(o["seed"])] - r) / abs(r) < 1e-3:
                    flag = f"   <-- DEGENERATE: indistinguishable from the constant policy '{name}'"
        first, last = (o["curve_rewards"][0], o["curve_rewards"][-1]) if o["curve_rewards"] else (float("nan"), float("nan"))
        print(f"  {o['scenario_name'].split(' (')[0]} {o['algo']:4s} seed {o['seed']:4d}: Test 2 {r:.4e} | training return {first:.3e} -> {last:.3e}{flag}")
    print("wrote results/multiseed_results_normalized.json and results/models_normalized/")


if __name__ == "__main__":
    main()
