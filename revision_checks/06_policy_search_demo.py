"""Is the task learnable by a model-free method once inputs and outputs are scaled?

This is NOT one of the manuscript's four algorithms and is not a substitute for re-running them. It is a
minimal proof of concept: a small neural feedback policy

    observation  ->  log10(1 + x) / 8  ->  16 tanh units  ->  2 sigmoids  ->  dose = sigmoid * (1000, 100)

trained with the cross-entropy method (a derivative-free policy search that only sees episode returns) in the
manuscript's Model 2 training setting (4D Default model, 5% initial-condition noise, 1% observation noise,
Regime A reward). The trained policy is then scored with the manuscript's evaluation protocol on Tests 1 and 2.

If a method this simple reaches the neighbourhood of the optimum, the 19-31% optimality gaps of the stored agents
reflect the training set-up (raw inputs, raw action scale), not the difficulty of the control problem.

Run:  python 06_policy_search_demo.py   (about 2 minutes; numpy only)
Out:  results/06_policy_search_demo.json, results/06_policy_search_demo.md
"""
from __future__ import annotations

import time

import numpy as np

from common import ODE_PARAMS, REGIME_A, RESULTS, TESTS, md_table, rk4_day_batch, save_json, seed_means

P = ODE_PARAMS["default"]
BASE = np.array([1e4, 100.0, 500.0, 100.0])
H = 16
N_PARAMS = H * 4 + H + 2 * H + 2
HORIZON = 60
T1, T2 = list(TESTS)[:2]


def unpack(theta):
    """theta: (B, N_PARAMS) -> weights with a leading batch axis."""
    B = theta.shape[0]
    i = 0
    W1 = theta[:, i:i + H * 4].reshape(B, H, 4); i += H * 4
    b1 = theta[:, i:i + H]; i += H
    W2 = theta[:, i:i + 2 * H].reshape(B, 2, H); i += 2 * H
    b2 = theta[:, i:i + 2]
    return W1, b1, W2, b2


def act(weights, obs):
    """obs: (4, B) raw observations -> doses (u1, u2), each (B,)."""
    W1, b1, W2, b2 = weights
    z = np.log10(1.0 + np.maximum(obs, 0.0)) / 8.0
    h = np.tanh(np.einsum("bij,jb->bi", W1, z) + b1)
    a = 1.0 / (1.0 + np.exp(-(np.einsum("bij,bj->bi", W2, h) + b2)))
    return 1000.0 * a[:, 0], 100.0 * a[:, 1]


def episode_cost(theta, y0, obs_noise_rng=None):
    """Closed-loop cost of each of the B policies from the initial state y0 (4,), with the episode's termination rule."""
    B = theta.shape[0]
    weights = unpack(theta)
    Y = np.repeat(y0[:, None], B, axis=1)
    alive = np.ones(B, dtype=bool)
    cost = np.zeros(B)
    for _ in range(HORIZON):
        obs = Y if obs_noise_rng is None else np.maximum(Y * (1.0 + 0.01 * obs_noise_rng.standard_normal(Y.shape)), 0.0)
        u1, u2 = act(weights, obs)
        Y = rk4_day_batch(Y, u1, u2, P, False, 20)
        tumor = Y[0] + Y[1]
        cost += alive * (tumor + 0.5 * REGIME_A[0] * u1 ** 2 + 0.5 * REGIME_A[1] * u2 ** 2)
        alive &= tumor >= 0.1
        if not alive.any():
            break
    return cost


def train(seed, generations=120, pop=64, n_elite=12, episodes_per_generation=2):
    rng = np.random.default_rng(seed)
    mu, sigma = np.zeros(N_PARAMS), np.ones(N_PARAMS)
    history, env_steps = [], 0
    for g in range(generations):
        theta = mu + sigma * rng.standard_normal((pop, N_PARAMS))
        cost = np.zeros(pop)
        for _ in range(episodes_per_generation):          # the same noisy episodes for the whole population
            y0 = np.clip(BASE * (1.0 + 0.05 * rng.standard_normal(4)), 0.0, 1e8)
            cost += episode_cost(theta, y0, rng)
        cost /= episodes_per_generation
        env_steps += pop * episodes_per_generation * 45  # ~45 daily steps per episode
        elite = theta[np.argsort(cost)[:n_elite]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.02)
        history.append(float(-episode_cost(mu[None, :], BASE)[0]))
    return mu, history, env_steps


def as_policy(theta):
    weights = unpack(theta[None, :])

    def f(t, obs):
        u1, u2 = act(weights, np.asarray(obs, dtype=float)[:, None])
        return float(u1[0]), float(u2[0])
    return f


if __name__ == "__main__":
    t0 = time.time()
    rows, out = [], {}
    for seed in (42, 123, 456, 789, 1024):
        theta, history, env_steps = train(seed)
        pol = as_policy(theta)
        r1, r2 = seed_means(pol, T1), seed_means(pol, T2)
        out[str(seed)] = {"test1_seed_cohort_means": r1, "test2": float(r2[0]), "history_nominal_return": history, "env_steps": env_steps}
        rows.append([seed, f"{history[0]:.4e}", f"{history[9]:.4e}", f"{history[29]:.4e}", f"{history[-1]:.4e}", f"{r1.mean():.4e}", f"{r2[0]:.4e}"])
        print(f"seed {seed}: nominal return after generation 1 / 10 / 30 / {len(history)}: {history[0]:.4e} / {history[9]:.4e} / {history[29]:.4e} / {history[-1]:.4e}"
              f" | exact Test 1 {r1.mean():.4e}, Test 2 {r2[0]:.4e}  ({env_steps:,} environment steps)", flush=True)
    t2_vals = np.array([out[s]["test2"] for s in out])
    t1_vals = np.array([np.mean(out[s]["test1_seed_cohort_means"]) for s in out])
    out["summary"] = {"test2_mean": t2_vals.mean(), "test2_sd": t2_vals.std(ddof=1), "test1_mean": t1_vals.mean(), "test1_sd": t1_vals.std(ddof=1),
                      "gap_to_optimum_test2_pct": 100 * (-1149609.8 - t2_vals.mean()) / 1149609.8}
    print(f"over 5 training seeds: Test 2 {t2_vals.mean():.4e} ± {t2_vals.std(ddof=1):.2e} (gap to the numerical optimum {out['summary']['gap_to_optimum_test2_pct']:.2f}%); "
          f"Test 1 {t1_vals.mean():.4e} ± {t1_vals.std(ddof=1):.2e}   [{time.time() - t0:.0f} s]")
    save_json("06_policy_search_demo.json", out)
    (RESULTS / "06_policy_search_demo.md").write_text(f"""# 06 - Model-free policy search with scaled inputs and outputs (proof of concept)

A 4-16-2 feedback network on log-scaled observations, trained with the cross-entropy method (population 64, 12 elites,
120 generations, two noisy training episodes per generation, i.e. about 5,760 environment steps per generation) in the
Model 2 training setting. Not one of the manuscript's algorithms; shown only to establish that the task is learnable
without a model once inputs and outputs are scaled. Generation 10 corresponds to about 57,600 environment steps, close to
the manuscript's 50,000-step budget.

{md_table(["training seed", "nominal return, generation 1", "generation 10", "generation 30", "final", "Test 1 (5 x 10 episodes)", "Test 2"], rows)}

Over the five training seeds: Test 2 {t2_vals.mean():.4e} ± {t2_vals.std(ddof=1):.2e}; Test 1 {t1_vals.mean():.4e} ± {t1_vals.std(ddof=1):.2e}.
References on Test 2: numerical optimum -1.1496e6, two-stage schedule -1.1651e6, half dose -1.3187e6, best stored
DRL configuration (Model 2 TD3, mean of 5 seeds) -1.3885e6, no dosing (= stored PPO/TRPO) -1.5033e6.
Each training run used about {out['42']['env_steps']:,} environment steps.
""")
    print("wrote results/06_policy_search_demo.json and .md")
