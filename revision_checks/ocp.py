"""Direct single-shooting solver for the daily piecewise-constant dosing problem.

This is the manuscript's own decision problem (one constant dose pair per day, the MDP's action
parameterisation), solved as a box-constrained nonlinear programme:

    minimise   sum_k [ (U+I)(t_{k+1}) + A1/2 u1_k^2 + A2/2 u2_k^2 ]      over 0 <= u_k <= u_max

The objective is evaluated with a batched fixed-step RK4 integrator and differentiated by central
finite differences (one batched simulation per gradient), then minimised with L-BFGS-B from several
starting schedules. The optimised schedule is always re-scored in the exact evaluation environment
(adaptive RK45, termination at U+I < 0.1), and that exact score is what the scripts report.

The smooth surrogate has no early termination: once U+I < 0.1 the remaining tumour cost is < 0.1 per
day and the optimiser sets the remaining doses to zero, so the two objectives agree to ~1e-6 relative.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from common import Env, REGIME_A, rk4_day_batch, run_episode


def surrogate_cost(X, y0, p, full, weights, n_sub):
    """X: (B, 2N) controls scaled to [0, 1]; returns the cost of each of the B schedules."""
    B = X.shape[0]
    N = X.shape[1] // 2
    u1 = X[:, :N] * 1000.0
    u2 = X[:, N:] * 100.0
    Y = np.repeat(np.asarray(y0, dtype=float)[:, None], B, axis=1)
    A1, A2 = weights
    cost = np.zeros(B)
    for k in range(N):
        Y = rk4_day_batch(Y, u1[:, k], u2[:, k], p, full, n_sub)
        cost += Y[0] + Y[1] + 0.5 * A1 * u1[:, k] ** 2 + 0.5 * A2 * u2[:, k] ** 2
    return cost


def make_objective(y0, p, full, weights, n_sub, eps=1e-4):
    def fg(x):
        n = x.size
        X = np.repeat(x[None, :], 2 * n + 1, axis=0)
        hi = np.minimum(x + eps, 1.0)
        lo = np.maximum(x - eps, 0.0)
        idx = np.arange(n)
        X[1 + idx, idx] = hi
        X[1 + n + idx, idx] = lo
        c = surrogate_cost(X, y0, p, full, weights, n_sub)
        return c[0], (c[1:1 + n] - c[1 + n:]) / (hi - lo)
    return fg


def default_starts(N, rng=None):
    two = np.zeros(2 * N)
    two[:10] = 1.0
    two[N:] = 1.0
    starts = {"zero": np.zeros(2 * N), "half": np.full(2 * N, 0.5), "two-stage": two, "full": np.ones(2 * N)}
    if rng is not None:
        starts["random"] = rng.uniform(0.0, 1.0, 2 * N)
    return starts


def solve(y0, p, full, weights=REGIME_A, N=60, starts=None, n_sub=None, maxiter=600):
    """Returns (best_cost, best_x, per_start_costs)."""
    n_sub = n_sub or (40 if full else 20)
    starts = starts if starts is not None else default_starts(N)
    fg = make_objective(y0, p, full, weights, n_sub)
    best, costs = None, {}
    for name, x0 in starts.items():
        res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * (2 * N),
                       options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-13, gtol=1e-9))
        costs[name] = float(res.fun)
        if best is None or res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best[0], best[1], costs


def schedule_policy(x):
    N = x.size // 2
    u1 = x[:N] * 1000.0
    u2 = x[N:] * 100.0
    return lambda t, o: (u1[t], u2[t]) if t < N else (0.0, 0.0)


def exact_score(x, model, p, weights=REGIME_A, base=None, noise=(0.0, 0.0), seed=None, record=False):
    """Score a schedule in the exact evaluation environment (RK45, early termination)."""
    env = Env(model, p, noise, weights, base=base)
    return run_episode(env, schedule_policy(x), seed=seed, record=record)
