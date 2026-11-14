"""Re-check of three statistical statements in the manuscript, from ../multiseed_results.json.

  1. Agent-versus-baseline tests. The agent and the half-dose baseline are scored on the same episodes
     (episode seed = 100 * seed + ep), so the five seed-level values are paired, and the baseline is a
     fixed policy whose five values differ only through those shared initial states. The script reports
     the correlation between the two columns, the Mann-Whitney p-value used in Table 9, the exact
     Wilcoxon signed-rank p-value for the five paired differences, and a simulation of how often the
     Mann-Whitney test rejects on paired samples when there is no true difference.
  2. The Holm statement ("within each 6-pair model-test cell, only a single comparison at p = 0.00794
     satisfies step-down Holm adjustment"). The script applies Holm's procedure to every cell.
  3. The statement that training with noise halves the between-seed SD of SAC (Model 2 vs Model 4).
     Models 2 and 4 differ in parameter set and hyperparameters as well as noise; the matched pair is
     Model 3 vs Model 4.

Run:  python 05_statistics_recheck.py   (a few seconds)
Out:  results/05_statistics_recheck.json, results/05_statistics_recheck.md
"""
from __future__ import annotations

import itertools

import numpy as np
import scipy.stats as st

from common import ALGOS, MODELS, RESULTS, SEEDS, TESTS, load_multiseed, md_table, rl_runs, save_json

data = load_multiseed()
HALF = "Constant Half-Dose Baseline (500, 50)"
half = {t: np.array([b["test_evals"][t]["reward_mean"] for b in sorted((b for b in data["baselines"] if b["baseline_name"] == HALF),
                                                                     key=lambda b: SEEDS.index(b["seed"]))]) for t in TESTS}
noisy_tests = [t for t in TESTS if "No Noise" not in t]
short = lambda s: s.split(" (")[0]
out = {}

# ---- 1. paired structure ------------------------------------------------------------------------
rows, n_mwu_sig, n_wil_sig, n_cmp, corrs = [], 0, 0, 0, []
n_same_sign = 0
for model in MODELS:
    for algo in ALGOS:
        for t in noisy_tests:
            v = np.array([r["test_evals"][t]["reward_mean"] for r in rl_runs(data, model, algo)])
            p_mwu = st.mannwhitneyu(v, half[t], alternative="two-sided").pvalue
            d = v - half[t]
            p_wil = st.wilcoxon(d, mode="exact").pvalue if np.all(d != 0) else float("nan")
            n_same_sign += bool(np.all(d > 0) or np.all(d < 0))
            rho = float(np.corrcoef(v, half[t])[0, 1]) if np.std(v) > 0 else float("nan")
            n_cmp += 1
            n_mwu_sig += p_mwu < 0.05
            n_wil_sig += p_wil < 0.05
            if algo in ("PPO", "TRPO"):
                corrs.append(rho)
            rows.append([short(model), algo, short(t), f"{p_mwu:.3f}", f"{p_wil:.4f}", f"{rho:.3f}"])


def mwu_null_rejection_rate(rho, n=5, trials=200_000, seed=0):
    """Share of trials in which the exact two-sided Mann-Whitney test rejects at 0.05 although both groups have the
    same distribution. Each pair shares a component that gives correlation rho (rho = None: independent groups)."""
    ranks = set(range(2 * n))                                      # exact null distribution of U by enumeration
    us = np.array([sum(sum(1 for j in ranks - set(c) if i > j) for i in c) for c in itertools.combinations(sorted(ranks), n)])
    crit = max(k for k in range(n * n // 2 + 1) if 2 * np.mean(us <= k) < 0.05)   # 2 for n = 5
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(trials, n))
    if rho is None:
        a, b = x, rng.normal(size=(trials, n))
    else:
        s = np.sqrt(1.0 / rho - 1.0)
        a, b = x + s * rng.normal(size=(trials, n)), x + s * rng.normal(size=(trials, n))
    u = (a[:, :, None] > b[:, None, :]).sum(axis=(1, 2))
    return float(np.mean(np.minimum(u, n * n - u) <= crit))


null_rate = {"independent": mwu_null_rejection_rate(None), "corr_0.9": mwu_null_rejection_rate(0.9), "corr_0.999": mwu_null_rejection_rate(0.999)}
out["agent_vs_half_dose"] = {"comparisons": n_cmp, "mann_whitney_p_below_0.05": int(n_mwu_sig), "paired_wilcoxon_p_below_0.05": int(n_wil_sig),
                             "all_five_paired_differences_same_sign": int(n_same_sign),
                             "min_attainable_mann_whitney_p_n5": 2 / 252, "min_attainable_paired_p_n5": 2 / 2 ** 5,
                             "median_corr_ppo_trpo_vs_baseline": float(np.nanmedian(corrs)),
                             "mann_whitney_rejection_rate_with_no_true_difference": null_rate}
print(f"[1] agent vs half-dose on Tests 1, 3, 4: {n_cmp} comparisons; Mann-Whitney p < 0.05 in {n_mwu_sig}; exact paired Wilcoxon p < 0.05 in {n_wil_sig} "
      f"(smallest attainable two-sided value with n = 5 is {2 / 2 ** 5}); all five paired differences share a sign in {n_same_sign}; "
      f"median correlation of PPO/TRPO seed values with the baseline's: {np.nanmedian(corrs):.3f}")
print("    Mann-Whitney rejection rate at 0.05 with no true difference:", {k: round(v, 4) for k, v in null_rate.items()})

# ---- 2. Holm within each six-pair cell ----------------------------------------------------------
def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

cell_rows, multi = [], 0
for model in MODELS:
    for t in noisy_tests:
        pairs = list(itertools.combinations(ALGOS, 2))
        ps = []
        for a, b in pairs:
            va = np.array([r["test_evals"][t]["reward_mean"] for r in rl_runs(data, model, a)])
            vb = np.array([r["test_evals"][t]["reward_mean"] for r in rl_runs(data, model, b)])
            ps.append(st.mannwhitneyu(va, vb, alternative="two-sided").pvalue)
        adj = holm(ps)
        n_raw = int(np.sum(np.array(ps) < 0.05))
        n_holm = int(np.sum(adj < 0.05))
        multi += n_holm > 1
        cell_rows.append([short(model), short(t), n_raw, n_holm, f"{np.min(adj):.4f}"])
out["holm"] = {"cells": len(cell_rows), "cells_with_more_than_one_holm_significant_pair": int(multi)}
print(f"[2] Holm within each six-pair cell: {multi} of {len(cell_rows)} cells keep more than one comparison below 0.05 after adjustment")

# ---- 3. between-seed SD and training noise ------------------------------------------------------
sd_rows = []
T1 = "Test 1 (Reduced, Default, Noise)"
for model in MODELS:
    sd_rows.append([model] + [f"{np.std([r['test_evals'][T1]['reward_mean'] for r in rl_runs(data, model, a)], ddof=1):.3g}" for a in ("SAC", "TD3")])
out["between_seed_sd_test1"] = sd_rows
print("[3] between-seed SD on Test 1 (SAC, TD3):", [(short(r[0]), r[1], r[2]) for r in sd_rows])

save_json("05_statistics_recheck.json", out)
(RESULTS / "05_statistics_recheck.md").write_text(f"""# 05 - Statistical statements re-checked

## 1. Agent versus half-dose baseline: paired samples, and what five seeds can show

Both columns are evaluated on the same episodes, so the baseline's five "seed" values are five evaluations of one
fixed policy on shared initial states, and the agent's values share those initial states. The Mann-Whitney test in
Table 9 treats the two columns as independent samples. The median correlation between PPO/TRPO seed-level returns and
the baseline's is {np.nanmedian(corrs):.3f}.

With five values per group the attainable p-values are coarse. The smallest two-sided Mann-Whitney value is 0.0079
(all five values of one group beyond all five of the other); the smallest exact Wilcoxon signed-rank value for five
paired differences is 0.0625. Of {n_cmp} agent-versus-baseline comparisons on Tests 1, 3 and 4, Mann-Whitney gives
p < 0.05 in {n_mwu_sig}. The paired test cannot give p < 0.05 at n = 5 ({n_wil_sig} of {n_cmp}). In {n_same_sign} of the {n_cmp} comparisons all five
paired differences have the same sign, which gives its smallest value, 0.0625.

Ignoring the pairing does not inflate significance. With no true difference, the Mann-Whitney test at the 0.05 level
rejects in {100 * null_rate["independent"]:.1f}% of simulated trials for independent groups, {100 * null_rate["corr_0.9"]:.2f}% at correlation 0.9 and {100 * null_rate["corr_0.999"]:.2f}% at
correlation 0.999 (200,000 trials each). What the pairing changes is the reading of the table:

- p = 0.008 says only that the two groups of five do not overlap. It is the same number for a 5% and a 99% difference.
- For PPO, TRPO and the constant-corner TD3/SAC runs (check 01), both columns are fixed policies scored on the same
  episodes. The difference is then a property of the ODE model, and the "between-seed" SD is evaluation noise from
  the shared initial states, not variability between trained agents.
- The paired differences (mean and range over seeds) are the informative summary. A paired exact test needs at least
  six seeds to reach p < 0.05.

{md_table(["trained on", "algorithm", "test", "Mann-Whitney p (Table 9)", "paired Wilcoxon p", "correlation with baseline"], rows)}

## 2. Holm adjustment inside each six-pair cell

Holm compares the k-th smallest p-value with alpha / (6 - k + 1). Four tied values of 0.00794 are compared with
0.00833, 0.0100, 0.0125 and 0.0167 and all four are rejected (adjusted p = 0.0476 each).
{multi} of the {len(cell_rows)} cells keep more than one comparison below 0.05 after adjustment.

{md_table(["trained on", "test", "pairs with raw p < 0.05", "pairs with Holm-adjusted p < 0.05", "smallest adjusted p"], cell_rows)}

## 3. Between-seed SD on Test 1

{md_table(["training model", "SAC", "TD3"], sd_rows)}

Model 2 and Model 4 differ in parameter set, hyperparameters and noise. The pair that differs only in training
noise is Model 3 (noise) and Model 4 (no noise).
""")
print("wrote results/05_statistics_recheck.json and .md")
