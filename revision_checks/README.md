# Revision checks

Checks run on the revised manuscript's own result files (`multiseed_results.json`, `dp_ablation_results.json`) and
models, added during the pre-resubmission review on 13 November 2026. Nothing in the parent folder was changed.

Scripts `01`-`06` and `make_figures.py` need only numpy, scipy and matplotlib and were run; their output is in
`results/`. The two `sb3_*.py` scripts need the benchmark's RL stack (gymnasium, torch, stable-baselines3,
sb3-contrib). **They were not run**: the review machine had no RL packages, so only their plumbing was exercised
with stand-in modules. Treat them as untested against real Stable-Baselines3 until you have run them.

## What the checks show

| # | Question | Result | Detail |
|---|---|---|---|
| 01 | Did the 80 agents learn dosing policies? | All 40 PPO/TRPO runs return the no-dosing value u = (0, 0) to within 0.15% on Tests 1, 2 and 4. 17 of 40 TD3/SAC runs are one fixed corner of the action box on every test (all 10 Model 2 runs); in the 8 TD3/SAC runs with saved actions (seed 42) every action is at a corner. On Test 2, 78 of the 80 runs score below the constant half dose. The logged training return changes by less than 1% between 10,000 and 50,000 steps in 61 of 80 runs. | `results/01_policy_audit.md` |
| 02 | What is the optimal return, and how far is each method from it? | Test 2 optimum -1,149,610. Shortfall: two-stage schedule 1.4%, paper's DP at 50^4 5.4%, half dose 14.7%, TD3/SAC 21%, PPO/TRPO 30.8% (= no dosing). In the 4D Modified model and Test 4 the optimum is about 1% better than no dosing; in Test 3 and in Model 1's training model the optimum is no dosing. | `results/02_optimal_control.md` |
| 02 | Can feedback beat one fixed schedule in these tests? | Upper bound on the gain: 0.001% on Test 1, 0.03% on the ten perturbed parameter sets. | same |
| 02 | Is Test 4 a 6D test? | With lysis off, its (U, I, V, C) trajectories equal the 4D Modified model's to 6e-7. | same |
| 03 | Is the DP result a property of DP or of the grid? | Same nine actions on a log-spaced grid that covers the reachable V range: 10^4 states, 0.4 s, 0.4% from the optimum. The paper's grid caps V at 1e5 while V reaches 1e7. | `results/03_dp_grid_design.md` |
| 03 | Can a 4D table be used on the 6D tests? | Yes, through the projection used for the agents. The paper's own 20^4 table scores -5.8 (Test 3) and -81,310 (Test 4), the same as PPO/TRPO. | same |
| 04 | Do the training and evaluation integrators agree to 1.2e-4? | Yes for 4D Default. No for 4D Modified (Models 3-4): 3.3e-3 in return, 8.1e-3 in state. 40 sub-steps per day brings it to 9e-6. | `results/04_solver_crosscheck.md` |
| 05 | Statistical statements | Agent-vs-baseline samples are paired (shared episodes). With five seeds, Mann-Whitney p = 0.008 means only that the two groups do not overlap, and an exact paired test cannot go below 0.0625, so the p-values add little; ignoring the pairing is conservative, not inflating. Holm keeps every tied p = 0.00794 comparison in a cell, not one. Noise vs no noise at matched parameters (Model 3 vs 4) shows no change in between-seed SD. | `results/05_statistics_recheck.md` |
| 06 | Is the task learnable model-free once inputs and outputs are scaled? | A 4-16-2 policy on log-scaled observations, trained by the cross-entropy method, reaches 0.07% from the optimum (five seeds), and 1.6-3.7% after roughly 58,000 environment steps. Proof of concept only; not one of the paper's algorithms. | `results/06_policy_search_demo.md` |

Figures: `results/fig_training_curves_vs_references.png`, `results/fig_test2_optimality_gap.png`,
`results/fig_optimal_schedule.png`.

## Likely cause of 01, and how to confirm it

`run_multiseed_drl.py` gives the agents raw states (1e2 to 1e7) and asks for raw doses in [0, 1000] x [0, 100].
With Stable-Baselines3 defaults the PPO/TRPO Gaussian policy starts near 0 with standard deviation 1, so it only
ever tries doses of 0-3 units, and the tanh outputs of TD3/SAC start saturated, where they receive no gradient.
TD3 is also created without `action_noise`, so it has no exploration noise. This diagnosis fits every number in
check 01 but it was not confirmed with Stable-Baselines3 itself. To confirm it on your machine:

```bash
cd revision_checks
python sb3_untrained_policy_check.py --workers 8
```

It calls the repository's own `train_and_evaluate_job` with `total_steps = 0` and compares the returns of the
freshly initialised networks with the stored trained ones.

## Proposed corrected training set-up

```bash
python sb3_train_normalized.py --models 2 --algos SAC --seeds 42     # first look, minutes
python sb3_train_normalized.py --workers 16                          # full benchmark
```

Actions in [-1, 1]^2 mapped to the dose box, log-scaled observations, rewards scaled during training only,
exploration noise for TD3, one discount factor near 1, 40 RK4 sub-steps per day, saved models, actions archived for
every seed, and an automatic flag when a run is indistinguishable from a constant policy. Output has the same
schema as `multiseed_results.json`. Hyperparameters are starting values and have not been tuned.

## Re-running the checks

```bash
cd revision_checks
python 01_policy_audit.py        # ~15 s
python 02_optimal_control.py     # ~3 min
python 03_dp_grid_design.py      # ~1 min
python 04_solver_crosscheck.py   # ~5 s
python 05_statistics_recheck.py  # ~2 s
python 06_policy_search_demo.py  # ~2.5 min
python make_figures.py
```

`common.py` holds a Gym-free replica of the evaluation environment; `01_policy_audit.py` first verifies that it
reproduces the 60 stored open-loop baseline values to 4e-16. `ocp.py` is the direct single-shooting solver.

## Post-Revision Verification (14 November 2026)

All checks above—including `sb3_untrained_policy_check.py`—were subsequently executed on the full RL stack (`Python 3.12.10`, `PyTorch 2.10.0`, `stable-baselines3 2.7.1`, `sb3-contrib 2.7.1`, `gymnasium 1.2.3`) both to confirm the initial diagnosis on the unnormalized benchmark and to verify the revised normalized benchmark (`run_multiseed_drl.py`, `dp_grid_ablation.py`, `analyze_and_plot.py`):

- **`sb3_untrained_policy_check.py`:** Confirmed on the unnormalized setup that freshly initialized networks reproduced the stored returns, and verified on the revised normalized setup (`NormalizedEnv`, `NormalActionNoise(0, 0.2)` for TD3, $\gamma = 0.995$) that **0 of 80 runs** in Models 1–4 are identical to or within 1% of their untrained (`total_steps = 0`) initialization (`results/sb3_untrained_policy_check.json`).
- **`01_policy_audit.py` & `02_optimal_control.py`:** On Test 2 (optimum `-1,149,610`), normalized Model 2 PPO (`-1,150,492 ± 768`, **0.08%** shortfall) and TRPO (`-1,150,817 ± 758`, **0.10%** shortfall) match 20^4 log-grid DP (**0.07%** shortfall) and outperform the two-stage open-loop baseline (**1.35%** shortfall), while TD3 (**2.21%**) and SAC (**2.82%**) have **0 of 10** runs stuck at a corner, `Model 3 TD3 == Model 4 TD3` is `False`, and all 5 seeds archive full action trajectories (`results/01_policy_audit.md`, `results/02_optimal_control.md`, `results/fig_test2_optimality_gap.png`, `results/fig_training_curves_vs_references.png`).
- **`03_dp_grid_design.py` & `04_solver_crosscheck.py`:** Both uniform linear and log-spaced 4D DP tables are evaluated on the shared 5 × 10 Test 1 episodes and projected into 6D via $\Pi_{6\to 4}$ (`results/03_dp_grid_design.md`). With 40 RK4 sub-steps/day across both 4D and 6D training models, fixed-step RK4 agrees with `RK45` (`rtol = atol = 1e-6`) to `2.4e-7` (4D Default), `3.3e-5` (4D Modified; `8.6e-6` vs `DOP853` at `1e-12`), `1.0e-5` (Test 4), and `<= 1.5e-4` on 6D models (`results/04_solver_crosscheck.md`).
- **`05_statistics_recheck.py`:** Verified paired seed-level statistics, exact Mann–Whitney $U$ / Wilcoxon signed-rank probabilities, and step-down Holm adjustments retaining all tied $p = 0.00794$ pairs (`results/05_statistics_recheck.md`).

