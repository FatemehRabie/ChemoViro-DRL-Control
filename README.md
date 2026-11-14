# Optimal Control of Virotherapy-Chemotherapy via Deep Reinforcement Learning

This repository contains the source code, normalized multi-seed experimental suite, saved Stable-Baselines3 policy weights, Dynamic Programming (DP) grid-design ablations, direct single-shooting optimal control baselines, and verification scripts for the paper:
**"Optimal Control of a Virotherapy-Chemotherapy Combination Model via Deep Reinforcement Learning: A Comprehensive Framework"** (*Computer Methods and Programs in Biomedicine*).

## Overview
This project benchmarks four continuous-action actor-critic Deep Reinforcement Learning (DRL) algorithms—**TRPO**, **PPO**, **TD3**, and **SAC**—against direct single-shooting numerical optimal control (`L-BFGS-B`), classical Dynamic Programming (both uniform linear and log-spaced $\log_{10}(1+x)$ grids, including $\Pi_{6\to 4}$ 4D-to-6D state projection), and matched open-loop constant and two-stage dosing schedules across nonlinear 4D (immune-excluded) and 6D (coupled immune-active) chemo-virotherapy ordinary differential equation (ODE) models based on Malinzi et al. (2018).

## Key Implementation Details
* **Normalized Observation, Action, and Training-Reward Wrapper (`NormalizedEnv`):** Observations are mapped via $\tilde{s}_i = \log_{10}(1 + \max(s_i, 0)) / 8$ to $[0, 1.5]^d$, agent actions $a \in [-1, 1]^2$ are mapped affinely to physical doses $[0, 1000] \times [0, 100]$, and training step rewards are scaled by $10^{-4}$ while all logged curves and evaluation returns are reported in exact physical objective units ($-J$).
* **Verified Two-Stage ODE Integration:** Training environments advance daily steps with 40 equal classical RK4 substeps ($h = 0.025\text{ d}$) in both 4D and 6D models (verified to agree with adaptive `RK45` to $\le 1.5 \times 10^{-4}$ relative return/state error across all environments), and all final policy evaluations use `scipy.integrate.solve_ivp(..., method='RK45', rtol=1e-6, atol=1e-6)` with early stopping at $(U+I)(t) < 0.1\text{ cells/mm}^3$.
* **Benchmark Coverage (`Models 1–6`, `Tests 1–6`, and Narrow/Wide Cohorts):** Covers both the original Malinzi initial-state benchmarks (`Models 1–4`, `Tests 1–4`) and redesigned persistent-tumor / coupled-6D / wide-heterogeneity benchmarks (`Models 5–6`, `Tests 5–6`, and the $M=10$ wide virus-free patient cohort with per-patient clairvoyant bounds).

## Repository Contents
* `run_multiseed_drl.py`: Normalized multi-seed training and evaluation suite (`5` seeds `[42, 123, 456, 789, 1024]` $\times$ `6` training scenarios $\times$ `4` algorithms = `120` trained models), saving all `.zip` policy weights in `models_normalized/`, archiving per-seed action trajectories (`actions_sample` for all 5 seeds), and scoring direct optimal control (`L-BFGS-B`), log-spaced 4D DP, and open-loop schedules on identical shared episodes.
* `dp_grid_ablation.py`: Vectorized 4D Dynamic Programming ablation comparing uniform linear grids ($10^4$ to $50^4$ states) against log-spaced $\log_{10}(1+x)$ grids ($10^4$ to $30^4$ states) on identical shared Test 1 episodes, Test 2, and $\Pi_{6\to 4}$-projected 6D tests, reporting runtime, full sweep array memory, and stationary `int8` policy table size.
* `run_baselines_and_dp_ablation.py`: Grid search for matched constant and two-stage piecewise-constant open-loop dosing schedules under High-Toxicity-Penalty Regime A ($A_1=10^{-2}, A_2=10^{-1}$) and Low-Toxicity-Penalty Regime B ($A_1=10^{-7}, A_2=10^{-5}$).
* `analyze_and_plot.py`: Statistical post-processing script computing 5-seed means, SDs, 95% $t$-CIs, paired seed-level differences, exact two-sided Mann–Whitney $U$ tests, paired Wilcoxon signed-rank tests, step-down Holm adjustments, multi-seed learning curves (`output.png`), and multi-seed Regime A control trajectory plots (`M2-T1/T2/T3/T6-*.png`, `M5-T6-*.png`).
* `ODE-RL-Control.ipynb`: Interactive companion notebook demonstrating the normalized `ChemoViroEnv` / `NormalizedEnv` setup, training, and evaluation workflow.
* `revision_checks/`: Independent audit and verification suite (`01_policy_audit.py` through `06_policy_search_demo.py`, `ocp.py`, `sb3_untrained_policy_check.py`, `sb3_train_normalized.py`) verifying policy learning, solver cross-checks, DP coordinate scaling, optimal control bounds, and statistical calculations.
* `models_normalized/`: Saved Stable-Baselines3 `.zip` checkpoints for all 120 trained runs.
* `multiseed_results.json`, `statistical_summary.json`, & `dp_ablation_results.json`: Raw seed-level evaluation outputs and statistical summaries.

## Software Environment & Reproducibility
* **Python:** `3.12.10`
* **PyTorch:** `2.10.0`
* **Stable-Baselines3:** `2.7.1`
* **SB3-Contrib (TRPO):** `2.7.1`
* **Gymnasium:** `1.2.3`
* **SciPy / NumPy:** `1.16.1` / `2.4.3`
* **Seaborn / Matplotlib:** `0.13.2` / `3.10.6`
* **Random Seeds:** `[42, 123, 456, 789, 1024]`

### Running the Experiments
```bash
pip install torch gymnasium stable-baselines3 sb3-contrib scipy matplotlib seaborn numpy

# 1. Run matched open-loop baseline grid search
python run_baselines_and_dp_ablation.py

# 2. Run 5-seed normalized DRL training, optimal control bounds, and evaluations (Models 1-6, Tests 1-6)
python run_multiseed_drl.py

# 3. Run 4D Dynamic Programming grid ablation (Linear 10^4..50^4 & Log-Spaced 10^4..30^4)
python dp_grid_ablation.py

# 4. Compute statistical summaries and generate multi-seed learning & trajectory figures
python analyze_and_plot.py
```
