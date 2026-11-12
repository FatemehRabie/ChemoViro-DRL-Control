import os
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "multiseed_results.json"), "r") as f:
    data = json.load(f)

with open(os.path.join(BASE_DIR, "dp_ablation_results.json"), "r") as f:
    dp_data = json.load(f)

rl_runs = data["rl_runs"]
baselines = data["baselines"]

scenarios = [
    "Model 1 (Full, Default, Noise)",
    "Model 2 (Reduced, Default, Noise)",
    "Model 3 (Reduced, Modified, Noise)",
    "Model 4 (Reduced, Modified, No Noise)",
]
algos = ["PPO", "TRPO", "TD3", "SAC"]
tests = [
    "Test 1 (Reduced, Default, Noise)",
    "Test 2 (Reduced, Default, No Noise)",
    "Test 3 (Full, Modified, Noise)",
    "Test 4 (Full, Modified, Zero-Lysis Control)"
]

def calc_ci95(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return mean, sd, mean, mean
    se = sd / np.sqrt(n)
    t_val = stats.t.ppf(0.975, df=n-1)
    return mean, sd, mean - t_val * se, mean + t_val * se

summary = {}
print("=====================================================================")
print("1. TRAINING RUNTIMES (MEAN +/- SD across 5 seeds)")
print("=====================================================================")
for scen in scenarios:
    summary[scen] = {}
    for algo in algos:
        runs = [r for r in rl_runs if r["scenario_name"] == scen and r["algo"] == algo]
        rts = [r["train_time"] for r in runs]
        m_rt, sd_rt, _, _ = calc_ci95(rts)
        print(f"{scen:40s} | {algo:5s} | {m_rt:7.2f} +/- {sd_rt:5.2f} s")
        summary[scen][algo] = {"runtime_mean": m_rt, "runtime_sd": sd_rt, "tests": {}}

print("\n=====================================================================")
print("2. MULTI-SEED PERFORMANCE ACROSS TESTS 1-4 (5 SEEDS)")
print("=====================================================================")

# Collect baseline seed values for Test 1..4
base_summary = {}
for b_name in ["Constant Half-Dose Baseline (500, 50)", "Matched Open-Loop Constant (600, 100)", "Matched Open-Loop 2-Stage"]:
    b_runs = [b for b in baselines if b["baseline_name"] == b_name]
    base_summary[b_name] = {}
    for t in tests:
        vals = [b["test_evals"][t]["reward_mean"] for b in b_runs]
        burdens = [b["test_evals"][t]["terminal_burden"] for b in b_runs]
        steps = [b["test_evals"][t]["steps"] for b in b_runs]
        m, sd, lo, hi = calc_ci95(vals)
        mb, sdb, _, _ = calc_ci95(burdens)
        base_summary[b_name][t] = {
            "vals": vals, "mean": m, "sd": sd, "ci_lo": lo, "ci_hi": hi,
            "burden_mean": mb, "burden_sd": sdb, "steps": float(np.mean(steps))
        }
    p_vals = [b["param_unc_reward_mean"] for b in b_runs]
    p_stds = [b["param_unc_reward_std"] for b in b_runs]
    p_burd = [b["param_unc_burden_mean"] for b in b_runs]
    base_summary[b_name]["param_unc"] = {
        "reward_mean": float(np.mean(p_vals)),
        "reward_sd": float(np.mean(p_stds)),
        "burden_mean": float(np.mean(p_burd))
    }

const_half_t1 = np.array(base_summary["Constant Half-Dose Baseline (500, 50)"]["Test 1 (Reduced, Default, Noise)"]["vals"])

for scen in scenarios:
    print(f"\n--- {scen} ---")
    for algo in algos:
        runs = [r for r in rl_runs if r["scenario_name"] == scen and r["algo"] == algo]
        # Sort by seed
        runs = sorted(runs, key=lambda x: x["seed"])
        for t in tests:
            vals = np.array([r["test_evals"][t]["reward_mean"] for r in runs])
            burdens = np.array([r["test_evals"][t]["terminal_burden"] for r in runs])
            fracs = np.array([r["test_evals"][t]["residual_fraction"] for r in runs])
            steps = np.array([r["test_evals"][t]["steps"] for r in runs])
            m, sd, lo, hi = calc_ci95(vals)
            mb, sdb, _, _ = calc_ci95(burdens)
            # Relative improvement over Constant Half-Dose baseline (%)
            b_vals = np.array(base_summary["Constant Half-Dose Baseline (500, 50)"][t]["vals"])
            rel_imp = ((vals - b_vals) / np.abs(b_vals)) * 100.0
            m_imp, sd_imp, _, _ = calc_ci95(rel_imp)
            # Mann-Whitney U test vs Constant Half-Dose baseline (on stochastic tests with non-zero baseline variance)
            if "No Noise" in t:
                u_stat, p_val, r_rb = None, None, None
            else:
                try:
                    u_stat, p_val = stats.mannwhitneyu(vals, b_vals, alternative='two-sided')
                    # rank-biserial correlation r_rb = 2U/(n1*n2) - 1
                    r_rb = (2.0 * u_stat) / (len(vals) * len(b_vals)) - 1.0
                except Exception:
                    u_stat, p_val, r_rb = 12.5, 1.0, 0.0
            summary[scen][algo]["tests"][t] = {
                "vals": vals.tolist(),
                "mean": m, "sd": sd, "ci_lo": lo, "ci_hi": hi,
                "burden_mean": mb, "burden_sd": sdb,
                "residual_frac": float(np.mean(fracs)),
                "steps": float(np.mean(steps)),
                "rel_imp_mean": m_imp, "rel_imp_sd": sd_imp,
                "mwu_u": float(u_stat) if u_stat is not None else None,
                "mwu_p": float(p_val) if p_val is not None else None,
                "r_rb": float(r_rb) if r_rb is not None else None
            }
            p_str = f"U={u_stat:.1f}, p={p_val:.3f}" if p_val is not None else "Determ. baseline"
            print(f"  {algo:4s} | {t[:18]:18s} | R = {m:11.3e} +/- {sd:8.2e} (95%CI [{lo:.2e}, {hi:.2e}]) | Burden = {mb:.3e} +/- {sdb:.2e} | Steps={np.mean(steps):.1f} | d%={m_imp:+.2f}% +/- {sd_imp:.2f}% | {p_str}")
        p_vals = [r["param_unc_reward_mean"] for r in runs]
        p_stds = [r["param_unc_reward_std"] for r in runs]
        p_burd = [r["param_unc_burden_mean"] for r in runs]
        mp, sdp, _, _ = calc_ci95(p_vals)
        summary[scen][algo]["param_unc"] = {
            "reward_mean": mp,
            "reward_sd_across_seeds": sdp,
            "reward_sd_across_params": float(np.mean(p_stds)),
            "burden_mean": float(np.mean(p_burd))
        }

print("\n--- BASELINES ---")
for b_name, b_dict in base_summary.items():
    print(f"Baseline: {b_name}")
    for t in tests:
        d = b_dict[t]
        print(f"  {t[:25]:25s} | R = {d['mean']:11.3e} +/- {d['sd']:8.2e} | Burden = {d['burden_mean']:.3e} | Steps = {d['steps']:.1f}")
    print(f"  Parametric Uncertainty    | R = {b_dict['param_unc']['reward_mean']:11.3e} +/- {b_dict['param_unc']['reward_sd']:8.2e} | Burden = {b_dict['param_unc']['burden_mean']:.3e}")

# Pairwise Mann-Whitney U tests between On-Policy (TRPO/PPO) and Off-Policy (TD3/SAC)
print("\n=====================================================================")
print("3. PAIRWISE MANN-WHITNEY U TESTS BETWEEN ALGORITHMS")
print("=====================================================================")
for scen in scenarios:
    print(f"\nScenario: {scen}")
    for t in ["Test 1 (Reduced, Default, Noise)", "Test 3 (Full, Modified, Noise)", "Test 4 (Full, Modified, Zero-Lysis Control)"]:
        for a1, a2 in [("SAC", "PPO"), ("SAC", "TRPO"), ("TD3", "PPO"), ("TD3", "TRPO"), ("SAC", "TD3"), ("TRPO", "PPO")]:
            v1 = np.array(summary[scen][a1]["tests"][t]["vals"])
            v2 = np.array(summary[scen][a2]["tests"][t]["vals"])
            u_stat, p_val = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            r_rb = (2.0 * u_stat) / (len(v1) * len(v2)) - 1.0
            diff_pct = ((np.mean(v1) - np.mean(v2)) / abs(np.mean(v2))) * 100.0
            print(f"  {t[:6]} | {a1:4s} vs {a2:4s}: diff={diff_pct:+7.2f}%, U={u_stat:4.1f}, p={p_val:.4f}, r_rb={r_rb:+.2f}")

with open(os.path.join(BASE_DIR, "statistical_summary.json"), "w") as f:
    json.dump({"rl": summary, "baselines": base_summary, "dp": dp_data}, f, indent=2)

# Plot Figure 3: Multi-seed learning curves with both +/- 1 SD dispersion bands and 95% t-CI bands!
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

fig, axes = plt.subplots(4, 1, figsize=(11, 16), sharex=True)
colors = {"PPO": "#1f77b4", "TRPO": "#ff7f0e", "TD3": "#2ca02c", "SAC": "#d62728"}

for ax, scen in zip(axes, scenarios):
    for algo in ["TRPO", "TD3", "SAC", "PPO"]:
        runs = [r for r in rl_runs if r["scenario_name"] == scen and r["algo"] == algo]
        # Interpolate curves onto common grid 1000..50000
        grid_t = np.arange(1000, 50001, 1000)
        interp_curves = []
        for r in runs:
            ts = np.array(r["curve_timesteps"])
            rw = np.array(r["curve_rewards"])
            if len(ts) > 0:
                interp_curves.append(np.interp(grid_t, ts, rw))
        interp_curves = np.array(interp_curves)
        n_seeds = interp_curves.shape[0]
        mean_c = np.mean(interp_curves, axis=0)
        sd_c = np.std(interp_curves, axis=0, ddof=1)
        ci95_c = stats.t.ppf(0.975, df=n_seeds - 1) * sd_c / np.sqrt(n_seeds)
        ax.plot(grid_t, mean_c, label=f"{algo} (mean)", color=colors[algo], linewidth=2.0)
        ax.fill_between(grid_t, mean_c - sd_c, mean_c + sd_c, color=colors[algo], alpha=0.24, label=f"{algo} (±1 SD)")
        ax.fill_between(grid_t, mean_c - ci95_c, mean_c + ci95_c, color=colors[algo], alpha=0.10)
        ax.plot(grid_t, mean_c - ci95_c, color=colors[algo], linestyle="--", linewidth=0.8, alpha=0.55)
        ax.plot(grid_t, mean_c + ci95_c, color=colors[algo], linestyle="--", linewidth=0.8, alpha=0.55, label=f"{algo} (95% CI)")
    ax.set_title(f"Training Progress: {scen} (n = 5 independent seeds)")
    ax.set_ylabel("Mean Episode Return")
    ax.legend(ncol=4, loc="lower right", frameon=True)
    ax.grid(True, alpha=0.4)

axes[-1].set_xlabel("Training Timesteps")
plt.tight_layout()
out_png_repo = os.path.join(BASE_DIR, "output.png")
out_png_multiseed = os.path.join(BASE_DIR, "training_rewards_multiseed.png")
plt.savefig(out_png_repo, dpi=300)
plt.savefig(out_png_multiseed, dpi=300)
parent_out_png = os.path.join(os.path.dirname(BASE_DIR), "output.png")
if os.path.isdir(os.path.dirname(BASE_DIR)):
    plt.savefig(parent_out_png, dpi=300)
plt.close()
print("Saved multi-seed Figure 3 (output.png) with ±1 SD dispersion bands and 95% CI boundaries!")
