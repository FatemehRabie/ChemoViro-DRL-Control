import os
import itertools
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

with open(os.path.join(BASE_DIR, "multiseed_results.json"), "r") as f:
    data = json.load(f)

dp_path = os.path.join(BASE_DIR, "dp_ablation_results.json")
dp_data = json.load(open(dp_path, "r")) if os.path.exists(dp_path) else {}

rl_runs = data["rl_runs"]
baselines = data["baselines"]
cohort_refs = data.get("cohort_references", {})

scenarios = [
    "Model 1 (Full, Default, Noise)",
    "Model 2 (Reduced, Default, Noise)",
    "Model 3 (Reduced, Modified, Noise)",
    "Model 4 (Reduced, Modified, No Noise)",
    "Model 5 (Full Coupled, Virus-Free, Noise)",
    "Model 6 (Reduced DR, Virus-Free, Noise)",
]
algos = ["PPO", "TRPO", "TD3", "SAC"]
tests = [
    "Test 1 (Reduced, Default, Noise)",
    "Test 2 (Reduced, Default, No Noise)",
    "Test 3 (Full, Modified, Noise)",
    "Test 4 (Full, Modified, Zero-Lysis Control)",
    "Test 5 (Reduced, Default, Virus-Free, Noise)",
    "Test 6 (Full Coupled, Virus-Free, Noise)",
]


def calc_ci95(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if sd == 0.0 or n <= 1:
        return mean, sd, mean, mean
    se = sd / np.sqrt(n)
    t_val = stats.t.ppf(0.975, df=n - 1)
    return mean, sd, float(mean - t_val * se), float(mean + t_val * se)


def holm_adjust(p_vals):
    p = np.asarray(p_vals, dtype=float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def compare_against_ref(vals, ref_vals, is_deterministic=False):
    diff = vals - ref_vals
    m_diff, sd_diff, _, _ = calc_ci95(diff)
    rel_pct = ((vals - ref_vals) / np.abs(ref_vals)) * 100.0
    m_rel, sd_rel, _, _ = calc_ci95(rel_pct)
    same_sign = int(np.all(diff > 0) or np.all(diff < 0))
    if is_deterministic:
        return {
            "rel_pct_mean": m_rel, "rel_pct_sd": sd_rel,
            "paired_diff_mean": m_diff, "paired_diff_sd": sd_diff,
            "paired_diff_min": float(np.min(diff)), "paired_diff_max": float(np.max(diff)),
            "all_5_same_sign": bool(same_sign),
            "mwu_u": None, "mwu_p": None, "r_rb": None, "wilcoxon_p": None
        }
    u_stat, p_mwu = stats.mannwhitneyu(vals, ref_vals, alternative="two-sided")
    r_rb = (2.0 * u_stat) / (len(vals) * len(ref_vals)) - 1.0
    p_wil = float(stats.wilcoxon(diff, mode="exact").pvalue) if np.all(diff != 0) else None
    return {
        "rel_pct_mean": m_rel, "rel_pct_sd": sd_rel,
        "paired_diff_mean": m_diff, "paired_diff_sd": sd_diff,
        "paired_diff_min": float(np.min(diff)), "paired_diff_max": float(np.max(diff)),
        "all_5_same_sign": bool(same_sign),
        "mwu_u": float(u_stat), "mwu_p": float(p_mwu), "r_rb": float(r_rb),
        "wilcoxon_p": p_wil
    }


# 1. Summarize all baselines
baseline_names = sorted(set(b["baseline_name"] for b in baselines))
base_summary = {}
for b_name in baseline_names:
    b_runs = sorted([b for b in baselines if b["baseline_name"] == b_name], key=lambda x: x["seed"])
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
    b0 = b_runs[0]
    base_summary[b_name]["param_unc"] = {
        "reward_mean": b0["param_unc_reward_mean"],
        "reward_sd": b0["param_unc_reward_std"],
        "burden_mean": b0["param_unc_burden_mean"],
        "patient_rewards": b0.get("param_unc_patient_rewards", [])
    }
    base_summary[b_name]["wide_cohort"] = {
        "reward_mean": b0["wide_cohort_reward_mean"],
        "reward_sd": b0["wide_cohort_reward_std"],
        "burden_mean": b0["wide_cohort_burden_mean"],
        "steps_mean": b0["wide_cohort_steps_mean"],
        "patient_rewards": b0.get("wide_cohort_patient_rewards", []),
        "noisy_reward_mean": b0.get("wide_cohort_noisy_reward_mean", None)
    }

# 2. Summarize all DRL runs
summary = {}
print("=====================================================================")
print("1. TRAINING RUNTIMES (MEAN +/- SD across 5 seeds)")
print("=====================================================================")
for scen in scenarios:
    summary[scen] = {}
    for algo in algos:
        runs = sorted([r for r in rl_runs if r["scenario_name"] == scen and r["algo"] == algo], key=lambda x: x["seed"])
        rts = [r["train_time"] for r in runs]
        m_rt, sd_rt, _, _ = calc_ci95(rts)
        print(f"{scen:43s} | {algo:4s} | {m_rt:7.2f} +/- {sd_rt:5.2f} s")
        summary[scen][algo] = {"runtime_mean": m_rt, "runtime_sd": sd_rt, "tests": {}}

        for t in tests:
            vals = np.array([r["test_evals"][t]["reward_mean"] for r in runs])
            burdens = np.array([r["test_evals"][t]["terminal_burden"] for r in runs])
            fracs = np.array([r["test_evals"][t]["residual_fraction"] for r in runs])
            steps = np.array([r["test_evals"][t]["steps"] for r in runs])
            m, sd, lo, hi = calc_ci95(vals)
            mb, sdb, _, _ = calc_ci95(burdens)

            is_det = "No Noise" in t
            half_vals = np.array(base_summary["Constant Half-Dose Baseline (500, 50)"][t]["vals"])
            twost_vals = np.array(base_summary["Matched Open-Loop 2-Stage"][t]["vals"])
            opt_vals = np.array(base_summary["Numerical Optimum Open-Loop (L-BFGS-B)"][t]["vals"])

            vs_half = compare_against_ref(vals, half_vals, is_deterministic=is_det)
            vs_twost = compare_against_ref(vals, twost_vals, is_deterministic=is_det)
            shortfall_opt = ((np.abs(vals) - np.abs(opt_vals)) / np.abs(opt_vals)) * 100.0
            m_sf, sd_sf, _, _ = calc_ci95(shortfall_opt)

            summary[scen][algo]["tests"][t] = {
                "vals": vals.tolist(),
                "mean": m, "sd": sd, "ci_lo": lo, "ci_hi": hi,
                "burden_mean": mb, "burden_sd": sdb,
                "residual_frac": float(np.mean(fracs)),
                "steps": float(np.mean(steps)),
                "shortfall_vs_opt_pct_mean": m_sf,
                "shortfall_vs_opt_pct_sd": sd_sf,
                "vs_half_dose": vs_half,
                "vs_two_stage": vs_twost,
                # Backward-compatible flat keys for vs_half_dose:
                "rel_imp_mean": vs_half["rel_pct_mean"],
                "rel_imp_sd": vs_half["rel_pct_sd"],
                "mwu_u": vs_half["mwu_u"],
                "mwu_p": vs_half["mwu_p"],
                "r_rb": vs_half["r_rb"],
            }

        # Narrow & Wide cohort summaries
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

        w_vals = [r["wide_cohort_reward_mean"] for r in runs]
        w_stds = [r["wide_cohort_reward_std"] for r in runs]
        w_burd = [r["wide_cohort_burden_mean"] for r in runs]
        w_steps = [r["wide_cohort_steps_mean"] for r in runs]
        mw, sdw, _, _ = calc_ci95(w_vals)
        # Per-patient shortfall vs clairvoyant on wide cohort
        cl_wide = np.array(cohort_refs.get("wide_clairvoyant_det_rewards", []))
        per_seed_gap_pct = []
        if len(cl_wide) == 10:
            for r in runs:
                pr = np.array(r["wide_cohort_patient_rewards"])
                per_seed_gap_pct.append(float(np.mean((np.abs(pr) - np.abs(cl_wide)) / np.abs(cl_wide) * 100.0)))
        mgap, sdgap, _, _ = calc_ci95(per_seed_gap_pct) if per_seed_gap_pct else (0.0, 0.0, 0.0, 0.0)
        summary[scen][algo]["wide_cohort"] = {
            "vals": w_vals,
            "reward_mean": mw,
            "reward_sd_across_seeds": sdw,
            "reward_sd_across_patients": float(np.mean(w_stds)),
            "burden_mean": float(np.mean(w_burd)),
            "steps_mean": float(np.mean(w_steps)),
            "mean_per_patient_shortfall_pct": mgap,
            "sd_per_patient_shortfall_pct": sdgap,
        }

# 3. Pairwise Mann-Whitney U + exact Holm step-down adjustment within each 6-pair (scenario, test) cell
pairwise_summary = {}
noisy_tests = [t for t in tests if "No Noise" not in t]
algo_pairs = [("SAC", "PPO"), ("SAC", "TRPO"), ("TD3", "PPO"), ("TD3", "TRPO"), ("SAC", "TD3"), ("TRPO", "PPO")]

for scen in scenarios:
    pairwise_summary[scen] = {}
    for t in noisy_tests:
        raw_ps, entries = [], []
        for a1, a2 in algo_pairs:
            v1 = np.array(summary[scen][a1]["tests"][t]["vals"])
            v2 = np.array(summary[scen][a2]["tests"][t]["vals"])
            u_stat, p_val = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            r_rb = (2.0 * u_stat) / (len(v1) * len(v2)) - 1.0
            diff_pct = ((np.mean(v1) - np.mean(v2)) / abs(np.mean(v2))) * 100.0
            raw_ps.append(float(p_val))
            entries.append({"pair": f"{a1} vs {a2}", "diff_pct": float(diff_pct), "u": float(u_stat), "p_raw": float(p_val), "r_rb": float(r_rb)})
        p_holm = holm_adjust(raw_ps)
        for ent, ph in zip(entries, p_holm):
            ent["p_holm"] = float(ph)
            ent["holm_sig_0.05"] = bool(ph < 0.05)
        pairwise_summary[scen][t] = entries

with open(os.path.join(BASE_DIR, "statistical_summary.json"), "w") as f:
    json.dump({"rl": summary, "baselines": base_summary, "pairwise": pairwise_summary, "dp": dp_data, "cohort_references": cohort_refs}, f, indent=2)

# ----------------------------------------------------------------------------------------------
# Figure 3: Multi-seed learning curves across Models 1-6 (3x2 grid) with No-Dosing & Optimum lines
# ----------------------------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5, "legend.fontsize": 8.5})

fig, axes = plt.subplots(3, 2, figsize=(14.5, 13.5))
colors = {"PPO": "#1f77b4", "TRPO": "#ff7f0e", "TD3": "#2ca02c", "SAC": "#d62728"}

# Map each training scenario to its corresponding reference test / cohort for horizontal reference lines
scen_ref_map = {
    "Model 1 (Full, Default, Noise)": None,
    "Model 2 (Reduced, Default, Noise)": "Test 1 (Reduced, Default, Noise)",
    "Model 3 (Reduced, Modified, Noise)": "Test 4 (Full, Modified, Zero-Lysis Control)",
    "Model 4 (Reduced, Modified, No Noise)": "Test 4 (Full, Modified, Zero-Lysis Control)",
    "Model 5 (Full Coupled, Virus-Free, Noise)": "Test 6 (Full Coupled, Virus-Free, Noise)",
    "Model 6 (Reduced DR, Virus-Free, Noise)": "wide_cohort",
}

for ax, scen in zip(axes.flat, scenarios):
    for algo in ["PPO", "TRPO", "TD3", "SAC"]:
        runs = [r for r in rl_runs if r["scenario_name"] == scen and r["algo"] == algo]
        max_step = max(r["train_steps"] for r in runs)
        grid_t = np.arange(1000, max_step + 1, 1000)
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
        ax.plot(grid_t / 1000.0, mean_c, label=f"{algo}", color=colors[algo], linewidth=1.9)
        ax.fill_between(grid_t / 1000.0, mean_c - sd_c, mean_c + sd_c, color=colors[algo], alpha=0.20)
        ax.plot(grid_t / 1000.0, mean_c - ci95_c, color=colors[algo], linestyle="--", linewidth=0.7, alpha=0.55)
        ax.plot(grid_t / 1000.0, mean_c + ci95_c, color=colors[algo], linestyle="--", linewidth=0.7, alpha=0.55)

    ref_key = scen_ref_map.get(scen)
    if ref_key == "wide_cohort":
        cl_m = cohort_refs.get("wide_clairvoyant_det_mean")
        if cl_m is not None:
            ax.axhline(cl_m, color="black", linestyle=":", linewidth=1.3, label="Clairvoyant Opt.")
    elif ref_key is not None:
        opt_m = base_summary["Numerical Optimum Open-Loop (L-BFGS-B)"][ref_key]["mean"]
        ax.axhline(opt_m, color="black", linestyle=":", linewidth=1.3, label="Numerical Opt.")
        nodose_m = base_summary["No Dosing (0, 0)"][ref_key]["mean"]
        if abs(nodose_m) < 5.0 * abs(opt_m):
            ax.axhline(nodose_m, color="gray", linestyle="-.", linewidth=1.1, label="No Dosing")
    else:
        ax.axhline(-4.2, color="black", linestyle=":", linewidth=1.3, label="Optimum / No Dosing (-4.2)")

    ax.set_title(f"{scen} ($n=5$ seeds)", fontweight="bold")
    ax.set_xlabel("Environment Steps ($\\times 10^3$)")
    ax.set_ylabel("Rolling Mean Return (Physical Units)")
    ax.legend(ncol=3, loc="lower right", frameon=True)
    ax.grid(True, alpha=0.35)

plt.tight_layout()
for target_dir in [BASE_DIR, PARENT_DIR]:
    if os.path.isdir(target_dir):
        plt.savefig(os.path.join(target_dir, "output.png"), dpi=300)
plt.savefig(os.path.join(BASE_DIR, "training_rewards_multiseed.png"), dpi=300)
plt.close()
print("Saved multi-seed Figure 3 (output.png)!")

# ----------------------------------------------------------------------------------------------
# Multi-Seed Regime A Control Action Trajectory Plots (Replacing Stale Regime B Single-Seed PNGs)
# ----------------------------------------------------------------------------------------------
def save_multiseed_action_plot(scen_name, test_name, algo, fname_stem):
    runs = sorted([r for r in rl_runs if r["scenario_name"] == scen_name and r["algo"] == algo], key=lambda x: x["seed"])
    seqs = [np.array(r["test_evals"][test_name]["actions_sample"], dtype=float) for r in runs]
    max_len = max(len(s) for s in seqs)
    # Pad post-termination steps with 0.0 (since episode terminated at U+I < 0.1, no drug/virus is administered thereafter)
    padded = np.zeros((len(seqs), max_len, 2), dtype=float)
    for idx, s in enumerate(seqs):
        if len(s) > 0:
            padded[idx, :len(s), :] = s

    t_days = np.arange(max_len)
    m_u1, sd_u1 = np.mean(padded[:, :, 0], axis=0), np.std(padded[:, :, 0], axis=0, ddof=1)
    m_u2, sd_u2 = np.mean(padded[:, :, 1], axis=0), np.std(padded[:, :, 1], axis=0, ddof=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for idx, s in enumerate(seqs):
        ax1.step(np.arange(len(s)), s[:, 0], where="post", color="#1f77b4", alpha=0.25, linewidth=1.0)
        ax2.step(np.arange(len(s)), s[:, 1], where="post", color="#d62728", alpha=0.25, linewidth=1.0)

    ax1.step(t_days, m_u1, where="post", color="#1f77b4", linewidth=2.2, label="5-Seed Mean $u_1(t)$")
    ax1.fill_between(t_days, np.maximum(0.0, m_u1 - sd_u1), np.minimum(1000.0, m_u1 + sd_u1), step="post", color="#1f77b4", alpha=0.18, label="$\\pm 1$ SD")
    ax1.set_ylim(-30, 1050)
    ax1.set_xlabel("Time Step $t$ (days)")
    ax1.set_ylabel("Viral Infusion $u_1(t)$ (virions/(mm$^3\\cdot$day))")
    ax1.set_title(f"{algo} Virotherapy Schedule — {test_name.split(' (')[0]}", fontweight="bold", fontsize=10.5)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, alpha=0.35)

    ax2.step(t_days, m_u2, where="post", color="#d62728", linewidth=2.2, label="5-Seed Mean $u_2(t)$")
    ax2.fill_between(t_days, np.maximum(0.0, m_u2 - sd_u2), np.minimum(100.0, m_u2 + sd_u2), step="post", color="#d62728", alpha=0.18, label="$\\pm 1$ SD")
    ax2.set_ylim(-5, 108)
    ax2.set_xlabel("Time Step $t$ (days)")
    ax2.set_ylabel("Chemotherapy Dose $u_2(t)$ (g/(ml$\\cdot$day))")
    ax2.set_title(f"{algo} Chemotherapy Schedule — {test_name.split(' (')[0]}", fontweight="bold", fontsize=10.5)
    ax2.legend(loc="lower left", fontsize=8.5)
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    for target_dir in [BASE_DIR, PARENT_DIR]:
        if os.path.isdir(target_dir):
            plt.savefig(os.path.join(target_dir, f"{fname_stem}.png"), dpi=250)
    plt.close()

for algo in algos:
    save_multiseed_action_plot("Model 2 (Reduced, Default, Noise)", "Test 1 (Reduced, Default, Noise)", algo, f"M2-T1-{algo}")
    save_multiseed_action_plot("Model 2 (Reduced, Default, Noise)", "Test 2 (Reduced, Default, No Noise)", algo, f"M2-T2-{algo}")
    save_multiseed_action_plot("Model 2 (Reduced, Default, Noise)", "Test 3 (Full, Modified, Noise)", algo, f"M2-T3-{algo}")
    save_multiseed_action_plot("Model 2 (Reduced, Default, Noise)", "Test 6 (Full Coupled, Virus-Free, Noise)", algo, f"M2-T6-{algo}")
    save_multiseed_action_plot("Model 5 (Full Coupled, Virus-Free, Noise)", "Test 6 (Full Coupled, Virus-Free, Noise)", algo, f"M5-T6-{algo}")

print("Saved all multi-seed Regime A control trajectory figures (M2-T1/T2/T3/T6-*.png and M5-T6-*.png)!")
