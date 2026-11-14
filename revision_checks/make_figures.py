"""Figures for the revision checks. Run after 01, 02, 03 and 06 (reads their JSON output).

  fig_training_curves_vs_references.png   stored training curves against the no-dosing and optimal returns
  fig_test2_optimality_gap.png            distance of every method from the optimal return on Test 2
  fig_optimal_schedule.png                the numerically optimal schedule and the tumour trajectories it produces

Run:  python make_figures.py
"""
from __future__ import annotations

import json

import numpy as np

from common import (ACCENT, ALGOS, DEEMPH, INK, INK2, MODELS, MUTED, ODE_PARAMS, REGIME_A, RESULTS, SERIES, Env,
                    const_policy, load_multiseed, new_figure, rl_runs, run_episode, style_axes, two_stage_policy)
from ocp import schedule_policy

import matplotlib.pyplot as plt  # noqa: E402  (backend set by new_figure)

audit = json.load(open(RESULTS / "01_policy_audit.json"))
ocp = json.load(open(RESULTS / "02_optimal_control.json"))
dp = json.load(open(RESULTS / "03_dp_grid_design.json"))
demo = json.load(open(RESULTS / "06_policy_search_demo.json")) if (RESULTS / "06_policy_search_demo.json").exists() else None
data = load_multiseed()

ENV_OF_MODEL = {
    "Model 1 (Full, Default, Noise)": "6D Default with lysis (training model of Model 1)",
    "Model 2 (Reduced, Default, Noise)": "4D Default (Test 2; training model of Model 2)",
    "Model 3 (Reduced, Modified, Noise)": "4D Modified (training model of Models 3 and 4)",
    "Model 4 (Reduced, Modified, No Noise)": "4D Modified (training model of Models 3 and 4)",
}


def fmt(v, kdec=0):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e3:
        return f"{v / 1e3:.{kdec}f}K"
    return f"{v:.1f}"


# ---- Figure 1: training curves against reference returns ---------------------------------------
fig, axes = new_figure(2, 2, figsize=(11.5, 7.6))
for ax, model in zip(axes.ravel(), MODELS):
    style_axes(ax)
    ends, lows = {}, []
    for algo in ALGOS:
        runs = rl_runs(data, model, algo)
        n = min(len(r["curve_rewards"]) for r in runs)
        ts = np.array(runs[0]["curve_timesteps"][-n:])
        mean = np.mean([r["curve_rewards"][-n:] for r in runs], axis=0)
        ax.plot(ts / 1e3, mean, color=SERIES[algo], lw=2, solid_capstyle="round",
                linestyle=(0, (4, 3)) if algo == "TRPO" else "-", label=algo, zorder=3)
        ends[algo] = mean[-1]
        lows.append(mean.min())
    env = ocp[ENV_OF_MODEL[model]]
    opt, zero = env["optimum_return"], env["references"]["no dosing (0, 0)"]["return"]
    lo, hi = min(min(lows), zero), max(opt, max(ends.values()))
    pad = 0.12 * (hi - lo if hi > lo else 1.0)
    max_k = max(np.max(r["curve_timesteps"]) / 1e3 for a in ALGOS for r in rl_runs(data, model, a))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, max_k * 1.15)
    ax.axhline(zero, color=INK2, lw=1.0, zorder=2)
    close = abs(opt - zero) < 0.03 * (hi - lo + 2 * pad)
    if close:
        ax.text(max_k * 1.13, zero, f"no dosing {fmt(zero)}\noptimum {fmt(opt)}", ha="right", va="bottom", fontsize=8.5, color=INK2, linespacing=1.15)
    else:
        ax.axhline(opt, color=INK2, lw=1.0, zorder=2)
        ax.text(max_k * 1.13, opt, f"optimum {fmt(opt)}", ha="right", va="bottom", fontsize=8.5, color=INK2)
        ax.text(max_k * 1.13, zero, f"no dosing {fmt(zero)}", ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_title(model.replace(" (", "  (", 1), fontsize=10.5)
    kdec = 1 if (hi - lo) < 8e3 else 0
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, kdec=kdec: fmt(v, kdec) if abs(v) > 1e-9 else "0"))
    ax.set_xlabel("training steps (thousands)")
    ax.set_ylabel("mean training-episode return")
handles = [plt.Line2D([0], [0], color=SERIES[a], lw=2, linestyle=(0, (4, 3)) if a == "TRPO" else "-") for a in ALGOS]
fig.legend(handles, ALGOS, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.955), fontsize=10, labelcolor=INK2, handlelength=2.6)
fig.suptitle("Training curves against no-dosing and numerically optimal reference returns (Models 1-4)",
             x=0.012, y=0.995, ha="left", fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.012, 0.008, "Curves: mean over 5 seeds of the return logged during training (multiseed_results.json).\n"
         "Grey lines: return of u = (0, 0) and of the numerically optimal schedule in the same training model (nominal initial state).",
         fontsize=8.5, color=MUTED, linespacing=1.3)
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig(RESULTS / "fig_training_curves_vs_references.png", facecolor=fig.get_facecolor())
plt.close(fig)

# ---- Figure 2: optimality gap on Test 2 --------------------------------------------------------
opt2 = ocp["4D Default (Test 2; training model of Model 2)"]["optimum_return"]
gap = lambda r: 100.0 * (opt2 - r) / abs(opt2)
rename = {"no dosing (0, 0)": "No dosing, u = (0, 0)", "half dose (500, 50)": "Constant half dose (500, 50)",
          "best constant in the paper (600, 100)": "Best constant in the paper (600, 100)", "virus only (1000, 0)": "Constant virus only (1000, 0)",
          "drug only (0, 100)": "Constant drug only (0, 100)", "full dose (1000, 100)": "Constant full dose (1000, 100)",
          "two-stage (1000,100) then (0,100) at day 10": "Two-stage open loop (paper's matched baseline)"}
bars = []
for e in ocp["test2_gap"]:
    if e["kind"] == "reference" or e["method"].startswith("best single"):
        continue
    name = rename.get(e["method"], e["method"])
    if e["kind"] == "DRL":
        name = name.replace(", trained on Model 2 (mean of 5 seeds)", " agent in the paper (Model 2, mean of 5 seeds)")
    if e["kind"] == "DP":
        name = name.replace("manuscript DP", "DP in the paper")
    bars.append((name, e["gap_pct"], e["kind"]))
for n in (10, 20):
    bars.append((f"DP on a log-spaced grid, {n}^4 states (this check)", gap(dp[f"log_dp_{n}"]["test2"]), "check"))
if demo:
    bars.append(("Policy search with scaled inputs/outputs (this check, 5 seeds)", gap(demo["summary"]["test2_mean"]), "check"))
bars.sort(key=lambda b: b[1])
fig, ax = new_figure(figsize=(10.5, 0.36 * len(bars) + 1.9))
style_axes(ax, grid_axis="x")
y = np.arange(len(bars))[::-1]
colors = [ACCENT if k == "DRL" else DEEMPH for _, _, k in bars]
ax.barh(y, [b[1] for b in bars], height=0.56, color=colors, zorder=3)
for yi, (name, g, k) in zip(y, bars):
    ax.text(g + 0.4, yi, f"{g:.1f}%" if g >= 0.95 else f"{g:.2f}%", va="center", ha="left", fontsize=9, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([b[0] for b in bars], fontsize=9.5, color=INK2)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 35)
ax.set_xlabel("shortfall from the optimal return on Test 2 (%)   -   optimum = -1.1496 x 10^6")
handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT), plt.Rectangle((0, 0), 1, 1, color=DEEMPH)]
fig.legend(handles, ["DRL agents reported in the paper", "open-loop schedules, DP, and the checks added here"], loc="upper left",
           bbox_to_anchor=(0.012, 0.955), ncol=2, fontsize=9.5, labelcolor=INK2)
drl_gaps = [b[1] for b in bars if b[2] == "DRL"]
fig.suptitle(f"Test 2: Model 2 DRL configurations span {min(drl_gaps):.2f}-{max(drl_gaps):.2f}% from the numerical optimum",
             x=0.012, y=0.992, ha="left", fontsize=12.5, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(RESULTS / "fig_test2_optimality_gap.png", facecolor=fig.get_facecolor())
plt.close(fig)

# ---- Figure 3: optimal schedule and trajectories ----------------------------------------------
x = np.load(RESULTS / "optimal_schedule_test2_regimeA.npy")
N = x.size // 2
P = ODE_PARAMS["default"]
runs = {"no dosing": (const_policy(0.0, 0.0), DEEMPH), "constant half dose": (const_policy(500.0, 50.0), SERIES["TD3"]),
        "two-stage": (two_stage_policy, SERIES["TRPO"]), "numerical optimum": (schedule_policy(x), ACCENT)}
fig, (ax1, ax2) = new_figure(2, 1, figsize=(9.5, 7.2), gridspec_kw={"height_ratios": [1.55, 1]}, sharex=True)
style_axes(ax1)
style_axes(ax2)
label_pos = {}
for name, (pol, col) in runs.items():
    r = run_episode(Env("reduced", P, (0.0, 0.0), REGIME_A), pol, record=True)
    tum = (r["traj"][:, 0] + r["traj"][:, 1]) / 1e3
    ax1.plot(np.arange(len(tum)), tum, color=col, lw=2, linestyle=(0, (4, 3)) if name == "two-stage" else "-",
             label=f"{name}  (return {r['reward'] / 1e6:.3f}M, cleared day {r['steps']})", zorder={"no dosing": 2, "two-stage": 5}.get(name, 3))
    label_pos[name] = (int(tum.argmax()), float(tum.max()))
leader = dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=3)
ax1.annotate("no dosing", xy=label_pos["no dosing"], xytext=(8.6, 150), fontsize=9, color=INK2, ha="right", va="center", arrowprops=leader)
ax1.annotate("constant half dose", xy=label_pos["constant half dose"], xytext=(8.6, 128), fontsize=9, color=INK2, ha="right", va="center", arrowprops=leader)
ax1.annotate("optimum and two-stage\n(nearly identical)", xy=label_pos["numerical optimum"], xytext=(8.6, 104), fontsize=9, color=INK2, ha="right", va="center",
             arrowprops=leader, linespacing=1.2)
ax1.set_ylabel("tumour burden U + I (10^3 cells/mm^3)")
ax1.set_ylim(0, 165)
ax1.legend(loc="upper right", fontsize=9, labelcolor=INK2)
ax1.set_title("Test 2: the optimal schedule against the paper's reference schedules", fontsize=12)
days = np.arange(N)
ax2.step(days, x[:N], where="post", color=ACCENT, lw=2, label="virus dose u1 / 1000")
ax2.step(days, x[N:], where="post", color="#86b6ef", lw=2, label="drug dose u2 / 100")
ax2.text(8.2, 0.62, "virus u1", fontsize=9, color=INK2)
ax2.text(21.5, 0.62, "drug u2", fontsize=9, color=INK2)
ax2.set_ylim(-0.03, 1.08)
ax2.set_xlim(0, 47)
ax2.set_ylabel("optimal dose (fraction of maximum)")
ax2.set_xlabel("day")
ax2.legend(loc="upper right", fontsize=9, labelcolor=INK2)
fig.tight_layout()
fig.savefig(RESULTS / "fig_optimal_schedule.png", facecolor=fig.get_facecolor())
plt.close(fig)
print("wrote results/fig_training_curves_vs_references.png, fig_test2_optimality_gap.png, fig_optimal_schedule.png")
