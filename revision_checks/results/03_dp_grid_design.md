# 03 - Dynamic programming: grid design, evaluation episodes, and the 6D tests

## 1. Where the Test 2 trajectories live

| schedule | max U | max I | max V | days with V above 1e5 |
|---|---|---|---|---|
| no dosing | 1.23e+05 | 8.52e+04 | 9.58e+06 | 35 of 47 |
| two-stage | 8.59e+04 | 5.35e+04 | 6.77e+06 | 35 of 45 |

The manuscript's DP grid spans V in [0, 1e5] and clips V at 1e5 inside its own transition model, so for most of every
episode it plans with a virus load one to two orders of magnitude too small. Its U and I axes are uniform on [0, 1e6]:
at 20 points per axis the spacing is 52,632 cells/mm^3 and at 50 points 20,408, so the initial state (U, I) = (10,000, 100)
and the termination threshold 0.1 both sit inside the first cell.

## 2. Same problem, same nine actions, grid uniform in log10(1 + x), V up to 1e8

| states | build time (s) | policy table (int8) | Test 2 return | Test 1 return (5 x 10 shared episodes) |
|---|---|---|---|---|
| 10^4 = 10,000 | 0.6 | 10 kB | -1154111 | -1152588 ± 9609 |
| 15^4 = 50,625 | 3.2 | 49 kB | -1151816 | -1150269 ± 9557 |
| 20^4 = 160,000 | 14.7 | 156 kB | -1150433 | -1148974 ± 9561 |

For comparison (Test 2): numerical optimum -1,149,610; two-stage schedule -1,165,143; manuscript DP -1,319,462 at 20^4 and
-1,211,746 at 50^4 (881 s).

## 3. The manuscript's 20^4 table on the same episodes as the other rows

`dp_grid_ablation.py` scores Test 1 with one noisy episode per seed from its own random stream:
-1334708 ± 28963. On the 5 x 10 episodes used for the agents and open-loop baselines the same table scores
-1314004 ± 9124.

## 4. 4D tables on the 6D tests, through the projection used for the DRL policies

| policy | Test 3 return | Test 4 return |
|---|---|---|
| manuscript DP, 20^4 linear grid (Default model) | -5.8 ± 0.5 | -81310 ± 2586 |
| log-grid DP, 20^4, built on the 4D Default model | -10630.6 ± 0.5 | -88596 ± 2417 |
| log-grid DP, 20^4, built on the 4D Modified model | -1255.7 ± 0.5 | -80339 ± 2504 |
| no dosing | -5.8 ± 0.5 | -81063 ± 2586 |
| half dose (500, 50) | -2755.7 ± 0.5 | -107524 ± 2665 |

Manuscript values for comparison: PPO/TRPO -5.85 ± 0.48 (Test 3) and -8.11e4 ± 2.59e3 (Test 4); half dose -2.76e3 and -1.075e5.
