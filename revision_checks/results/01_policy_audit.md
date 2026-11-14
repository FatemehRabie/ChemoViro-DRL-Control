# 01 - Audit of the trained agents

Source: `multiseed_results.json` (80 runs). Replica environment reproduces the stored open-loop baselines
with a maximum relative difference of 0.0e+00 over 60 values.

## Constant policies on the four tests (mean over the five seed cohorts, Regime A)

| constant policy | Test 1 | Test 2 | Test 3 | Test 4 |
|---|---|---|---|---|
| no dosing (0, 0) | -1.5008e+06 | -1.5033e+06 | -5.8467e+00 | -8.1063e+04 |
| virus only (1000, 0) | -1.3995e+06 | -1.4004e+06 | -1.0006e+04 | -1.8017e+05 |
| drug only (0, 100) | -1.4082e+06 | -1.4104e+06 | -1.0058e+03 | -9.1522e+04 |
| full dose (1000, 100) | -1.3294e+06 | -1.3311e+06 | -1.1006e+04 | -1.9048e+05 |

## PPO and TRPO against the no-dosing policy u = (0, 0)

Largest relative difference between an agent's seed-level return and the no-dosing return on the same episodes.
Test 3 returns are about -6, so the absolute difference is given there.

| trained on | algorithm | Test 1: max rel. diff. to no dosing | Test 2: max rel. diff. to no dosing | Test 4: max rel. diff. to no dosing | Test 3: max abs. diff. |
|---|---|---|---|---|---|
| Model 1 | PPO | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.000 |
| Model 1 | TRPO | 6.9e-02 | 6.8e-02 | 1.5e-03 | 0.000 |
| Model 2 | PPO | 3.1e-01 | 3.1e-01 | 6.4e-02 | 10522.641 |
| Model 2 | TRPO | 3.1e-01 | 3.1e-01 | 1.0e-01 | 10597.951 |
| Model 3 | PPO | 3.6e-02 | 3.7e-02 | 1.0e-02 | 844.229 |
| Model 3 | TRPO | 6.8e-02 | 6.9e-02 | 9.7e-03 | 3765.874 |
| Model 4 | PPO | 2.5e-02 | 2.6e-02 | 1.0e-02 | 626.104 |
| Model 4 | TRPO | 4.0e-02 | 4.0e-02 | 1.0e-02 | 1857.435 |

Across all 40 PPO/TRPO runs the largest relative difference on Tests 1, 2 and 4 is 3.09e-01;
7 runs are identical to no dosing to nine digits on every test.

## TD3 and SAC against the corners of the action box

| trained on | algorithm | seed | behaviour on Tests 1-4 | Test 2 return | worse than no dosing on Test 2 |
|---|---|---|---|---|---|
| Model 1 | TD3 | 42 | no dosing (0, 0) | -1503323 | no |
| Model 1 | TD3 | 123 | no dosing (0, 0) | -1503323 | no |
| Model 1 | TD3 | 456 | no dosing (0, 0) | -1503323 | no |
| Model 1 | TD3 | 789 | no dosing (0, 0) | -1503323 | no |
| Model 1 | TD3 | 1024 | not one fixed corner | -1503323 | no |
| Model 1 | SAC | 42 | not one fixed corner | -1470317 | no |
| Model 1 | SAC | 123 | not one fixed corner | -1472051 | no |
| Model 1 | SAC | 456 | not one fixed corner | -1471177 | no |
| Model 1 | SAC | 789 | not one fixed corner | -1465061 | no |
| Model 1 | SAC | 1024 | not one fixed corner | -1471954 | no |
| Model 2 | TD3 | 42 | not one fixed corner | -1192790 | no |
| Model 2 | TD3 | 123 | not one fixed corner | -1183806 | no |
| Model 2 | TD3 | 456 | not one fixed corner | -1170658 | no |
| Model 2 | TD3 | 789 | not one fixed corner | -1170835 | no |
| Model 2 | TD3 | 1024 | not one fixed corner | -1156756 | no |
| Model 2 | SAC | 42 | not one fixed corner | -1171510 | no |
| Model 2 | SAC | 123 | not one fixed corner | -1190059 | no |
| Model 2 | SAC | 456 | not one fixed corner | -1188702 | no |
| Model 2 | SAC | 789 | not one fixed corner | -1173164 | no |
| Model 2 | SAC | 1024 | not one fixed corner | -1186919 | no |
| Model 3 | TD3 | 42 | not one fixed corner | -1352092 | no |
| Model 3 | TD3 | 123 | not one fixed corner | -1437636 | no |
| Model 3 | TD3 | 456 | not one fixed corner | -1418450 | no |
| Model 3 | TD3 | 789 | not one fixed corner | -1445772 | no |
| Model 3 | TD3 | 1024 | not one fixed corner | -1477710 | no |
| Model 3 | SAC | 42 | not one fixed corner | -1446636 | no |
| Model 3 | SAC | 123 | not one fixed corner | -1418774 | no |
| Model 3 | SAC | 456 | not one fixed corner | -1435736 | no |
| Model 3 | SAC | 789 | not one fixed corner | -1437450 | no |
| Model 3 | SAC | 1024 | not one fixed corner | -1463252 | no |
| Model 4 | TD3 | 42 | not one fixed corner | -1490539 | no |
| Model 4 | TD3 | 123 | not one fixed corner | -1502489 | no |
| Model 4 | TD3 | 456 | not one fixed corner | -1458890 | no |
| Model 4 | TD3 | 789 | not one fixed corner | -1485214 | no |
| Model 4 | TD3 | 1024 | not one fixed corner | -1475249 | no |
| Model 4 | SAC | 42 | not one fixed corner | -1456681 | no |
| Model 4 | SAC | 123 | not one fixed corner | -1457356 | no |
| Model 4 | SAC | 456 | not one fixed corner | -1455034 | no |
| Model 4 | SAC | 789 | not one fixed corner | -1453123 | no |
| Model 4 | SAC | 1024 | not one fixed corner | -1447152 | no |

4 of the 40 runs are an exact constant corner policy on all four tests; 0 runs score below
no dosing on Test 2. Model 3 TD3 and Model 4 TD3 return identical numbers for every seed and test: False.

For the other 36 runs the stored returns show only that the policy is not one fixed corner. Actions were saved for
seed 42 alone: in all 5 of those 60 TD3/SAC runs every saved action, on all four tests, lies at a corner of the
action box (one or two distinct corners per run; see the last table).

## Training curves (mean episodic training return stored by the Monitor callback)

| trained on | algorithm | mean return, first logged point (~1k steps) | at 10k steps | at 50k steps | median |change| 10k to 50k across seeds | max |change| |
|---|---|---|---|---|---|---|
| Model 1 | PPO | -4064 | -319.2 | -4.588 | 98.49% | 98.8% |
| Model 1 | TRPO | -4064 | -444.7 | -4.587 | 99.00% | 99.0% |
| Model 1 | TD3 | -3780 | -62.53 | -55.72 | 16.67% | 23.6% |
| Model 1 | SAC | -3756 | -2465 | -63.23 | 97.45% | 97.7% |
| Model 2 | PPO | -1.36e+06 | -1.353e+06 | -1.154e+06 | 14.93% | 15.3% |
| Model 2 | TRPO | -1.36e+06 | -1.332e+06 | -1.152e+06 | 13.45% | 15.1% |
| Model 2 | TD3 | -1.347e+06 | -1.399e+06 | -1.181e+06 | 14.22% | 21.7% |
| Model 2 | SAC | -1.345e+06 | -1.342e+06 | -1.184e+06 | 11.93% | 13.4% |
| Model 3 | PPO | -1.228e+05 | -9.32e+04 | -8.064e+04 | 13.69% | 14.3% |
| Model 3 | TRPO | -1.228e+05 | -8.928e+04 | -8.092e+04 | 9.56% | 11.2% |
| Model 3 | TD3 | -1.186e+05 | -8.249e+04 | -8.174e+04 | 1.16% | 2.0% |
| Model 3 | SAC | -1.177e+05 | -1.073e+05 | -8.16e+04 | 23.75% | 25.0% |
| Model 4 | PPO | -1.223e+05 | -9.368e+04 | -8.054e+04 | 13.64% | 15.2% |
| Model 4 | TRPO | -1.223e+05 | -8.838e+04 | -8.048e+04 | 8.86% | 9.3% |
| Model 4 | TD3 | -1.185e+05 | -8.218e+04 | -8.147e+04 | 1.06% | 1.3% |
| Model 4 | SAC | -1.177e+05 | -1.07e+05 | -8.134e+04 | 24.35% | 24.4% |

The logged return changes by less than 1% between 10,000 and 50,000 steps in 4 of 80 runs
(less than 5% in 10).

Model 1 SAC is the one cell where the logged return keeps changing after 10,000 steps (median change 33%; the median in
every other cell is below 5%). No dosing returns about -4.2 in that training model and is its optimum (check 02), so a
return moving toward zero is an agent learning to stop dosing:

| seed | training return, first logged point | at 10k steps | at 50k steps | Test 2 return | Test 3 return |
|---|---|---|---|---|---|
| 42 | -3895 | -2477 | -70 | -1470317 | -27 |
| 123 | -3684 | -2336 | -59 | -1472051 | -29 |
| 456 | -4120 | -2597 | -59 | -1471177 | -27 |
| 789 | -3559 | -2560 | -65 | -1465061 | -26 |
| 1024 | -3524 | -2355 | -62 | -1471954 | -24 |

Single runs against the simplest baseline: 62 of the 120 runs score below the constant half dose on Test 2 (-1318721).

## Archived seed-42 actions on Test 1

| trained on | algorithm | u1 range (max 1000) | u2 range (max 100) | distinct actions on Test 1 (seed 42) |
|---|---|---|---|---|
| Model 1 | PPO | 0 to 0 | 0 to 0 | (0, 0) |
| Model 1 | TRPO | 0 to 0 | 0 to 18.9 | 28 |
| Model 1 | TD3 | 0 to 0 | 0 to 0 | (0, 0) |
| Model 1 | SAC | 32.1 to 44.7 | 9.79 to 12.4 | 46 |
| Model 2 | PPO | 0 to 1e+03 | 0 to 100 | 12 |
| Model 2 | TRPO | 0 to 1e+03 | 0.453 to 100 | 33 |
| Model 2 | TD3 | 0 to 1e+03 | 0.000882 to 66.9 | 44 |
| Model 2 | SAC | 7.03 to 997 | 13.3 to 98.8 | 44 |
| Model 3 | PPO | 0 to 168 | 0 to 0 | (0, 0); (167.982, 0) |
| Model 3 | TRPO | 0 to 249 | 0 to 24 | 11 |
| Model 3 | TD3 | 0 to 338 | 0 to 34.3 | 36 |
| Model 3 | SAC | 12.5 to 266 | 2.44 to 20.7 | 46 |
| Model 4 | PPO | 0 to 342 | 0 to 0 | (0, 0); (341.644, 0) |
| Model 4 | TRPO | 0 to 351 | 0 to 1.81 | (0, 0); (32.835, 0); (350.949, 1.807) |
| Model 4 | TD3 | 0 to 114 | 9.54e-05 to 13.3 | 46 |
| Model 4 | SAC | 23.9 to 160 | 6.8 to 16.3 | 46 |

Action sequences are archived for 5 of the 5 seeds only.
