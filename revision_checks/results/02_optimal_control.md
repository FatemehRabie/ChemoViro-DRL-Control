# 02 - Numerically optimal schedules and what they imply

All returns are computed in the exact evaluation environment (RK45, termination at U+I < 0.1), nominal initial
condition, no noise, unless stated. "Gap to optimum" is (optimum - return) / |optimum|.

## a. Optimum per environment

### 4D Default (Test 2; training model of Model 2)

Weights (A1, A2) = (0.01, 0.1). Optimum found from 5 starting schedules; their final costs agree to 7.7e-14 (relative).

| schedule | return | gap to optimum | days to U+I<0.1 | tumour cost | dose cost | peak U+I |
|---|---|---|---|---|---|---|
| numerical optimum | -1149609.8 | 0.00% | 44 | 1100083 | 49526 | 105645 |
| no dosing (0, 0) | -1503322.9 | 30.77% | 46 | 1503323 | 0 | 150790 |
| half dose (500, 50) | -1318720.6 | 14.71% | 45 | 1256846 | 61875 | 122730 |
| best constant in the paper (600, 100) | -1281244.6 | 11.45% | 44 | 1180045 | 101200 | 113033 |
| virus only (1000, 0) | -1400399.3 | 21.82% | 44 | 1180399 | 220000 | 117577 |
| drug only (0, 100) | -1410354.4 | 22.68% | 45 | 1387854 | 22500 | 133848 |
| full dose (1000, 100) | -1331054.8 | 15.78% | 44 | 1089055 | 242000 | 104735 |
| two-stage (1000,100) then (0,100) at day 10 | -1165143.5 | 1.35% | 44 | 1093143 | 72000 | 105066 |

Optimal u1 by day (first 24): 1000, 1000, 1000, 1000, 1000, 1000, 1000, 732, 490, 315, 194, 113, 62, 32, 15, 6, 2, 0, 0, 0, 0, 0, 0, 0

Optimal u2 by day (first 32): 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 69, 41, 25, 15, 9, 5, 3, 2, 1, 1, 0, 0, 0

### 4D Default, Regime B weights

Weights (A1, A2) = (1e-07, 1e-05). Optimum found from 5 starting schedules; their final costs agree to 1.7e-15 (relative).

| schedule | return | gap to optimum | days to U+I<0.1 | tumour cost | dose cost | peak U+I |
|---|---|---|---|---|---|---|
| numerical optimum | -1089057.7 | 0.00% | 44 | 1089055 | 3 | 104735 |
| no dosing (0, 0) | -1503322.9 | 38.04% | 46 | 1503323 | 0 | 150790 |
| half dose (500, 50) | -1256846.7 | 15.41% | 45 | 1256846 | 1 | 122730 |
| best constant in the paper (600, 100) | -1180047.6 | 8.35% | 44 | 1180045 | 3 | 113033 |
| virus only (1000, 0) | -1180401.5 | 8.39% | 44 | 1180399 | 2 | 117577 |
| drug only (0, 100) | -1387856.6 | 27.44% | 45 | 1387854 | 2 | 133848 |
| full dose (1000, 100) | -1089059.2 | 0.00% | 44 | 1089055 | 4 | 104735 |
| two-stage (1000,100) then (0,100) at day 10 | -1093146.2 | 0.38% | 44 | 1093143 | 3 | 105066 |

Optimal u1 by day (first 24): 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 199, 10, 0, 0, 0

Optimal u2 by day (first 32): 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100

### 4D Modified (training model of Models 3 and 4)

Weights (A1, A2) = (0.01, 0.1). Optimum found from 5 starting schedules; their final costs agree to 1.2e-11 (relative).

| schedule | return | gap to optimum | days to U+I<0.1 | tumour cost | dose cost | peak U+I |
|---|---|---|---|---|---|---|
| numerical optimum | -80472.7 | 0.00% | 21 | 79498 | 974 | 24328 |
| no dosing (0, 0) | -81266.9 | 0.99% | 21 | 81267 | 0 | 24493 |
| half dose (500, 50) | -107966.4 | 34.17% | 21 | 79091 | 28875 | 24294 |
| best constant in the paper (600, 100) | -126980.6 | 57.79% | 21 | 78681 | 48300 | 24254 |
| virus only (1000, 0) | -182152.9 | 126.35% | 21 | 77153 | 105000 | 24097 |
| drug only (0, 100) | -91756.0 | 14.02% | 21 | 81256 | 10500 | 24491 |
| full dose (1000, 100) | -192642.5 | 139.39% | 21 | 77143 | 115500 | 24095 |
| two-stage (1000,100) then (0,100) at day 10 | -137642.5 | 71.04% | 21 | 77143 | 60500 | 24095 |

Optimal u1 by day (first 24): 439, 43, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

Optimal u2 by day (first 30): 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

### 6D Modified, zero lysis (Test 4)

Weights (A1, A2) = (0.01, 0.1). Optimum found from 5 starting schedules; their final costs agree to 1.2e-11 (relative).

| schedule | return | gap to optimum | days to U+I<0.1 | tumour cost | dose cost | peak U+I |
|---|---|---|---|---|---|---|
| numerical optimum | -80456.8 | 0.00% | 21 | 79707 | 750 | 24349 |
| no dosing (0, 0) | -81266.9 | 1.01% | 21 | 81267 | 0 | 24493 |
| half dose (500, 50) | -107966.4 | 34.19% | 21 | 79091 | 28875 | 24294 |
| best constant in the paper (600, 100) | -126980.6 | 57.82% | 21 | 78681 | 48300 | 24254 |
| virus only (1000, 0) | -182152.9 | 126.40% | 21 | 77153 | 105000 | 24097 |
| drug only (0, 100) | -91756.0 | 14.04% | 21 | 81256 | 10500 | 24491 |
| full dose (1000, 100) | -192642.5 | 139.44% | 21 | 77143 | 115500 | 24095 |
| two-stage (1000,100) then (0,100) at day 10 | -137642.5 | 71.08% | 21 | 77143 | 60500 | 24095 |

Optimal u1 by day (first 24): 386, 37, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

Optimal u2 by day (first 30): 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

### 6D Modified with lysis (Test 3)

Weights (A1, A2) = (0.01, 0.1). Optimum found from 5 starting schedules; their final costs agree to 0.0e+00 (relative).

| schedule | return | gap to optimum | days to U+I<0.1 | tumour cost | dose cost | peak U+I |
|---|---|---|---|---|---|---|
| numerical optimum | -5.5 | 0.00% | 2 | 5 | 0 | 10100 |
| no dosing (0, 0) | -5.5 | 0.00% | 2 | 5 | 0 | 10100 |
| half dose (500, 50) | -2755.4 | 50086.59% | 2 | 5 | 2750 | 10100 |
| best constant in the paper (600, 100) | -4605.4 | 83782.12% | 2 | 5 | 4600 | 10100 |
| virus only (1000, 0) | -10005.3 | 182136.24% | 2 | 5 | 10000 | 10100 |
| drug only (0, 100) | -1005.5 | 18214.00% | 2 | 5 | 1000 | 10100 |
| full dose (1000, 100) | -11005.3 | 200350.24% | 2 | 5 | 11000 | 10100 |
| two-stage (1000,100) then (0,100) at day 10 | -11005.3 | 200350.24% | 2 | 5 | 11000 | 10100 |

Optimal u1 by day (first 6): 0, 0, 0, 0, 0, 0

Optimal u2 by day (first 6): 0, 0, 0, 0, 0, 0

### 6D Default with lysis (training model of Model 1)

Weights (A1, A2) = (0.01, 0.1). Optimum found from 5 starting schedules; their final costs agree to 0.0e+00 (relative).

| schedule | return | gap to optimum | days to U+I<0.1 | tumour cost | dose cost | peak U+I |
|---|---|---|---|---|---|---|
| numerical optimum | -4.2 | 0.00% | 2 | 4 | 0 | 10100 |
| no dosing (0, 0) | -4.2 | 0.00% | 2 | 4 | 0 | 10100 |
| half dose (500, 50) | -2754.2 | 64900.03% | 2 | 4 | 2750 | 10100 |
| best constant in the paper (600, 100) | -4604.2 | 108559.91% | 2 | 4 | 4600 | 10100 |
| virus only (1000, 0) | -10004.2 | 236001.79% | 2 | 4 | 10000 | 10100 |
| drug only (0, 100) | -1004.2 | 23599.27% | 2 | 4 | 1000 | 10100 |
| full dose (1000, 100) | -11004.2 | 259601.06% | 2 | 4 | 11000 | 10100 |
| two-stage (1000,100) then (0,100) at day 10 | -11004.2 | 259601.06% | 2 | 4 | 11000 | 10100 |

Optimal u1 by day (first 6): 0, 0, 0, 0, 0, 0

Optimal u2 by day (first 6): 0, 0, 0, 0, 0, 0


## b. Optimality gap of every method on Test 2 (Regime A)

| method | Test 2 return | gap to optimum | kind |
|---|---|---|---|
| numerical optimum (this script) | -1149610 | 0.0% | reference |
| best single DRL run (Model 2 PPO, seed 42) | -1149942 | 0.0% | DRL |
| PPO, trained on Model 2 (mean of 5 seeds) | -1150492 | 0.1% | DRL |
| TRPO, trained on Model 2 (mean of 5 seeds) | -1150817 | 0.1% | DRL |
| two-stage (1000,100) then (0,100) at day 10 | -1165143 | 1.4% | open-loop |
| TD3, trained on Model 2 (mean of 5 seeds) | -1174969 | 2.2% | DRL |
| SAC, trained on Model 2 (mean of 5 seeds) | -1182071 | 2.8% | DRL |
| manuscript DP, 50^4 linear grid | -1211746 | 5.4% | DP |
| manuscript DP, 30^4 linear grid | -1237374 | 7.6% | DP |
| best constant in the paper (600, 100) | -1281245 | 11.5% | open-loop |
| half dose (500, 50) | -1318721 | 14.7% | open-loop |
| manuscript DP, 20^4 linear grid | -1319462 | 14.8% | DP |
| full dose (1000, 100) | -1331055 | 15.8% | open-loop |
| manuscript DP, 15^4 linear grid | -1359943 | 18.3% | DP |
| virus only (1000, 0) | -1400399 | 21.8% | open-loop |
| drug only (0, 100) | -1410354 | 22.7% | open-loop |
| manuscript DP, 10^4 linear grid | -1442529 | 25.5% | DP |
| no dosing (0, 0) | -1503323 | 30.8% | open-loop |

## c. How much could a feedback policy gain over one fixed schedule?

Test 1 (5% initial-condition noise, 1% observation noise; the manuscript's 5 x 10 episodes; mean ± SD over seed cohorts):

| policy | Test 1 return | gap to the per-episode bound |
|---|---|---|
| no dosing | -1500825 ± 13065 | 30.719% |
| half dose (500, 50) | -1316683 ± 10855 | 14.681% |
| two-stage | -1163664 ± 9681 | 1.353% |
| one fixed schedule: the nominal optimum | -1148140 ± 9571 | 0.001% |
| bound: schedule re-optimised for each episode's true initial state | -1148126 ± 9573 | 0.000% |

The manuscript's ten perturbed parameter sets (one noisy episode each):

| policy | mean return | SD across sets | gap to the per-set bound |
|---|---|---|---|
| no dosing | -1536403 | 295125 | 30.96% |
| half dose (500, 50) | -1345244 | 238385 | 14.66% |
| two-stage | -1188893 | 202172 | 1.34% |
| one fixed schedule: the nominal optimum | -1173626 | 203284 | 0.03% |
| bound: schedule re-optimised for each parameter set | -1173229 | 203309 | 0.00% |

The last row of each table re-optimises the whole schedule with exact knowledge of that episode's initial state
(or parameter set). No feedback policy can do better, so the difference between the last two rows is the largest
improvement closed-loop control can show in these tests.

## d. Test 4 and the 4D Modified model

With nu_U = nu_I = tau = 0 the immune variables no longer act on U, I, V or C. Under four constant schedules the
(U, I, V, C) trajectories of Test 4 and of the 4D Modified model differ by at most 5.8e-07 (relative).

Total run time 254 s.
