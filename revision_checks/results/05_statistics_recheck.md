# 05 - Statistical statements re-checked

## 1. Agent versus half-dose baseline: paired samples, and what five seeds can show

Both columns are evaluated on the same episodes, so the baseline's five "seed" values are five evaluations of one
fixed policy on shared initial states, and the agent's values share those initial states. The Mann-Whitney test in
Table 9 treats the two columns as independent samples. The median correlation between PPO/TRPO seed-level returns and
the baseline's is 0.974.

With five values per group the attainable p-values are coarse. The smallest two-sided Mann-Whitney value is 0.0079
(all five values of one group beyond all five of the other); the smallest exact Wilcoxon signed-rank value for five
paired differences is 0.0625. Of 48 agent-versus-baseline comparisons on Tests 1, 3 and 4, Mann-Whitney gives
p < 0.05 in 47. The paired test cannot give p < 0.05 at n = 5 (0 of 48). In 47 of the 48 comparisons all five
paired differences have the same sign, which gives its smallest value, 0.0625.

Ignoring the pairing does not inflate significance. With no true difference, the Mann-Whitney test at the 0.05 level
rejects in 3.2% of simulated trials for independent groups, 0.03% at correlation 0.9 and 0.00% at
correlation 0.999 (200,000 trials each). What the pairing changes is the reading of the table:

- p = 0.008 says only that the two groups of five do not overlap. It is the same number for a 5% and a 99% difference.
- For PPO, TRPO and the constant-corner TD3/SAC runs (check 01), both columns are fixed policies scored on the same
  episodes. The difference is then a property of the ODE model, and the "between-seed" SD is evaluation noise from
  the shared initial states, not variability between trained agents.
- The paired differences (mean and range over seeds) are the informative summary. A paired exact test needs at least
  six seeds to reach p < 0.05.

| trained on | algorithm | test | Mann-Whitney p (Table 9) | paired Wilcoxon p | correlation with baseline |
|---|---|---|---|---|---|
| Model 1 | PPO | Test 1 | 0.008 | 0.0625 | 0.987 |
| Model 1 | PPO | Test 3 | 0.008 | 0.0625 | 1.000 |
| Model 1 | PPO | Test 4 | 0.008 | 0.0625 | 0.999 |
| Model 1 | TRPO | Test 1 | 0.008 | 0.0625 | 0.772 |
| Model 1 | TRPO | Test 3 | 0.008 | 0.0625 | 1.000 |
| Model 1 | TRPO | Test 4 | 0.008 | 0.0625 | 0.999 |
| Model 1 | TD3 | Test 1 | 0.008 | 0.0625 | 0.987 |
| Model 1 | TD3 | Test 3 | 0.008 | 0.0625 | 1.000 |
| Model 1 | TD3 | Test 4 | 0.008 | 0.0625 | 0.999 |
| Model 1 | SAC | Test 1 | 0.008 | 0.0625 | 0.942 |
| Model 1 | SAC | Test 3 | 0.008 | 0.0625 | 0.323 |
| Model 1 | SAC | Test 4 | 0.008 | 0.0625 | 0.999 |
| Model 2 | PPO | Test 1 | 0.008 | 0.0625 | 0.992 |
| Model 2 | PPO | Test 3 | 0.008 | 0.0625 | -0.480 |
| Model 2 | PPO | Test 4 | 0.008 | 0.0625 | 0.974 |
| Model 2 | TRPO | Test 1 | 0.008 | 0.0625 | 0.991 |
| Model 2 | TRPO | Test 3 | 0.008 | 0.0625 | -0.887 |
| Model 2 | TRPO | Test 4 | 0.008 | 0.0625 | 0.935 |
| Model 2 | TD3 | Test 1 | 0.008 | 0.0625 | -0.192 |
| Model 2 | TD3 | Test 3 | 0.008 | 0.0625 | -0.183 |
| Model 2 | TD3 | Test 4 | 0.008 | 0.0625 | 0.819 |
| Model 2 | SAC | Test 1 | 0.008 | 0.0625 | 0.827 |
| Model 2 | SAC | Test 3 | 0.008 | 0.0625 | 0.106 |
| Model 2 | SAC | Test 4 | 0.008 | 0.0625 | 0.909 |
| Model 3 | PPO | Test 1 | 0.008 | 0.0625 | 0.825 |
| Model 3 | PPO | Test 3 | 0.008 | 0.0625 | 0.865 |
| Model 3 | PPO | Test 4 | 0.008 | 0.0625 | 0.998 |
| Model 3 | TRPO | Test 1 | 0.008 | 0.0625 | -0.339 |
| Model 3 | TRPO | Test 3 | 0.151 | 0.1875 | -0.156 |
| Model 3 | TRPO | Test 4 | 0.008 | 0.0625 | 0.995 |
| Model 3 | TD3 | Test 1 | 0.008 | 0.0625 | 0.590 |
| Model 3 | TD3 | Test 3 | 0.008 | 0.0625 | -0.244 |
| Model 3 | TD3 | Test 4 | 0.008 | 0.0625 | 0.989 |
| Model 3 | SAC | Test 1 | 0.008 | 0.0625 | 0.875 |
| Model 3 | SAC | Test 3 | 0.008 | 0.0625 | 0.278 |
| Model 3 | SAC | Test 4 | 0.008 | 0.0625 | 0.993 |
| Model 4 | PPO | Test 1 | 0.008 | 0.0625 | 0.975 |
| Model 4 | PPO | Test 3 | 0.008 | 0.0625 | -0.821 |
| Model 4 | PPO | Test 4 | 0.008 | 0.0625 | 0.999 |
| Model 4 | TRPO | Test 1 | 0.008 | 0.0625 | 0.677 |
| Model 4 | TRPO | Test 3 | 0.008 | 0.0625 | 0.325 |
| Model 4 | TRPO | Test 4 | 0.008 | 0.0625 | 0.999 |
| Model 4 | TD3 | Test 1 | 0.008 | 0.0625 | -0.388 |
| Model 4 | TD3 | Test 3 | 0.008 | 0.0625 | -0.074 |
| Model 4 | TD3 | Test 4 | 0.008 | 0.0625 | 0.996 |
| Model 4 | SAC | Test 1 | 0.008 | 0.0625 | 0.941 |
| Model 4 | SAC | Test 3 | 0.008 | 0.0625 | -0.410 |
| Model 4 | SAC | Test 4 | 0.008 | 0.0625 | 0.999 |

## 2. Holm adjustment inside each six-pair cell

Holm compares the k-th smallest p-value with alpha / (6 - k + 1). Four tied values of 0.00794 are compared with
0.00833, 0.0100, 0.0125 and 0.0167 and all four are rejected (adjusted p = 0.0476 each).
6 of the 12 cells keep more than one comparison below 0.05 after adjustment.

| trained on | test | pairs with raw p < 0.05 | pairs with Holm-adjusted p < 0.05 | smallest adjusted p |
|---|---|---|---|---|
| Model 1 | Test 1 | 2 | 2 | 0.0476 |
| Model 1 | Test 3 | 3 | 3 | 0.0476 |
| Model 1 | Test 4 | 0 | 0 | 1.0000 |
| Model 2 | Test 1 | 4 | 4 | 0.0476 |
| Model 2 | Test 3 | 2 | 2 | 0.0476 |
| Model 2 | Test 4 | 0 | 0 | 0.5714 |
| Model 3 | Test 1 | 0 | 0 | 0.3333 |
| Model 3 | Test 3 | 3 | 2 | 0.0476 |
| Model 3 | Test 4 | 0 | 0 | 1.0000 |
| Model 4 | Test 1 | 3 | 1 | 0.0476 |
| Model 4 | Test 3 | 5 | 5 | 0.0476 |
| Model 4 | Test 4 | 0 | 0 | 1.0000 |

## 3. Between-seed SD on Test 1

| training model | SAC | TD3 |
|---|---|---|
| Model 1 (Full, Default, Noise) | 1.35e+04 | 1.31e+04 |
| Model 2 (Reduced, Default, Noise) | 1.44e+04 | 8.09e+03 |
| Model 3 (Reduced, Modified, Noise) | 2.57e+04 | 5.3e+04 |
| Model 4 (Reduced, Modified, No Noise) | 9.98e+03 | 6.77e+03 |

Model 2 and Model 4 differ in parameter set, hyperparameters and noise. The pair that differs only in training
noise is Model 3 (noise) and Model 4 (no noise).
