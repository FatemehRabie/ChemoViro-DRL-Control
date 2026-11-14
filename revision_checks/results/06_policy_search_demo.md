# 06 - Model-free policy search with scaled inputs and outputs (proof of concept)

A 4-16-2 feedback network on log-scaled observations, trained with the cross-entropy method (population 64, 12 elites,
120 generations, two noisy training episodes per generation, i.e. about 5,760 environment steps per generation) in the
Model 2 training setting. Not one of the manuscript's algorithms; shown only to establish that the task is learnable
without a model once inputs and outputs are scaled. Generation 10 corresponds to about 57,600 environment steps, close to
the manuscript's 50,000-step budget.

| training seed | nominal return, generation 1 | generation 10 | generation 30 | final | Test 1 (5 x 10 episodes) | Test 2 |
|---|---|---|---|---|---|---|
| 42 | -1.3086e+06 | -1.1838e+06 | -1.1630e+06 | -1.1511e+06 | -1.1496e+06 | -1.1511e+06 |
| 123 | -1.2987e+06 | -1.1676e+06 | -1.1512e+06 | -1.1502e+06 | -1.1487e+06 | -1.1502e+06 |
| 456 | -1.3251e+06 | -1.1922e+06 | -1.1528e+06 | -1.1502e+06 | -1.1488e+06 | -1.1502e+06 |
| 789 | -1.3350e+06 | -1.1804e+06 | -1.1525e+06 | -1.1502e+06 | -1.1487e+06 | -1.1502e+06 |
| 1024 | -1.2837e+06 | -1.1702e+06 | -1.1529e+06 | -1.1504e+06 | -1.1489e+06 | -1.1504e+06 |

Over the five training seeds: Test 2 -1.1504e+06 ± 3.74e+02; Test 1 -1.1489e+06 ± 3.74e+02.
References on Test 2: numerical optimum -1.1496e6, two-stage schedule -1.1651e6, half dose -1.3187e6, best stored
DRL configuration (Model 2 TD3, mean of 5 seeds) -1.3885e6, no dosing (= stored PPO/TRPO) -1.5033e6.
Each training run used about 691,200 environment steps.
