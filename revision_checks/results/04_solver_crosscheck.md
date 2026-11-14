# 04 - Training integrator (RK4) against evaluation integrator (RK45)

Worst case over four dosing schedules and three initial states (nominal and two 5% perturbations), Regime A.

| environment | return: RK4 vs RK45 | states > 1: RK4 vs RK45 | tumour burden > 1: RK4 vs RK45 | return: RK45 vs tight reference | within the stated 1.2e-4 |
|---|---|---|---|---|---|
| 4D Default (Model 2 training; Tests 1-2) | 2.4e-07 | 2.7e-06 | 2.2e-06 | 2.4e-07 | yes |
| 4D Modified (Models 3-4 training) | 3.3e-05 | 8.8e-05 | 8.8e-05 | 1.3e-07 | yes |
| 6D Default (Model 1 training) | 1.5e-04 | 1.5e-04 | 1.5e-04 | 3.8e-06 | NO |
| 6D Modified (Test 3) | 9.5e-05 | 1.4e-04 | 1.4e-04 | 1.5e-06 | NO |
| 6D Modified, zero lysis (Test 4) | 1.0e-05 | 2.8e-05 | 2.8e-05 | 1.3e-07 | yes |

RK4 return error against the tight reference in the 4D Modified model, nominal initial state, by sub-steps per day:

| sub-steps per day | worst relative return error |
|---|---|
| 20 | 1.9e-03 |
| 40 | 8.6e-06 |
| 80 | 2.6e-07 |
| 160 | 1.4e-08 |
