# What Has Been Checked

These are local verification results, not a claim that the full study has been
repeated. The simulator fingerprint in the recorded known-model runs is
`eeb3b494dafdc5389c204294`, version `integer-actions-ftva-v2`.

On 27 September 2026, the staged release files were also tested outside the
working directory, in a new Python 3.13 virtual environment using only
`requirements.txt`. The smoke test and all 58 regression tests passed. This
check caught and resolved a missing `tabulate` dependency in the report tools.

| Check | Completed runs | Scope |
|---|---:|---|
| Regression tests | 58 tests passed | Budgets, transitions, FTVA, caches, summaries, timing and online learning |
| Deadline comparison | 900 | 9 policies; N=20/50/100/200/500; 20 seeds; T=200 |
| Longer deadline FTVA | 15 | N=20/100/500; 5 seeds; T=4000 |
| Three-instance comparison | 243 | Deadline, maintenance, random; 9 policies; 3 N values; 3 seeds; T=200 |
| Library smoke check | 88 | 11 instances; 8 policies; N=20; one seed; T=30 |
| Fresh homogeneous timing | 96 + 48 setups | 3 instances; 8 policies; 2 N values; 2 repetitions; T=20 |
| Heterogeneous checks | 56 + 56 | Five shared types, then distinct per-arm models; 7 policies; T=10 |
| Online-learning check | 112 | 4 instances; 5 baselines and 2 known-model references; T=200 |

No simulation failed in these checks. The deadline subset overlaps with the
three-instance comparison; the counts are not independent evidence to add up.
The 243-run cache replay returned identical per-seed rewards. Short extension
checks establish that the code runs, not that a learner has converged or that
a runtime scaling law has been measured.

## Deadline and FTVA

The former FTVA exception came from dividing by zero when converting a zero-mass
LP state into action probabilities. The correction uses probability 1/2 for
both actions there, as specified by Hong et al., and starts virtual arms from
the LP stationary distribution. Successful execution does not establish the
assumptions of an asymptotic-optimality theorem.

At N=500 in the expanded check, FTVA's mean reward is 0.088074185 (T=200,
20 seeds). In the longer check it is 0.0870666595 (T=4000, five seeds).
It runs successfully but is not the best-reward policy on this example.
The two protocols do not isolate the effect of horizon alone.

The deadline stationary LP is 0.09. At N=500, T=200, the finite-horizon LP and
Myopic reward are both 0.091175 to solver precision. The stationary relative
difference is therefore about -1.31%. This is startup reward, not a violation
of the matched finite-horizon bound. Missing log-gap slopes and overlapping
curves must not be counted as failed simulations.

## Reproduction

```bash
python smoke_test.py
python -m unittest discover -s tests -v
python verify_deadline.py --long
```

For the other checks, use `verify_corrections.py --with-qwhittle`,
`verify_extensions.py`, and `verify_online_learning.py`. They write separate
folders under `verification_outputs/`. The Q-Whittle setting remains 50,000
updates per penalty, 41 penalties, discount 0.95 and training seed 2026.

## Still Open

The full eleven-instance, nine-policy, twenty-seed matrix has not been rerun.
Neither a universal policy recommendation nor all theoretical assumptions have
been established. The newer LP-Update discussed with the supervisors has not
been integrated. Larger heterogeneous and timing experiments remain necessary.
Historical rankings should not be presented as results of this corrected code.
