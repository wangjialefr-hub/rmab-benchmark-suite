# LP-Update V2 Replacement Notes

Bruno mentioned a newer LP-Update method developed with Nicolas Gast and Chen
Yan. The current code still uses the original rolling-horizon LP-Update:

- homogeneous implementation: `strategies.LPupdateStrategy`
- factory entry: `make_policy.py`, `LPupdateStrategy`
- heterogeneous implementation: `heterogeneous_rmab.HeterogeneousLPUpdatePolicy`

## Current LP-Update

The current policy solves a finite-horizon LP from the current empirical state
distribution and uses the first-stage decision. This can be expensive because a
new LP may be solved repeatedly as the state distribution changes.

## Expected V2 Goal

The new method should avoid solving a fresh LP at every step. Possible
implementation hooks:

1. Add a new class in a separate file, for example `lpupdate_v2.py`.
2. Add a new policy name in `make_policy.py`, for example `LPupdateStrategyV2`.
3. Keep the original `LPupdateStrategy` for comparison.
4. Add `LPupdateStrategyV2` to:
   - `run_instance_matrix_benchmark.py`
   - `run_computation_cost_benchmark.py`
   - `run_computation_cost_suite.py`
5. Compare old vs new LP-Update on:
   - reward;
   - relative gap;
   - setup time;
   - online simulation time.

## Needed Before Implementation

Do not implement this from memory. Wait for one of:

- paper draft;
- algorithm pseudocode;
- code from Bruno/Nicolas/Chen Yan;
- exact recurrence/update rule.

Once the new method is available, the report should include a dedicated
comparison:

```text
Old LP-Update vs New LP-Update:
    reward
    relative gap
    online computation time
    number of LP solves
```
