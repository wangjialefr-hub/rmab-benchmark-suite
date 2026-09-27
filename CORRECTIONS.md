# Simulator and FTVA Corrections

Verified on 19 September 2026. This change affects simulation results. Historical caches and CSVs
remain available, but should not be mixed with corrected trajectories.

## What Was Wrong

1. `bandit_lp.next_x_from_y` computed rewards from fractional actions while
   drawing transitions from integer actions. It also added a rounding epsilon
   after `floor`, so values just below an integer lost an activation.
2. Homogeneous FTVA divided by zero in states with zero LP occupation. The
   `markovianbandit` import changed NumPy's global error settings, which exposed
   this as a `FloatingPointError` during policy construction.
3. The reported gap compared a finite simulation, including startup, against
   a stationary average-reward LP. That reference is not generally an upper
   bound on a finite run from the specified initial state.
4. Log-gap plots replaced zero and negative differences with `1e-12` without
   labeling the transformation. This hid the signs and made curves overlap.
5. The outer cache's `force_recompute=True` could still hit the simulator's
   inner cache.

## Implemented Changes

### Integer Actions

`simulation_utils.py` converts state distributions to exactly N arms using
largest remainders. The action allocation preserves these state counts and
exactly `floor(alpha*N)` activations, with a small tolerance for floating-point
representations of integers. Remaining activations go to the largest action
remainders; ties use state order. Both reward and transitions use that allocation.
The saved `y_values` now contain the actions actually executed.

The public API and matrix runner pass the effective fraction
`floor(alpha*N)/N` to policies and LP references. For the default paper settings,
alpha*N is already integral. Direct low-level users should use the same fraction
when constructing a policy and its LP reference.

FTVA and RoundRobin reconstruct individual states from the exact state counts.
Their internal states reset before a new simulation. The uncached timing loop
uses the same simulation step as the reward experiment.

### FTVA

[Hong et al., equation (8)](https://arxiv.org/html/2306.00196#S3.SS1) defines
the activation probability as 1/2 when the stationary occupation of a state is
zero. The implementation now uses that rule, including for heterogeneous FTVA.
Algorithm 1 initializes virtual arms independently from the LP stationary
distribution; the previous implementation copied the real initial states.
Real arms still use the experiment's initial distribution.

These are two separate corrections. The first prevents undefined probabilities;
the second aligns initialization with the paper. They do not establish the
synchronization or other assumptions of an asymptotic-optimality theorem for
every supplied model. Budget-repair tie-breaking is otherwise unchanged.

The NumPy error settings are restored after importing `markovianbandit`.
Whittle computation uses its required error settings in a local context.

### LP and Model Validation

LP failures now raise an explicit exception instead of returning solver values
from a non-optimal solve. Transition rows and rewards are checked. Rows whose
sums differ from 1 by at most `1e-7` are accepted as decimal rounding; residuals
larger than `1e-12` are normalized. The original discrepancy is available as
`bandit.transition_row_error`. Invalid rows are rejected, not normalized.

The heterogeneous LP budget also uses the actual integer activation fraction.
No LP-Priority ranking logic or LP-Update optimization objective was replaced.
This remains the existing LP-Update implementation, not the unpublished/new
version discussed with the supervisors.

### Metrics and Plots

- `mean_reward`: average over measured steps, after optional burn-in.
- `mean_relative_gap`: signed difference to the stationary LP, normalized by
  the absolute LP value. The existing CSV column name is retained for compatibility.
- `mean_full_horizon_reward`: average over all simulated steps, including burn-in.
- `mean_finite_horizon_relative_gap`: signed difference between the finite-time
  LP and that full-horizon mean. The LP uses the same rounded initial state and
  the same integer budget. It bounds expected reward, not each random sample.
- `burn_in=0` remains the default. Burn-in is configurable, not claimed to be
  sufficient for every instance. Increasing it does not by itself prove stationarity.

Gap curves retain negative and zero values. If necessary, the stationary plot
uses a symmetric-log scale with a linear region between -0.001 and 0.001.
There is no confidence-interval shading. Distinct markers and line styles help
identify overlaps; identical results are not artificially separated.

Beta remains a descriptive finite-range log-log slope. At least three positive
resolved points are required. When replicate standard deviations are available,
the pointwise 95% t-interval must exclude zero. The CSV records excluded points
and the reason for an unavailable fit. This selection is not a confidence
interval or an asymptotic guarantee for beta.

### Caches and Output Locations

New simulation keys include `simulation-cache-v2`, a simulation version,
source-code fingerprint and burn-in. The outer cache bypasses the inner cache.
`force_recompute=True` really executes a new simulation. A missing seed disables
simulation caching. Old files are not deleted.

Direct calls to `strategies.simulate` also use a new versioned filename.
Stationary LP caches can be reused when the model hash is unchanged, since the
stationary LP definition was not changed.

- Full corrected matrix: `instance_matrix_outputs/corrected_v2/`.
- Validation experiments: `verification_outputs/corrected_v2/`.
- Per-policy success/failure counts: `policy_run_status.csv` in each run folder.
- Experiment settings and source fingerprint: `experiment_settings.json`.

The paper aggregation scripts now read the corrected directory by default.
`--known-model-dir` selects a different completed corrected experiment explicitly.
They validate the code fingerprint, raw experiment keys and protocol, signed
gap formula, replicate counts, reward means and standard deviations. They do
not fall back to historical results when corrected data are missing.

Rank comparisons use shared instances with complete results for all selected
policies. Negative gaps remain visible in the summary heatmap. The earlier
weighted recommendation score has been removed: reward and gap duplicate the
same ordering, and an unavailable beta is not evidence of poor performance.
The current tables are descriptive, not an automatic recommendation.

Report outputs record their data and source hashes in `report_inputs.json`.
Partial experiments are labeled as preliminary. Historical cost, heterogeneous
and learning outputs are not automatically merged with corrected rewards.
The old manuscript template requires explicit `--legacy` opt-in. The readiness
check reports coverage and unverified components, not publication readiness.

## Reproduce the Verification

```bash
python -m unittest discover -s tests -v
python verify_corrections.py
```

The second command runs the regression tests, all 11 main instances at N=20
with 8 policies, and targeted deadline/maintenance/random experiments at
N=20,100,500, horizon 200, with 3 seeds. It also checks the deterministic
deadline finite-time LP at horizons 20,50,100,200,500. A second pass must hit
all targeted caches and reproduce every per-seed reward exactly. Read the generated
`RESULTS.md` and `verification_summary.json` for the observed outcomes.

Q-Whittle's 50,000 training steps per penalty are unchanged; the verification
script excludes its expensive retraining by default and records that omission.
Use `python verify_corrections.py --with-qwhittle` to include it on the three
targeted instances. These outputs go to `verification_outputs/corrected_v2_with_qwhittle/`.
They reuse compatible validation caches, but never old-version trajectories.
The verification command also generates a labeled summary preview in its
`report_preview/` folder, including signed gaps and a data-integrity report.
This verification is not the full 20-seed paper experiment. Historical result
comparisons are descriptive because protocols and replication counts can differ.

## September 20: Timing and Heterogeneous Runners

- Heterogeneous LP-Update previously retained its population-state solve cache
  across simulation seeds. `reset()` now clears it. Reuse within one trajectory
  remains enabled; optimization and action selection were not changed.
- The heterogeneous simulator validates binary actions before converting them
  to integers. Fractional or negative outputs are not silently interpreted as
  feasible actions.
- Both runners retain failed raw rows even when every policy construction fails.
  Raw files are checkpointed after each policy. Empty/invalid experiment axes
  are rejected before running.
- Homogeneous timing uses the actual integer-budget activation fraction. Cold
  setup and runtime fits are grouped by that fraction, so rounding does not mix
  different policy parameters into one comparison.
- Corrected timing and heterogeneous defaults write to `corrected_v2/`
  subdirectories. Settings include source hashes and timer boundaries. The
  timing collector rejects missing/stale provenance and no longer loads
  historical unknown-model results by default.
- Heterogeneous preprocessing no longer benefits from a model-grouping cache
  left by the preceding policy. Setup is still measured once per configuration;
  this is not enough to establish a reliable setup-time scaling law.

`python verify_extensions.py` ran 96 fresh timing simulations and 48 cold setups,
plus 56 heterogeneous simulations for each of five shared types and fully
distinct per-arm models. No runs failed. All seven heterogeneous policies were
included. Cold Q-Whittle training was explicitly excluded from this short audit.
At that stage the regression suite had 44 passing tests. Exact settings and raw data
are saved in `verification_outputs/corrected_extensions_v2/`.

## Online Learning and Interpretation

### Online-learning protocol audit

The online policies' factory receives S,A,N and the budget, not the true P,R
arrays. Regression tests forbid access to those arrays along the unknown-policy
construction and update paths. The two known-model reference oracles are the
explicit exception and are now labeled correctly in combined timing tables.

The simulator now checks dimensions, binary actions, probability distributions,
tail fractions and the actual integer budget. Known LP references use that
budget fraction. Each seed resets the learner. Reward/transition observations
include passive arms, rewards are deterministic R[s,a], and learned tables pool
all N homogeneous arms. This feedback assumption is recorded in metadata.

Online setup and simulation timers are separate; their sum retains the existing
runtime_seconds column. Previously, construction was included in the runtime
but the cost suite labeled the whole interval as online time. Plug-in Whittle's
estimated-model solver failures and reward-advantage fallback decisions are now
counted and exported, rather than silently presented as pure Whittle execution.

The Q update formula is unchanged. OnlineQLearningIndex learns a discounted
single-arm Q-difference heuristic, not a Whittle subsidy. Plug-in Whittle is
model-based online learning, not model-free. Learning curves retain their
per-step reward meaning; sampled times now start at step 1 and include the last
step. They are not cumulative reward or regret curves.

Corrected online results are separate in unknown_model_outputs/corrected_v2/.
Unknown-policy failures are retained, incomplete groups are marked and excluded
from plots, and settings/source hashes accompany the CSV files. Run
`python verify_online_learning.py` for the 112-run short verification matrix.

### Scientific wording

The earlier statement that FTVA simply cannot run on deadline/maintenance is
incorrect for this implementation after the correction. The old failure arose
from undefined zero-occupancy action probabilities. Revised results should use
newly generated data. Retain a distinction between successful execution and a
theoretical performance guarantee.

Do not describe a negative empirical stationary difference as an LP-bound
violation. For the deterministic deadline example, the stationary LP is 0.09;
the 200-step finite-time LP and corrected Myopic reward are both 0.091175.
The signed stationary difference is therefore approximately -0.01306.

## September 27: Expanded Deadline Verification

A clean-environment check caught a missing `tabulate` dependency used by the
Markdown report tables. It is now declared in `requirements.txt`.

`verify_deadline.py --long` reproduces a 900-run comparison (nine policies,
five population sizes, 20 seeds, 200 steps), plus 15 FTVA runs of 4,000 steps.
All completed without failures. The current 58 regression tests also pass.
The deadline execution error is fixed, but FTVA does not give the highest
reward on this instance. See `VALIDATION.md` for measured values and remaining
work. No full-library performance ranking is claimed from these subsets.
