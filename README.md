# RMAB Benchmark Suite

Python experiments for finite-state restless multi-armed bandits (RMABs).
The suite runs several policies on the same instances, saves per-seed results,
and produces reward, LP-gap and runtime comparisons.

The primary benchmark assumes that the transition and reward models are known.
The repository also contains smaller extensions for heterogeneous arms and
unknown-model online learning.

**September 2026 correction:** previous simulations could score fractional
actions while executing rounded actions. FTVA also had undefined probabilities
at states with zero LP occupation. Both are fixed; old results must be rerun
before being compared with corrected trajectories. See [CORRECTIONS.md](CORRECTIONS.md)
for the changes and [VALIDATION.md](VALIDATION.md) for the experiments checked.

## Start Here

Clone the repository and install the dependencies:

```bash
git clone https://github.com/wangjialefr-hub/rmab-benchmark-suite.git
cd rmab-benchmark-suite
python -m pip install -r requirements.txt
```

Then run the quick end-to-end check:

```bash
python smoke_test.py
```

This executes one small experiment twice. The first run computes the result and
the second reuses it from a temporary cache. A successful installation prints
`Smoke test passed` and `Cache behavior: miss -> hit`.

For the regression suite:

```bash
python -m unittest discover -s tests -v
```

Use a separate Python environment for the project. Local verification used
Python 3.13 on Windows; the GitHub workflow also tests Python 3.11 and Linux.
The workflow status, not this sentence, indicates whether those remote jobs pass.

## Run One Experiment

`benchmark_api.py` is the simplest entry point. It can be used from a Python
script or a Jupyter notebook:

```python
from benchmark_api import available_policies, list_instances, run_named_experiment

print(available_policies())
print(list_instances().head(10).to_string(index=False))

result = run_named_experiment(
    "random_S10_seed123",
    "WhittleIndexStrategy",
    N=100,
    horizon=200,
    seed=123,
)

print("Mean reward:", result["mean_reward"])
print("Relative gap:", result["relative_gap"])
print("Cache hit:", result["cache_hit"])
```

To use a new homogeneous instance, call
`run_custom_experiment(P, R, policy_name, alpha=...)`. The transition tensor
`P` must have shape `(S, 2, S)`, and the reward matrix `R` must have shape
`(S, 2)`.

Each step activates exactly `floor(alpha*N)` arms. The code uses the actual
integer-budget fraction in its LP references. `relative_gap` is a signed
difference to the stationary LP: a negative finite-run value is possible and
is not clipped to zero. It does not mean that the policy beat the optimal
long-run reward.

## Reproduce the Deadline Check

```bash
python verify_deadline.py
```

This runs all nine policies at N=20, 50, 100, 200, 500, with 20 seeds and 200
steps. Results go to `verification_outputs/release_v2/deadline_matrix/`.
Add `--long` to include a separate 4,000-step FTVA check; this is slower.
Both checks completed without failures locally. Full settings and limits of
that result are in [VALIDATION.md](VALIDATION.md).

## Run the Full Benchmarks

The full sweeps are much slower than the smoke test. In particular, LP-Update
performs optimization during simulation and QWhittleKnownModel has a separate
training phase. Start with the smoke test or one API call before launching a
complete run.

| Script | Experiment | Output folder |
|---|---|---|
| `run_instance_matrix_benchmark.py` | Known-model homogeneous arms | `instance_matrix_outputs/corrected_v2/` |
| `run_heterogeneous_benchmark.py` | Known-model heterogeneous arms | `heterogeneous_outputs/corrected_v2/` |
| `run_unknown_model_benchmark.py` | Online learners and separately labeled known-model references | `unknown_model_outputs/corrected_v2/` |
| `run_computation_cost_benchmark.py` | Homogeneous setup and simulation time | `computation_cost_outputs/corrected_v2/` |
| `run_computation_cost_suite.py` | Known-model homogeneous and heterogeneous timing; online learning is opt-in | `computation_cost_suite_outputs/corrected_v2/` |
| `generate_paper_figures.py` | Checked homogeneous summary figures | `paper_summary_outputs/corrected_v2/figures/` |

From Jupyter:

```python
%cd "C:/path/to/rmab-benchmark-suite"
%run "run_instance_matrix_benchmark.py"
```

From a terminal:

```bash
python run_instance_matrix_benchmark.py
```

The generated data, figures, and caches are intentionally not committed to the
repository. See `JUPYTER_RUN_GUIDE.md` for the recommended execution order and
the location of each output.

## Repository Map

- `bandit_lp.py`, `strategies.py`: reference-based classes and policies, with the documented simulator corrections.
- `make_policy.py`: common policy factory and added baselines.
- `rmab_instances.py`, `known_model_extra_instances.py`: named instance library.
- `simulation_cache.py`: cache keyed by the model and experiment settings.
- `simulation_utils.py`: exact arm counts, budget rounding and FTVA completion.
- `tests/`, `verify_corrections.py`: regression tests and reproducible validation.
- `verify_deadline.py`: expanded deadline comparison and optional longer FTVA check.
- `verify_extensions.py`: small uncached timing and heterogeneous checks (five types and distinct per-arm models).
- `verify_online_learning.py`: short online-learning check with recorded index-solver fallbacks and split timers.
- `benchmark_metadata.py`: input checks and source/protocol records for uncached experiments.
- `report_data.py`: corrected-result validation and shared-instance ranking.
- `benchmark_api.py`: importable interface for named or user-supplied instances.
- `heterogeneous_rmab.py`: heterogeneous-arm models and policies.
- `unknown_model_learning.py`: exploratory online-learning policies.
- `run_*.py`: experiment runners.
- `generate_*.py`, `check_*.py`: report figures, tables, and consistency checks.

## Reproducibility Notes

- Seeds, horizons, and tested values of `N` are declared near the top of each runner.
- Reward experiments use `rmab_cache/`. Pass `force_recompute=True` through the
  public API when a fresh result is required.
- Timing experiments bypass the reward cache. Loading a cached file is not a
  measurement of policy execution time.
- Report scripts default to corrected homogeneous data and reject mixed or
  incomplete files. Use `--known-model-dir` for a smaller validation sweep;
  the report labels it as preliminary and does not import historical extensions.
- Report rankings are descriptive. The earlier weighted recommendation is no
  longer generated, and missing beta values are not interpreted as poor scores.
- The reported convergence coefficient is a descriptive log-log slope over the
  tested values of `N`; it is not a proof of an asymptotic rate.
- A signed empirical difference to the stationary LP can be negative because of
  startup transients or Monte Carlo error. The optional finite-horizon LP uses
  the same rounded initial state and time window; it bounds expected reward.
- Unsupported policy-instance combinations are recorded as failures rather than
  silently removed from the result files.

The known-model benchmark is the main contribution. The heterogeneous and
unknown-model experiments are useful extensions, but their instance sets and
evaluation protocols are intentionally smaller.

Two names need care: `RandomPriority` fixes a random **state** order in the
homogeneous code; it is not fresh uniform random activation. `LPRandomized`
uses LP activation masses with deterministic budget repair. These are project
baselines, not claims to reproduce every similarly named policy in the literature.
The online learners observe passive arms too. Plug-in Whittle estimates a model;
the online Q-difference baseline is not a Whittle-subsidy learner.

## Reference Code

`bandit_lp.py` and `strategies.py` were provided by Nicolas Gast and are included
with his permission. The benchmark framework and extensions were developed
around those files. See `ATTRIBUTION.md` for the redistribution note.
The reference files now include project corrections described in `CORRECTIONS.md`;
they should not be described as unchanged copies of the instructor's originals.
