# RMAB Benchmark Suite

This repository contains the code developed for a benchmarking study of
restless multi-armed bandit (RMAB) policies. Its main purpose is to run several
policies under one experimental protocol, rather than compare numbers produced
by unrelated scripts.

The primary benchmark assumes that the transition and reward models are known.
The repository also contains smaller extensions for heterogeneous arms and
unknown-model online learning.

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

## Run the Full Benchmarks

The full sweeps are much slower than the smoke test. In particular, LP-Update
performs optimization during simulation and QWhittleKnownModel has a separate
training phase. Start with the smoke test or one API call before launching a
complete run.

| Script | Experiment | Output folder |
|---|---|---|
| `run_instance_matrix_benchmark.py` | Known-model homogeneous arms | `instance_matrix_outputs/` |
| `run_heterogeneous_benchmark.py` | Known-model heterogeneous arms | `heterogeneous_outputs/` |
| `run_unknown_model_benchmark.py` | Unknown-model online learning | `unknown_model_outputs/` |
| `run_computation_cost_suite.py` | Setup and online computation time | `computation_cost_suite_outputs/` |
| `generate_paper_figures.py` | Paper-level summary figures | `paper_summary_outputs/figures/` |

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

- `bandit_lp.py`, `strategies.py`: reference RMAB classes, policies, and simulator.
- `make_policy.py`: common policy factory and added baselines.
- `rmab_instances.py`, `known_model_extra_instances.py`: named instance library.
- `simulation_cache.py`: cache keyed by the model and experiment settings.
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
- The reported convergence coefficient is a descriptive log-log slope over the
  tested values of `N`; it is not a proof of an asymptotic rate.
- A small negative empirical LP gap can arise from finite-horizon transients or
  Monte Carlo error.
- Unsupported policy-instance combinations are recorded as failures rather than
  silently removed from the result files.

The known-model benchmark is the main contribution. The heterogeneous and
unknown-model experiments are useful extensions, but their instance sets and
evaluation protocols are intentionally smaller.

## Reference Code

`bandit_lp.py` and `strategies.py` were provided by Nicolas Gast and are included
with his permission. The benchmark framework and extensions were developed
around those files. See `ATTRIBUTION.md` for the redistribution note.
