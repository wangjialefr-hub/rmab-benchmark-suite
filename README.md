# RMAB Benchmark Suite

This repository provides a reproducible benchmark for restless multi-armed
bandit (RMAB) policies. It covers three regimes:

1. homogeneous arms with known transition and reward models;
2. heterogeneous arms with arm-dependent models;
3. online learning when the policy does not know the model.

The benchmark reports average reward, relative gap to the relaxed LP upper
bound, empirical finite-range convergence slopes, and computation time.

## Installation

Clone the repository, create an environment, and install the dependencies:

```bash
git clone https://github.com/wangjialefr-hub/rmab-benchmark-suite.git
cd rmab-benchmark-suite
python -m pip install -r requirements.txt
```

## Quick Start

The smallest public interface runs one named instance-policy pair:

```python
from benchmark_api import available_policies, list_instances, run_named_experiment

print(available_policies())
display(list_instances())

result = run_named_experiment(
    "random_S10_seed123",
    "WhittleIndexStrategy",
    N=100,
    horizon=200,
    seed=123,
)
print(result["mean_reward"], result["relative_gap"], result["cache_hit"])
```

Users can evaluate their own homogeneous known-model instance with
`run_custom_experiment(P, R, policy_name, alpha=...)`. Here `P` has shape
`(S, 2, S)` and `R` has shape `(S, 2)`.

## Full Experiments

From Jupyter, first change into the cloned repository and then run a script:

```python
%cd "C:/path/to/rmab-benchmark-suite"
%run "run_instance_matrix_benchmark.py"
```

Main entry points:

| Script | Experiment | Output folder |
|---|---|---|
| `run_instance_matrix_benchmark.py` | Known-model homogeneous benchmark | `instance_matrix_outputs/` |
| `run_heterogeneous_benchmark.py` | Known-model heterogeneous benchmark | `heterogeneous_outputs/` |
| `run_unknown_model_benchmark.py` | Unknown-model online learning | `unknown_model_outputs/` |
| `run_computation_cost_suite.py` | Computation-cost study | `computation_cost_suite_outputs/` |
| `generate_paper_figures.py` | Paper-level figures | `paper_summary_outputs/figures/` |

See `JUPYTER_RUN_GUIDE.md` for the complete execution order.

The same experiments can be started outside Jupyter:

```bash
python run_instance_matrix_benchmark.py
python run_heterogeneous_benchmark.py
python run_unknown_model_benchmark.py
python run_computation_cost_suite.py
```

## Project Structure

- `bandit_lp.py`, `strategies.py`: reference RMAB and policy implementations.
- `make_policy.py`: unified policy factory and added baselines.
- `rmab_instances.py`, `known_model_extra_instances.py`: instance library.
- `simulation_cache.py`: deterministic cache keyed by model and experiment settings.
- `heterogeneous_rmab.py`: heterogeneous-arm models and policies.
- `unknown_model_learning.py`: online-learning policies.
- `benchmark_api.py`: compact importable interface for external users.
- `paper_config.py`, `generate_*`, `check_*`: paper tables, figures, and checks.

## Reproducibility Notes

- Experiment seeds and horizons are declared near the top of each runner.
- Cached reward experiments are stored in `rmab_cache/`; set
  `force_recompute=True` in the relevant API when a fresh run is required.
- Timing experiments must bypass reward caches. Reusing a cached result measures
  file loading, not policy execution.
- A fitted convergence value is a descriptive log-log slope over the tested
  finite set of `N` values, not a proof of an asymptotic rate.
- A negative empirical LP gap can occur through finite-horizon transients or
  Monte Carlo error and should not be interpreted as defeating the LP bound.
- `FTVA_Strategy` is incompatible with some current maintenance and deadline
  instances; failed combinations are recorded instead of silently omitted.
- The current `LPupdateStrategy` is the available reference implementation; a
  newer LP-Update variant is listed as future work in `LPUPDATE_V2_TODO.md`.

## Attribution

`bandit_lp.py` and `strategies.py` were provided by Nicolas Gast and are
included here with his permission. See `ATTRIBUTION.md` for details.
