# Running the Benchmark from Jupyter

This guide starts with a small installation check and then lists the full
experiments. Run only the sections needed for the result you want to reproduce.

## 1. Open the Repository

Change `ROOT` to the folder where the repository was cloned:

```python
from pathlib import Path
import os

ROOT = Path(r"C:\path\to\rmab-benchmark-suite")
os.chdir(ROOT)
print("Current folder:", Path.cwd())
```

## 2. Check the Installation

```python
%run "smoke_test.py"
```

The expected final messages are:

```text
Smoke test passed.
Cache behavior: miss -> hit
```

## 3. Try One Instance and Policy

```python
from benchmark_api import run_named_experiment

result = run_named_experiment(
    "random_S10_seed123",
    "WhittleIndexStrategy",
    N=100,
    horizon=200,
    seed=123,
)
result
```

Running the same cell again should return `cache_hit=True`.

## 4. Run the Experiment Families

Known-model homogeneous benchmark:

```python
%run "run_instance_matrix_benchmark.py"
```

Known-model heterogeneous benchmark:

```python
%run "run_heterogeneous_benchmark.py"
```

Unknown-model online-learning benchmark:

```python
%run "run_unknown_model_benchmark.py"
```

Computation-cost benchmark:

```python
%run "run_computation_cost_suite.py"
```

The first homogeneous run can take a long time. It includes expensive policies
and many instance-policy-seed combinations. Completed reward simulations are
stored in `rmab_cache/`, so an unchanged rerun can reuse them. Timing experiments
are deliberately uncached.

## 5. Regenerate Report Outputs

Run this block after the required experiments have completed:

```python
%run "detect_duplicate_instances.py"
%run "export_instance_taxonomy.py"
%run "generate_benchmark_findings.py"
%run "generate_paper_figures.py"
%run "check_paper_readiness.py"
```

## Output Folders

| Folder | Contents |
|---|---|
| `instance_matrix_outputs/` | Homogeneous rewards, gaps, convergence fits, and plots |
| `heterogeneous_outputs/` | Heterogeneous rewards and runtimes |
| `unknown_model_outputs/` | Learning summaries and learning curves |
| `computation_cost_suite_outputs/` | Setup, online, and total runtime summaries |
| `paper_summary_outputs/` | Filtered tables and paper-level figures |
| `rmab_cache/` | Reusable homogeneous reward simulations |

All of these folders are generated locally and excluded from Git.
