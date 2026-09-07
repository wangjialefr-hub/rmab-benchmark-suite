# Jupyter Run Guide

Use this file when running the benchmark from the Jupyter web interface.

## 0. Set Working Folder

Run this first:

```python
from pathlib import Path
import os

ROOT = Path(r"C:\path\to\rmab-benchmark-suite")
os.chdir(ROOT)
print("Current folder:", Path.cwd())
```

## 1. Known-Model Homogeneous Benchmark

This is the main benchmark for known transition matrix `P` and reward matrix
`R`.

```python
%run "run_instance_matrix_benchmark.py"
```

Output folder:

```text
instance_matrix_outputs/
```

This script can be slow the first time. Later runs reuse `rmab_cache`.

## 2. Heterogeneous-Arm Benchmark

Use this when arms have different `P_i, R_i`.

```python
%run "run_heterogeneous_benchmark.py"
```

Output folder:

```text
heterogeneous_outputs/
```

## 3. Unknown-Model Online-Learning Benchmark

Use this when policies do not know `P, R` and must learn from samples.

```python
%run "run_unknown_model_benchmark.py"
```

Output folder:

```text
unknown_model_outputs/
```

## 4. Computation-Cost Summary

Fast version, reusing existing timing CSV files:

```python
from run_computation_cost_suite import collect_existing_computation_cost_outputs
collect_existing_computation_cost_outputs()
```

Full rerun version, slower:

```python
%run "run_computation_cost_suite.py"
```

Output folder:

```text
computation_cost_suite_outputs/
```

## 5. Regenerate Paper Tables, Figures, and Draft

Run these after experiments:

```python
%run "detect_duplicate_instances.py"
%run "export_instance_taxonomy.py"
%run "generate_benchmark_findings.py"
%run "generate_paper_figures.py"
%run "check_paper_readiness.py"
%run "generate_paper_draft.py"
```

Output folder:

```text
paper_summary_outputs/
```

## 6. Files to Read Before Meeting

Read these in order:

1. `FINAL_STATUS.md`
2. `README_FOR_PAPER.md`
3. `paper_summary_outputs\RMAB_paper_initial_draft.md`
4. `paper_summary_outputs\paper_readiness_report.md`
