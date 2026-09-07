# RMAB Benchmark Paper Workflow

This folder contains a benchmark suite for RMAB experiments. The code is
organized around three experimental regimes:

1. known-model homogeneous RMAB;
2. known-model heterogeneous RMAB;
3. unknown-model online learning.

## 1. Main Experiments

Active code lives in the repository root. The `Code_Description*` folders are
historical copies and should not be used for final runs.

Run these from Jupyter or Anaconda Python.

### Known-model homogeneous benchmark

```python
%run "run_instance_matrix_benchmark.py"
```

Outputs:

```text
instance_matrix_outputs/
```

### Known-model heterogeneous benchmark

```python
%run "run_heterogeneous_benchmark.py"
```

Outputs:

```text
heterogeneous_outputs/
```

### Unknown-model online-learning benchmark

```python
%run "run_unknown_model_benchmark.py"
```

Outputs:

```text
unknown_model_outputs/
```

This benchmark includes:

- OnlinePlugInWhittle
- OnlineQLearningIndex
- OnlineUCBReward
- OnlineRewardGreedy
- OnlineRandom
- KnownWhittleOracle and KnownLPPriorityOracle as reference curves

### Computation-cost benchmark

To reuse existing cost results:

```python
from run_computation_cost_suite import collect_existing_computation_cost_outputs
collect_existing_computation_cost_outputs()
```

To rerun the full cost suite:

```python
%run "run_computation_cost_suite.py"
```

Outputs:

```text
computation_cost_suite_outputs/
```

## 2. Paper Tables and Figures

After running experiments, refresh all paper-level summaries:

```python
%run "export_instance_taxonomy.py"
%run "generate_benchmark_findings.py"
%run "generate_paper_figures.py"
%run "check_paper_readiness.py"
%run "generate_paper_draft.py"
```

Outputs:

```text
paper_summary_outputs/
```

Important generated files:

- `paper_draft_rmab_benchmark.md`
- `benchmark_findings_draft.md`
- `instance_taxonomy.csv`
- `instance_taxonomy.md`
- `paper_readiness_report.md`
- `paper_figure_index.csv`
- `figures/*.png`

## 3. What Can Be Claimed

Safe claims:

- The benchmark compares multiple RMAB algorithms under a unified pipeline.
- The benchmark includes known-model, heterogeneous, and unknown-model regimes.
- Policy rankings vary across instance structures.
- Computation cost changes the practical ranking of policies.
- Online learning baselines lag behind known-model oracle policies unless enough samples are collected.
- Paper-level aggregate conclusions exclude duplicate instance entries. `conveyor_eg4action-gap-tb_S8` is identical to `hong_counterexample`, so `hong_counterexample` is kept as the representative. Non-duplicate conveyor examples are retained.

Claims to be careful about:

- Do not claim all instances are directly from literature; some are synthetic or application-inspired.
- Do not call OnlinePlugInWhittle a state-of-the-art learning algorithm; it is a strong RMAB-specific baseline.
- Do not over-interpret negative relative gaps; explain finite-horizon transients and Monte Carlo noise.
- Do not say heterogeneous LP-Update is scalable without qualification; it can be expensive.
- Do not double-count `hong_counterexample` and `conveyor_eg4action-gap-tb_S8` in aggregate conclusions.

## 4. Recommended Paper Structure

1. Introduction and motivation
2. Problem setting
3. Benchmark design
4. Algorithms
5. Instance library
6. Metrics
7. Known-model experiments
8. Unknown-model online-learning experiments
9. Heterogeneous-arm experiments
10. Computation-cost analysis
11. Limitations and future work
12. Conclusion

## 5. Minimum Before Submission

Before sending the paper to your advisor, make sure:

- `unknown_model_summary.csv` exists and is not empty.
- `paper_readiness_report.md` has no missing critical experiment.
- Figures in `paper_summary_outputs/figures` are generated.
- The text clearly separates known P,R from unknown P,R.
- The instance taxonomy distinguishes literature examples from synthetic ones.
