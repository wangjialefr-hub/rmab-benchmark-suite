# Final RMAB Benchmark Status

The repository root is the active folder.

Historical folders such as `Code_Description*` are old copies and should not be
used for final runs.

## Main Entry Points

- `run_instance_matrix_benchmark.py`: known-model homogeneous benchmark.
- `run_heterogeneous_benchmark.py`: known-model heterogeneous-arm benchmark.
- `run_unknown_model_benchmark.py`: unknown-model online-learning benchmark.
- `run_computation_cost_suite.py`: computation-cost aggregation or rerun.
- `generate_paper_draft.py`: regenerate the draft report from CSV outputs.

## Final Known-Model Instances

The final known-model run includes 11 instances:

- `random_S10_seed123`
- `hong_counterexample`
- `yan_gast_example1`
- `yan_gast_example2`
- `yan_gast_example3`
- `conveyor_eg4unif-tb_S8`
- `conveyor_eg4archive1_S8`
- `conveyor_eg4archive2_S8`
- `maintenance_S10_a20`
- `wireless_channel_S10_a40`
- `deadline_S10_a20`

`conveyor_eg4action-gap-tb_S8` is excluded from paper-level analysis because it
duplicates `hong_counterexample` in the current construction.

## Final Policy Set

Known-model homogeneous:

- `WhittleIndexStrategy`
- `LPPriorityStrategy`
- `FTVA_Strategy`
- `LPupdateStrategy`
- `Myopic`
- `RandomPriority`
- `RoundRobin`
- `LPRandomized`
- `QWhittleKnownModel`

Unknown-model online-learning and heterogeneous policies are summarized in
`README_FOR_PAPER.md`.

## Important Caveats

- `FTVA_Strategy` fails on `maintenance_S10_a20` and `deadline_S10_a20`; treat
  this as an algorithm/instance compatibility limitation.
- Some final-N relative gaps are negative because finite-horizon simulation can
  slightly exceed the LP average-reward reference; explain this rather than
  treating it as a bug.
- Some convergence beta values are `NaN` when there are too few positive gap
  points to fit a log-log slope.
- `QWhittleKnownModel` uses `QWHITTLE_STEPS_PER_PENALTY = 50000`.

## Main Outputs

- `instance_matrix_outputs`: raw known-model results and per-instance plots.
- `heterogeneous_outputs`: heterogeneous-arm results.
- `unknown_model_outputs`: online-learning results.
- `computation_cost_suite_outputs`: combined cost results.
- `paper_summary_outputs`: paper-ready tables, figures, readiness checks, and
  `paper_draft_rmab_benchmark.md`.
