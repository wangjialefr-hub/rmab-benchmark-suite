# Running the Benchmark from Jupyter

This guide starts with a small installation check and then lists the full
experiments. Run only the sections needed for the result you want to reproduce.

After updating Python source files, restart the Jupyter kernel before running
these cells. For the September corrections and validation runs, start with
`AUDIT_GUIDE_ZH.md` or `CORRECTIONS.md`.

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

For the corrected deadline comparison:

```python
%run "verify_deadline.py"
```

This includes nine policies, five N values, twenty seeds and 200 steps.
Use `%run "verify_deadline.py" --long` to add the slower 4,000-step FTVA check.
Results are in `verification_outputs/release_v2/`; old caches are not deleted.

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

For a smaller online check (four instances, seven policies, two N values and
two seeds, 200 steps), run `%run "verify_online_learning.py"`. Outputs go to
`verification_outputs/corrected_online_v2/`; the full runner uses
`unknown_model_outputs/corrected_v2/`. These do not overwrite historical CSVs.

Online policies observe all arms' outcomes, including passive arms, and pool
samples across homogeneous arms. Plug-in Whittle estimates a model; the Q
baseline ranks by discounted Q-value differences and is not Q-Whittle.
Known-model oracles are explicitly labeled. Setup and online simulation time
are recorded separately. The tail metric uses the last 25% of steps by default.

Computation-cost benchmark:

```python
%run "run_computation_cost_suite.py"
```

The corrected suite defaults to known-model homogeneous and heterogeneous
policies. Unknown-model timing is exploratory and requires `run_unknown=True`
when calling its Python function. It is not automatically imported from old CSVs.

For a small uncached check of both timing and heterogeneous code:

```python
%run "verify_extensions.py"
```

This writes to `verification_outputs/corrected_extensions_v2/`. It tests all
seven heterogeneous policies with five shared types and with distinct per-arm
models. Its short horizons are for functional verification, not final comparisons.

The first homogeneous run can take a long time. It includes expensive policies
and many instance-policy-seed combinations. Completed reward simulations are
stored in `rmab_cache/`, so an unchanged rerun can reuse them. Timing experiments
are deliberately uncached.

Corrected homogeneous outputs now go to `instance_matrix_outputs/corrected_v2/`.
The historical files in `instance_matrix_outputs/` are not overwritten. Old
simulation caches are retained but are not used by the corrected simulator.

## 5. Regenerate Report Outputs

After completing the corrected homogeneous experiment, run:

```python
%run "generate_benchmark_findings.py"
%run "generate_paper_figures.py"
%run "check_paper_readiness.py"
```

These commands read `instance_matrix_outputs/corrected_v2/` and write
`paper_summary_outputs/corrected_v2/`. Missing or inconsistent corrected data
raise an error rather than silently loading historical CSVs. Timing,
heterogeneous and learning results are not automatically mixed into this report.

To preview only the small verification run, explicitly select it:

```python
%run "generate_benchmark_findings.py" --known-model-dir verification_outputs/corrected_v2/targeted --output-dir verification_outputs/corrected_v2/report_preview
%run "generate_paper_figures.py" --known-model-dir verification_outputs/corrected_v2/targeted --output-dir verification_outputs/corrected_v2/report_preview
%run "check_paper_readiness.py" --known-model-dir verification_outputs/corrected_v2/targeted --output-dir verification_outputs/corrected_v2/report_preview
```

The preview is labeled a preliminary subset, not the full paper experiment.
`generate_paper_draft.py` remains a historical text template and now requires
explicit `--legacy` opt-in; it must not regenerate an allegedly corrected paper.

## Output Folders

| Folder | Contents |
|---|---|
| `instance_matrix_outputs/corrected_v2/` | Corrected homogeneous rewards, signed gaps, convergence fits, and plots |
| `heterogeneous_outputs/corrected_v2/` | Corrected heterogeneous rewards and runtimes |
| `unknown_model_outputs/corrected_v2/` | Online rewards, split timers, fallback diagnostics and learning curves |
| `computation_cost_outputs/corrected_v2/` | Standalone homogeneous timing |
| `computation_cost_suite_outputs/corrected_v2/` | Selected families' setup, simulation, and estimated total times |
| `paper_summary_outputs/corrected_v2/` | Checked homogeneous tables, summary figures and input manifest |
| `rmab_cache/` | Reusable homogeneous reward simulations |

All of these folders are generated locally and excluded from Git.
