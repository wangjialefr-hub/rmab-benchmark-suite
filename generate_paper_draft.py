"""
Generate a first paper draft in Markdown.

The draft intentionally separates known-model planning, heterogeneous arms, and
unknown-model online learning, so the experimental claims stay clean.
"""

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "paper_summary_outputs"


def read_text_if_exists(path):
    path = Path(path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def table_or_placeholder(path, max_rows=12):
    df = read_csv_if_exists(path)
    if df.empty:
        return "_Table not available yet. Run the corresponding experiment script._"
    return df.head(max_rows).to_markdown(index=False)


def generate_paper_draft(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    known_top_table = table_or_placeholder(
        output_dir / "known_model_top_policies.csv"
    )
    recommendation_table = table_or_placeholder(
        output_dir / "aggregate_policy_recommendation.csv",
        max_rows=9,
    )
    beta_table = table_or_placeholder(output_dir / "top_convergence_beta.csv")
    cost_table = table_or_placeholder(output_dir / "combined_cost_ranking.csv")
    unknown_table = table_or_placeholder(
        output_dir / "unknown_model_top_policies.csv"
    )
    hetero_table = table_or_placeholder(
        output_dir / "heterogeneous_top_policies.csv"
    )

    taxonomy = read_csv_if_exists(output_dir / "instance_taxonomy.csv")
    if taxonomy.empty:
        taxonomy_summary = "_Instance taxonomy not generated yet._"
    else:
        taxonomy_summary = (
            taxonomy.groupby(["source_type", "structure"], as_index=False)
            .agg(num_instances=("instance", "count"))
            .to_markdown(index=False)
        )

    draft = f"""# A Unified Benchmarking Suite for Restless Multi-Armed Bandits

## Abstract

Restless multi-armed bandits (RMABs) provide a modeling framework for sequential resource allocation problems in which each arm evolves even when it is not selected. The literature contains many algorithms, including Whittle-index policies, LP-relaxation policies, finite-time virtual-arm policies, model predictive control methods, and learning-based heuristics. However, empirical comparisons are often conducted on instance families designed within a single paper, making it difficult to understand robustness across heterogeneous problem structures.

We develop a unified RMAB benchmarking suite that evaluates multiple policies on a shared collection of finite-state instances. The benchmark covers known-model homogeneous RMABs, known-model heterogeneous RMABs, and unknown-model online-learning baselines. We compare policies using average reward, relative gap to an LP upper bound, empirical convergence rate with respect to the number of arms, and computation cost. The resulting suite provides a reproducible platform for studying when different RMAB policies are robust, when they are instance-dependent, and what computational tradeoffs arise in practice.

## 1. Introduction

RMAB algorithms are often motivated by strong theoretical guarantees or good performance on specific application examples. In practice, however, a policy that performs well on one instance family may be less reliable on another. This motivates a benchmark that evaluates algorithms across multiple instance structures rather than only on a single source of examples.

This work focuses on the following questions:

1. How stable are policy rankings across random, counterexample, and application-inspired RMAB instances?
2. How quickly do policies approach the LP relaxation upper bound as the number of arms increases?
3. What is the computational cost of each policy, separating preprocessing from online decision making?
4. How large is the gap between known-model planning policies and unknown-model online-learning baselines?
5. How do policies behave when arms are heterogeneous and have different transition and reward models?

## 2. Contributions

The main contributions are:

- A modular Python benchmark suite built on top of existing RMAB implementations.
- A unified instance library containing random, counterexample, maintenance, wireless, deadline, and recovering-bandit instances.
- A comparison of known-model policies including Whittle Index, LP-Priority, FTVA, LP-Update, LPRandomized, Myopic, RandomPriority, RoundRobin, and Q-Whittle-style planning.
- A heterogeneous-arm simulator with explicit per-arm transition and reward models.
- Unknown-model online-learning baselines, including Online Q-learning and an RMAB-specific plug-in Whittle policy that learns empirical P,R before computing indices.
- Evaluation metrics for reward, relative gap, convergence rate, and computation cost.

## 3. Problem Setting

We consider finite-state RMABs with state space size S and two actions: passive and active. At each time step, a controller may activate at most an alpha fraction of N arms. In the homogeneous known-model setting, all arms share the same transition matrix P and reward matrix R. In the heterogeneous setting, arm i may have its own P_i and R_i. In the unknown-model setting, the environment is generated by a true P,R, but the policy only observes sampled transitions and rewards.

The benchmark distinguishes three experimental regimes:

- **Known-model homogeneous:** policies may use the true shared P,R.
- **Known-model heterogeneous:** policies may use each arm type's true P_i,R_i.
- **Unknown-model online learning:** policies do not use P,R and must learn from samples.

## 4. Algorithms

### 4.1 Known-model homogeneous policies

The benchmark includes WhittleIndexStrategy, LPPriorityStrategy, FTVA_Strategy, LPupdateStrategy, Myopic, RandomPriority, RoundRobin, LPRandomized, and QWhittleKnownModel. These policies are evaluated when the model is known.

### 4.2 Known-model heterogeneous policies

The heterogeneous simulator supports HeterogeneousWhittle, HeterogeneousLPPriority, HeterogeneousFTVA, HeterogeneousLPUpdate, HeterogeneousMyopic, RandomActivation, and RoundRobin. These policies operate on explicit arms and can compare arms with different P_i,R_i.

### 4.3 Unknown-model online-learning policies

The unknown-model benchmark includes OnlineQLearningIndex, OnlineUCBReward, OnlineRewardGreedy, OnlineRandom, and OnlinePlugInWhittle. OnlinePlugInWhittle is the most RMAB-specific learning baseline: it estimates P_hat and R_hat from observed samples, constructs an empirical bandit model, and computes Whittle indices from the learned model.

KnownWhittleOracle and KnownLPPriorityOracle are included only as reference curves; they are not unknown-model policies.

## 5. Instance Library

The benchmark uses the following instance taxonomy:

{taxonomy_summary}

Instances are separated by source type. Some are direct literature or public-code examples; others are synthetic robustness or application-inspired examples. This distinction is important when interpreting empirical conclusions.

## 6. Evaluation Metrics

We report:

- **Average reward:** empirical reward averaged across time and Monte Carlo replications.
- **Tail average reward:** for unknown-model learning, reward averaged over the final portion of the trajectory.
- **Relative gap to LP upper bound:** (LP upper bound - policy reward) divided by the absolute LP upper bound.
- **Convergence beta:** slope-based estimate of how the relative gap decreases as N grows.
- **Computation cost:** setup time, online simulation time, and estimated cold total time.

Some finite-horizon estimates can produce negative relative gaps due to transient effects or Monte Carlo noise. In plots, clipped relative gaps may be used for readability, while raw values remain available in CSV files.

The paper-level aggregate tables exclude duplicate instance entries. In particular, `conveyor_eg4action-gap-tb_S8` is identical to `hong_counterexample` under the current finite-state P,R construction, so `hong_counterexample` is kept as the representative instance. Non-duplicate conveyor examples are retained.

## 7. Results

### 7.1 Known-model performance

{known_top_table}

The table shows that policy rankings are instance-dependent. This supports the main motivation for a cross-instance benchmark.

### 7.2 Aggregate policy recommendation

{recommendation_table}

If one known-model policy must be selected across all tested instance structures, WhittleIndexStrategy is the natural default candidate when Whittle indices are available: it is usually competitive in reward, cheap online, and robust across several stress-test instances. LPupdateStrategy can be preferable when reward is the only criterion and computation is less constrained, while LPPriorityStrategy is a cheap alternative that is competitive on many but not all instances.

### 7.3 Convergence rate

{beta_table}

Higher beta indicates faster empirical convergence to the LP upper bound. NaN values indicate insufficient positive-gap points for a reliable fit.

### 7.4 Computation cost

{cost_table}

Computation cost materially changes the practical ranking. Some methods have low online cost but high preprocessing cost; others are cheap to initialize but expensive during simulation.

### 7.5 Unknown-model online learning

{unknown_table}

Unknown-model policies should be compared against oracle curves but interpreted separately, because they begin without access to P,R.

### 7.6 Heterogeneous arms

{hetero_table}

The heterogeneous results test whether algorithms remain useful when arms differ by type. This section should be expanded after running the full heterogeneous suite.

## 8. Reproducibility

Main scripts:

- `run_instance_matrix_benchmark.py`: known-model homogeneous benchmark.
- `run_heterogeneous_benchmark.py`: known-model heterogeneous benchmark.
- `run_unknown_model_benchmark.py`: unknown-model online benchmark.
- `run_computation_cost_suite.py`: computation-cost aggregation and optional full cost rerun.
- `export_instance_taxonomy.py`: instance taxonomy.
- `generate_benchmark_findings.py`: result tables and draft findings.
- `generate_paper_figures.py`: paper-level summary figures.
- `check_paper_readiness.py`: output completeness check.

All outputs are stored under `instance_matrix_outputs`, `heterogeneous_outputs`, `unknown_model_outputs`, `computation_cost_outputs`, `computation_cost_suite_outputs`, and `paper_summary_outputs`.

## 9. Limitations and Next Steps

The current benchmark is sufficient for a first paper draft, but several limitations should be stated clearly:

- Some application-inspired instances are synthetic rather than directly copied from a paper.
- Unknown-model learning baselines are preliminary and should be interpreted as baselines, not state-of-the-art learning algorithms.
- LP-Update in heterogeneous settings is computationally expensive and may need selective evaluation.
- Relative gap to the LP upper bound is most meaningful for long-run average reward; finite-horizon transients can create negative empirical gaps.
- Heterogeneous experiments currently focus on typed arms rather than arbitrary fully heterogeneous arms at large scale.
- A newer LP-Update method that avoids solving a fresh LP at every step has been mentioned by Bruno/Nicolas Gast/Chen Yan. It is not implemented in this version because the exact algorithmic specification is still needed. The benchmark is structured so that the new method can be added as `LPupdateStrategyV2` and compared directly against the current LP-Update.

## 10. Conclusion

This benchmark suite shows that RMAB policy performance is strongly instance-dependent and that reward comparisons alone are incomplete. A useful RMAB benchmark should report reward, convergence rate, model-knowledge assumptions, arm heterogeneity, and computation cost together. The suite developed here provides a reusable foundation for such comparisons and can be extended with additional literature instances and learning-based policies.
"""

    draft_path = output_dir / "paper_draft_rmab_benchmark.md"
    draft_path.write_text(draft, encoding="utf-8")
    print(f"Saved paper draft to:\n{draft_path}")
    return draft_path


if __name__ == "__main__":
    DRAFT = generate_paper_draft()
