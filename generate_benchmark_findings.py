"""
Generate report-ready findings from benchmark CSV outputs.

Run this after the benchmark scripts. It does not run simulations; it only reads
existing CSV files and writes compact tables plus a Markdown conclusion draft.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from paper_config import (
    DUPLICATE_INSTANCE_GROUPS,
    PAPER_EXCLUDED_INSTANCES,
    filter_paper_instances,
)


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "paper_summary_outputs"


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def top_known_reward(summary_final):
    if summary_final.empty:
        return pd.DataFrame()
    return (
        summary_final.sort_values(
            ["instance", "mean_reward"],
            ascending=[True, False],
        )
        .groupby("instance", as_index=False)
        .head(3)
        [
            [
                "instance",
                "policy",
                "N",
                "mean_reward",
                "mean_relative_gap",
                "alpha",
            ]
        ]
    )


def top_beta(beta_df):
    if beta_df.empty:
        return pd.DataFrame()
    valid = beta_df[np.isfinite(beta_df["convergence_beta"])]
    return (
        valid.sort_values(
            ["instance", "convergence_beta"],
            ascending=[True, False],
        )
        .groupby("instance", as_index=False)
        .head(3)
    )


def summarize_known_cost(cost_df):
    if cost_df.empty:
        return pd.DataFrame()
    return (
        cost_df.groupby("policy", as_index=False)
        .agg(
            mean_setup_seconds=("mean_setup_seconds", "mean"),
            mean_simulation_seconds=("mean_simulation_seconds", "mean"),
            mean_total_seconds=("estimated_cold_total_seconds", "mean"),
            max_total_seconds=("estimated_cold_total_seconds", "max"),
            rows=("estimated_cold_total_seconds", "size"),
        )
        .sort_values("mean_total_seconds", ascending=False)
    )


def summarize_cost_suite(cost_suite):
    if cost_suite.empty:
        return pd.DataFrame()
    return (
        cost_suite.groupby(["experiment_family", "policy"], as_index=False)
        .agg(
            mean_setup_seconds=("setup_seconds", "mean"),
            mean_online_seconds=("online_seconds", "mean"),
            mean_total_seconds=("total_seconds", "mean"),
            max_total_seconds=("total_seconds", "max"),
            rows=("total_seconds", "size"),
        )
        .sort_values(
            ["experiment_family", "mean_total_seconds"],
            ascending=[True, False],
        )
    )


def summarize_unknown(unknown_summary):
    if unknown_summary.empty:
        return pd.DataFrame()
    max_n = unknown_summary["N"].max()
    valid = unknown_summary[unknown_summary["N"] == max_n]
    return (
        valid.sort_values(
            ["instance", "mean_tail_reward"],
            ascending=[True, False],
        )
        .groupby("instance", as_index=False)
        .head(4)
        [
            [
                "instance",
                "policy",
                "policy_type",
                "N",
                "mean_reward",
                "mean_tail_reward",
                "mean_runtime_seconds",
            ]
        ]
    )


def summarize_heterogeneous(hetero_summary):
    if hetero_summary.empty:
        return pd.DataFrame()
    max_n = hetero_summary["N"].max()
    valid = hetero_summary[hetero_summary["N"] == max_n]
    return (
        valid.sort_values(
            ["instance", "mean_reward"],
            ascending=[True, False],
        )
        .groupby("instance", as_index=False)
        .head(4)
        [
            [
                "instance",
                "policy",
                "N",
                "mean_reward",
                "setup_seconds",
                "mean_simulation_seconds",
            ]
        ]
    )


def aggregate_known_model_recommendation(summary_final, beta_df, cost_known):
    """
    Build a paper-level recommendation table for known-model homogeneous policies.

    Lower score is better. The score combines:

    - reward_rank: average rank by final-N reward across non-duplicate instances;
    - gap_rank: average rank by clipped relative gap;
    - cost_rank: rank by total computation cost;
    - beta_rank: average rank by convergence beta.

    This is intentionally simple and transparent, so it can be explained in a
    report without pretending to be a universal theorem.
    """
    summary_final = filter_paper_instances(summary_final)
    beta_df = filter_paper_instances(beta_df)

    if summary_final.empty:
        return pd.DataFrame()

    score_df = summary_final.copy()
    score_df["clipped_relative_gap"] = score_df["mean_relative_gap"].clip(lower=0.0)
    score_df["reward_rank"] = score_df.groupby("instance")["mean_reward"].rank(
        ascending=False,
        method="average",
    )
    score_df["gap_rank"] = score_df.groupby("instance")[
        "clipped_relative_gap"
    ].rank(
        ascending=True,
        method="average",
    )

    reward_summary = (
        score_df.groupby("policy", as_index=False)
        .agg(
            avg_reward_rank=("reward_rank", "mean"),
            avg_gap_rank=("gap_rank", "mean"),
            avg_final_reward=("mean_reward", "mean"),
            avg_clipped_relative_gap=("clipped_relative_gap", "mean"),
            num_instances=("instance", "nunique"),
        )
    )

    if not beta_df.empty and "convergence_beta" in beta_df.columns:
        beta_valid = beta_df[np.isfinite(beta_df["convergence_beta"])].copy()
        if not beta_valid.empty:
            beta_valid["beta_rank"] = beta_valid.groupby("instance")[
                "convergence_beta"
            ].rank(ascending=False, method="average")
            beta_summary = (
                beta_valid.groupby("policy", as_index=False)
                .agg(
                    avg_beta_rank=("beta_rank", "mean"),
                    avg_convergence_beta=("convergence_beta", "mean"),
                    beta_fit_instances=("instance", "nunique"),
                )
            )
        else:
            beta_summary = pd.DataFrame()
    else:
        beta_summary = pd.DataFrame()

    if not cost_known.empty and "mean_total_seconds" in cost_known.columns:
        cost_rank = cost_known.copy()
        cost_rank["cost_rank"] = cost_rank["mean_total_seconds"].rank(
            ascending=True,
            method="average",
        )
        cost_rank = cost_rank[
            ["policy", "cost_rank", "mean_total_seconds"]
        ]
    else:
        cost_rank = pd.DataFrame()

    recommendation = reward_summary
    if not beta_summary.empty:
        recommendation = recommendation.merge(beta_summary, on="policy", how="left")
    if not cost_rank.empty:
        recommendation = recommendation.merge(cost_rank, on="policy", how="left")

    for col in ["avg_beta_rank", "cost_rank"]:
        if col not in recommendation.columns:
            recommendation[col] = np.nan
        recommendation[col] = recommendation[col].fillna(
            recommendation[col].max(skipna=True) + 1
        )

    recommendation["overall_score"] = (
        0.40 * recommendation["avg_reward_rank"]
        + 0.25 * recommendation["avg_gap_rank"]
        + 0.20 * recommendation["cost_rank"]
        + 0.15 * recommendation["avg_beta_rank"]
    )
    recommendation["recommendation_note"] = np.where(
        recommendation["policy"] == "WhittleIndexStrategy",
        "Strong default when the model is known and index computation is available.",
        np.where(
            recommendation["policy"] == "LPupdateStrategy",
            "Often high reward, but online computation can be expensive.",
            np.where(
                recommendation["policy"] == "LPPriorityStrategy",
                "Cheap and competitive, but less robust on some counterexamples.",
                "Useful baseline or instance-dependent alternative.",
            ),
        ),
    )

    return recommendation.sort_values("overall_score").reset_index(drop=True)


def dataframe_to_markdown(df, max_rows=20):
    if df.empty:
        return "_No data available yet._"
    return df.head(max_rows).to_markdown(index=False)


def write_findings_markdown(
    output_dir,
    known_top,
    beta_top,
    cost_known,
    cost_suite,
    unknown_top,
    hetero_top,
    recommendation,
):
    lines = [
        "# Benchmark Findings Draft",
        "",
        "This file is generated automatically from the current CSV outputs. "
        "Rerun the benchmark scripts and then rerun this file to refresh the conclusions.",
        "",
        "## 1. Known-model performance",
        "",
        "At the largest available N, the strongest policies are usually the "
        "structured planning/index policies rather than simple baselines. "
        "The exact ranking is instance-dependent, which supports the benchmark "
        "motivation: one RMAB policy is not uniformly best across all structures.",
        "",
        dataframe_to_markdown(known_top),
        "",
        "### Aggregate recommendation",
        "",
        "If only one known-model policy must be selected across the benchmark, "
        "the recommendation table combines reward rank, relative-gap rank, "
        "computation cost, and convergence beta. This is a practical benchmark "
        "recommendation rather than a theoretical dominance result.",
        "",
        dataframe_to_markdown(recommendation),
        "",
        "## 2. Convergence-rate behavior",
        "",
        "The convergence beta table estimates how quickly the relative gap to "
        "the LP upper bound decreases as N grows. Larger beta means faster "
        "empirical convergence. NaN means the fit was not meaningful, usually "
        "because the measured gap was non-positive or there were too few valid points.",
        "",
        dataframe_to_markdown(beta_top),
        "",
        "## 3. Computation cost",
        "",
        "Computation cost should be reported separately from reward. LP-Update "
        "and Q-Whittle-style methods can have substantially higher setup or "
        "online costs, while priority/index rules are often cheaper online.",
        "",
        "### Existing known-model cost output",
        "",
        dataframe_to_markdown(cost_known),
        "",
        "### Combined cost-suite output",
        "",
        dataframe_to_markdown(cost_suite),
        "",
        "## 4. Unknown-model online learning",
        "",
        "Unknown-model policies should be compared against known-model oracle "
        "curves, but interpreted separately. OnlinePlugInWhittle is the most "
        "RMAB-specific learning baseline: it estimates P,R and then computes "
        "Whittle indices from the learned model.",
        "",
        dataframe_to_markdown(unknown_top),
        "",
        "## 5. Heterogeneous arms",
        "",
        "The heterogeneous experiment studies arms with different P_i,R_i. "
        "Whittle, LP-Priority, FTVA, and LP-Update now have heterogeneous "
        "implementations, but LP-Update is expected to be expensive because it "
        "solves a finite-horizon LP repeatedly.",
        "",
        dataframe_to_markdown(hetero_top),
        "",
        "## 6. Current paper-level message",
        "",
        "- The suite now covers known-model homogeneous RMAB, known-model heterogeneous RMAB, and unknown-model online learning.",
        "- Instance structure matters: random, counterexample, maintenance, wireless, and deadline instances can produce different policy rankings.",
        "- Reward alone is not enough; convergence rate and computation cost change the practical ranking.",
        "- The next strongest addition would be a more theoretically grounded unknown-model RMAB learner, such as UCB/Thompson model learning plus Whittle or LP-based control.",
        "- Duplicate instance handling: conveyor_eg4action-gap-tb_S8 is excluded because it is identical to hong_counterexample; non-duplicate conveyor examples are retained.",
        "",
    ]
    (output_dir / "benchmark_findings_draft.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def generate_benchmark_findings(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_outputs = CURRENT_DIR / "instance_matrix_outputs"
    known_cost_outputs = CURRENT_DIR / "computation_cost_outputs"
    cost_suite_outputs = CURRENT_DIR / "computation_cost_suite_outputs"
    unknown_outputs = CURRENT_DIR / "unknown_model_outputs"
    hetero_outputs = CURRENT_DIR / "heterogeneous_outputs"

    summary_final = read_csv_if_exists(
        instance_outputs / "summary_final_N500.csv"
    )
    summary_final = filter_paper_instances(summary_final)
    beta_df = read_csv_if_exists(
        instance_outputs / "convergence_beta_by_instance.csv"
    )
    beta_df = filter_paper_instances(beta_df)
    known_cost_df = read_csv_if_exists(
        known_cost_outputs / "computation_cost_summary.csv"
    )
    known_cost_df = filter_paper_instances(known_cost_df)
    cost_suite_df = read_csv_if_exists(
        cost_suite_outputs / "combined_computation_cost_summary.csv"
    )
    cost_suite_df = filter_paper_instances(cost_suite_df)
    unknown_summary = read_csv_if_exists(
        unknown_outputs / "unknown_model_summary.csv"
    )
    hetero_summary = read_csv_if_exists(
        hetero_outputs / "heterogeneous_summary.csv"
    )

    known_top = top_known_reward(summary_final)
    beta_top_df = top_beta(beta_df)
    cost_known = summarize_known_cost(known_cost_df)
    cost_suite = summarize_cost_suite(cost_suite_df)
    unknown_top = summarize_unknown(unknown_summary)
    hetero_top = summarize_heterogeneous(hetero_summary)
    recommendation = aggregate_known_model_recommendation(
        summary_final,
        beta_df,
        cost_known,
    )

    known_top.to_csv(output_dir / "known_model_top_policies.csv", index=False)
    beta_top_df.to_csv(output_dir / "top_convergence_beta.csv", index=False)
    cost_known.to_csv(output_dir / "known_model_cost_ranking.csv", index=False)
    cost_suite.to_csv(output_dir / "combined_cost_ranking.csv", index=False)
    unknown_top.to_csv(output_dir / "unknown_model_top_policies.csv", index=False)
    hetero_top.to_csv(output_dir / "heterogeneous_top_policies.csv", index=False)
    recommendation.to_csv(
        output_dir / "aggregate_policy_recommendation.csv",
        index=False,
    )

    write_findings_markdown(
        output_dir=output_dir,
        known_top=known_top,
        beta_top=beta_top_df,
        cost_known=cost_known,
        cost_suite=cost_suite,
        unknown_top=unknown_top,
        hetero_top=hetero_top,
        recommendation=recommendation,
    )

    print(f"Saved benchmark findings to:\n{output_dir}")
    return {
        "known_top": known_top,
        "beta_top": beta_top_df,
        "cost_known": cost_known,
        "cost_suite": cost_suite,
        "unknown_top": unknown_top,
        "heterogeneous_top": hetero_top,
    }


if __name__ == "__main__":
    FINDINGS = generate_benchmark_findings()
