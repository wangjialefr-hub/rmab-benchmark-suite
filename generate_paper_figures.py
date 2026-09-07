"""
Generate paper-level summary figures from existing benchmark CSV files.

This script does not run simulations. It reads the outputs produced by the
benchmark scripts and creates aggregate figures suitable for a report draft.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_config import filter_paper_instances


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "paper_summary_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

POLICY_DISPLAY_NAMES = {
    "WhittleIndexStrategy": "Whittle",
    "LPPriorityStrategy": "LP-Priority",
    "LPupdateStrategy": "LP-Update",
    "QWhittleKnownModel": "Q-Whittle",
    "FTVA_Strategy": "FTVA",
    "LPRandomized": "LP-Randomized",
    "RandomPriority": "Random Priority",
    "RoundRobin": "Round Robin",
    "HeterogeneousWhittle": "Whittle",
    "HeterogeneousLPPriority": "LP-Priority",
    "HeterogeneousMyopic": "Myopic",
    "HeterogeneousFTVA": "FTVA",
    "RandomActivation": "Random Activation",
}


def display_policy_names(names):
    return [POLICY_DISPLAY_NAMES.get(name, name) for name in names]


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def configure_matplotlib():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def plot_heatmap(matrix, title, colorbar_label, filename, cmap="viridis", fmt=".2g"):
    if matrix.empty:
        return

    matrix = matrix.copy()
    values = matrix.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)

    fig_width = max(7.0, 0.48 * len(matrix.columns) + 2.5)
    fig_height = max(4.5, 0.38 * len(matrix.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(masked_values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(
        display_policy_names(matrix.columns),
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(
                    j,
                    i,
                    format(values[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if values[i, j] > np.nanmean(values) else "black",
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def reward_rank_matrix(summary_df, N):
    """Return instance-by-policy reward ranks for one population size."""
    df = summary_df[summary_df["N"] == N].copy()
    df["reward_rank"] = df.groupby("instance")["mean_reward"].rank(
        method="min",
        ascending=False,
    )
    return df.pivot(index="instance", columns="policy", values="reward_rank")


def policy_order_from_largest_n(summary_df):
    """Order policies by mean reward rank at the largest available N."""
    max_n = summary_df["N"].max()
    matrix = reward_rank_matrix(summary_df, max_n)
    return matrix.mean(axis=0, skipna=True).sort_values().index.tolist()


def plot_known_model_rank_heatmap(summary_final):
    if summary_final.empty:
        return

    df = summary_final.copy()
    df["reward_rank"] = df.groupby("instance")["mean_reward"].rank(
        method="min",
        ascending=False,
    )
    matrix = df.pivot(
        index="instance",
        columns="policy",
        values="reward_rank",
    )
    policy_order = matrix.mean(axis=0, skipna=True).sort_values().index
    matrix = matrix.reindex(columns=policy_order)
    plot_heatmap(
        matrix=matrix,
        title="Known-model reward rank at largest N",
        colorbar_label="Rank, lower is better",
        filename="known_model_reward_rank_heatmap.png",
        cmap="viridis_r",
        fmt=".0f",
    )


def plot_known_model_rank_overview(summary_df, detail_n_values=(20, 100, 200)):
    """Plot the largest-N rank heatmap and three smaller N cross-sections."""
    if summary_df.empty:
        return

    max_n = int(summary_df["N"].max())
    policy_order = policy_order_from_largest_n(summary_df)
    largest_matrix = reward_rank_matrix(summary_df, max_n).reindex(
        columns=policy_order
    )

    fig = plt.figure(figsize=(13.5, 10.5))
    grid = fig.add_gridspec(2, 3, height_ratios=(2.2, 1.0), hspace=0.48, wspace=0.28)
    main_ax = fig.add_subplot(grid[0, :])

    values = largest_matrix.to_numpy(dtype=float)
    image = main_ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        cmap="viridis_r",
        vmin=1,
        vmax=max(2, len(policy_order)),
    )
    main_ax.set_title(
        f"Reward rank at N={max_n}; policies ordered by mean rank"
    )
    main_ax.set_xticks(np.arange(len(policy_order)))
    main_ax.set_xticklabels(
        display_policy_names(policy_order),
        rotation=35,
        ha="right",
    )
    main_ax.set_yticks(np.arange(len(largest_matrix.index)))
    main_ax.set_yticklabels(largest_matrix.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                main_ax.text(
                    column,
                    row,
                    f"{values[row, column]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    colorbar = fig.colorbar(image, ax=main_ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Reward rank (lower is better)")

    available_n = set(summary_df["N"].astype(int))
    selected_n = [int(N) for N in detail_n_values if int(N) in available_n]
    for ax, N in zip(
        [fig.add_subplot(grid[1, i]) for i in range(3)],
        selected_n,
    ):
        matrix = reward_rank_matrix(summary_df, N).reindex(columns=policy_order)
        mean_rank = matrix.mean(axis=0, skipna=True)
        ax.plot(
            np.arange(len(policy_order)),
            mean_rank.to_numpy(),
            marker="o",
            linewidth=1.5,
        )
        ax.set_title(f"Mean reward rank at N={N}")
        ax.set_xticks(np.arange(len(policy_order)))
        ax.set_xticklabels(
            display_policy_names(policy_order),
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("Mean rank")
        ax.set_ylim(len(policy_order) + 0.5, 0.5)
        ax.grid(axis="y", alpha=0.25)

    fig.savefig(
        FIGURE_DIR / "known_model_reward_rank_overview.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    pd.DataFrame(
        {
            "display_order": np.arange(1, len(policy_order) + 1),
            "policy": policy_order,
            "mean_reward_rank_at_largest_N": largest_matrix.mean(
                axis=0,
                skipna=True,
            ).reindex(policy_order).to_numpy(),
        }
    ).to_csv(OUTPUT_DIR / "known_model_policy_display_order.csv", index=False)


def plot_known_model_gap_heatmap(summary_final, policy_order=None):
    if summary_final.empty:
        return

    df = summary_final.copy()
    df["display_gap"] = df["mean_relative_gap"].clip(lower=0.0)
    matrix = df.pivot(
        index="instance",
        columns="policy",
        values="display_gap",
    )
    if policy_order is not None:
        matrix = matrix.reindex(columns=policy_order)
    plot_heatmap(
        matrix=matrix,
        title="Known-model clipped relative gap at largest N",
        colorbar_label="max(relative gap, 0)",
        filename="known_model_relative_gap_heatmap.png",
        cmap="magma_r",
        fmt=".2g",
    )


def plot_beta_heatmap(beta_df, policy_order=None):
    if beta_df.empty:
        return

    matrix = beta_df.pivot(
        index="instance",
        columns="policy",
        values="convergence_beta",
    )
    if policy_order is not None:
        matrix = matrix.reindex(columns=policy_order)
    plot_heatmap(
        matrix=matrix,
        title="Estimated convergence rate beta",
        colorbar_label="beta, higher is faster",
        filename="convergence_beta_heatmap.png",
        cmap="viridis",
        fmt=".2g",
    )


def plot_cost_bar(cost_ranking):
    if cost_ranking.empty:
        return

    if "experiment_family" in cost_ranking.columns:
        for family, family_df in cost_ranking.groupby("experiment_family"):
            _plot_single_cost_bar(
                family_df,
                title=f"Computation cost: {family}",
                filename=f"cost_ranking_{family}.png",
            )
    else:
        _plot_single_cost_bar(
            cost_ranking,
            title="Known-model computation cost",
            filename="known_model_cost_ranking.png",
        )


def _plot_single_cost_bar(df, title, filename):
    df = df.copy()
    total_col = (
        "mean_total_seconds"
        if "mean_total_seconds" in df.columns
        else "estimated_cold_total_seconds"
    )
    df = df.sort_values(total_col, ascending=True)

    fig_height = max(4.5, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    ax.barh(df["policy"], df[total_col])
    ax.set_xscale("log")
    ax.set_xlabel("Mean total time (seconds, log scale)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_unknown_tail_reward(unknown_summary):
    if unknown_summary.empty:
        return

    max_n = unknown_summary["N"].max()
    df = unknown_summary[unknown_summary["N"] == max_n]
    matrix = df.pivot(
        index="instance",
        columns="policy",
        values="mean_tail_reward",
    )
    plot_heatmap(
        matrix=matrix,
        title=f"Unknown-model tail reward at N={max_n}",
        colorbar_label="Tail average reward",
        filename="unknown_model_tail_reward_heatmap.png",
        cmap="viridis",
        fmt=".3g",
    )


def plot_heterogeneous_reward(heterogeneous_summary):
    if heterogeneous_summary.empty:
        return

    max_n = heterogeneous_summary["N"].max()
    df = heterogeneous_summary[heterogeneous_summary["N"] == max_n]
    matrix = df.pivot(
        index="instance",
        columns="policy",
        values="mean_reward",
    )
    plot_heatmap(
        matrix=matrix,
        title=f"Heterogeneous reward at N={max_n}",
        colorbar_label="Average reward",
        filename="heterogeneous_reward_heatmap.png",
        cmap="viridis",
        fmt=".3g",
    )


def normalized_score(series, higher_is_better):
    """Min-max normalize one criterion so that 1 always means better."""
    series = pd.to_numeric(series, errors="coerce")
    finite = series[np.isfinite(series)]
    if finite.empty:
        return pd.Series(np.nan, index=series.index)
    lower = finite.min()
    upper = finite.max()
    if np.isclose(lower, upper):
        return pd.Series(1.0, index=series.index)
    score = (series - lower) / (upper - lower)
    return score if higher_is_better else 1.0 - score


def plot_heterogeneous_normalized_scores(heterogeneous_summary):
    """Show normalized heterogeneous criteria in one readable heatmap."""
    if heterogeneous_summary.empty:
        return

    max_n = int(heterogeneous_summary["N"].max())
    df = heterogeneous_summary[heterogeneous_summary["N"] == max_n].copy()
    criteria = [
        ("mean_reward", True, "Reward score"),
        ("mean_simulation_seconds", False, "Online-efficiency score"),
        ("estimated_cold_total_seconds", False, "Total-efficiency score"),
    ]

    score_columns = []
    for metric, higher_is_better, _ in criteria:
        score_name = f"normalized_{metric}"
        df[score_name] = df.groupby("instance")[metric].transform(
            lambda values: normalized_score(values, higher_is_better)
        )
        score_columns.append(score_name)

    reward_matrix = df.pivot(
        index="instance",
        columns="policy",
        values=score_columns[0],
    )
    policy_order = (
        reward_matrix.mean(axis=0, skipna=True).sort_values(ascending=False).index
    )

    matrices = []
    for (_, _, title), score_name in zip(criteria, score_columns):
        matrix = df.pivot(
            index="instance",
            columns="policy",
            values=score_name,
        ).reindex(columns=policy_order)
        matrix.index = [f"{instance} | {title}" for instance in matrix.index]
        matrices.append(matrix)

    combined_matrix = pd.concat(matrices)
    plot_heatmap(
        matrix=combined_matrix,
        title=(
            f"Heterogeneous performance at N={max_n}; "
            "1 is best within each instance"
        ),
        colorbar_label="Normalized score (higher is better)",
        filename="heterogeneous_normalized_scores.png",
        cmap="viridis",
        fmt=".2f",
    )

    df[
        ["instance", "policy", "N"] + score_columns
    ].to_csv(OUTPUT_DIR / "heterogeneous_normalized_scores.csv", index=False)


def generate_paper_figures(output_dir=OUTPUT_DIR, figure_dir=None):
    global OUTPUT_DIR, FIGURE_DIR
    OUTPUT_DIR = Path(output_dir)
    FIGURE_DIR = (
        Path(figure_dir)
        if figure_dir is not None
        else OUTPUT_DIR / "figures"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    summary_final = read_csv_if_exists(
        CURRENT_DIR / "instance_matrix_outputs" / "summary_final_N500.csv"
    )
    summary_final = filter_paper_instances(summary_final)
    known_summary = read_csv_if_exists(
        CURRENT_DIR / "instance_matrix_outputs" / "summary_instance_matrix.csv"
    )
    known_summary = filter_paper_instances(known_summary)
    beta_df = read_csv_if_exists(
        CURRENT_DIR / "instance_matrix_outputs" / "convergence_beta_by_instance.csv"
    )
    beta_df = filter_paper_instances(beta_df)
    combined_cost = read_csv_if_exists(
        CURRENT_DIR
        / "computation_cost_suite_outputs"
        / "combined_computation_cost_ranking.csv"
    )
    known_cost = read_csv_if_exists(
        CURRENT_DIR / "paper_summary_outputs" / "known_model_cost_ranking.csv"
    )
    unknown_summary = read_csv_if_exists(
        CURRENT_DIR / "unknown_model_outputs" / "unknown_model_summary.csv"
    )
    heterogeneous_summary = read_csv_if_exists(
        CURRENT_DIR / "heterogeneous_outputs" / "heterogeneous_summary.csv"
    )

    policy_order = policy_order_from_largest_n(known_summary)

    plot_known_model_rank_heatmap(summary_final)
    plot_known_model_rank_overview(known_summary)
    plot_known_model_gap_heatmap(summary_final, policy_order)
    plot_beta_heatmap(beta_df, policy_order)
    plot_cost_bar(combined_cost if not combined_cost.empty else known_cost)
    plot_unknown_tail_reward(unknown_summary)
    plot_heterogeneous_reward(heterogeneous_summary)

    figure_index = pd.DataFrame(
        [
            {
                "figure": file.name,
                "path": str(file),
            }
            for file in sorted(FIGURE_DIR.glob("*.png"))
        ]
    )
    figure_index.to_csv(OUTPUT_DIR / "paper_figure_index.csv", index=False)

    print(f"Saved paper figures to:\n{FIGURE_DIR}")
    return figure_index


if __name__ == "__main__":
    FIGURES = generate_paper_figures(
        figure_dir=OUTPUT_DIR / "figures"
    )
