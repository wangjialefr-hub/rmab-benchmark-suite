"""
Run a small benchmark in which different arms have different P_i and R_i.

This experiment is independent from run_instance_matrix_benchmark.py and does
not use, delete, or overwrite the existing homogeneous simulation cache.
"""

import importlib
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and imports
# ============================================================

# Explicit path keeps the script directly runnable from Jupyter.
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import bandit_lp
import heterogeneous_rmab as heterogeneous_rmab_module

importlib.reload(heterogeneous_rmab_module)

from heterogeneous_rmab import (
    build_heterogeneous_maintenance,
    build_heterogeneous_wireless,
    make_heterogeneous_policy,
    simulate_heterogeneous,
)
from benchmark_metadata import experiment_axes, positive_integer, write_experiment_settings


OUTPUT_DIR = CURRENT_DIR / "heterogeneous_outputs" / "corrected_v2"


# ============================================================
# 2. Experiment settings
# ============================================================

N_VALUES = [20, 50, 100, 200, 500]
NUM_MONTE_CARLO = 5
MC_SEED = 123
SIMULATION_HORIZON = 200

# Five types means arms are heterogeneous but share five recurring models.
# Change to None to give every arm its own P_i and R_i. Fully heterogeneous
# Whittle is much slower because it computes a separate index table per arm.
NUM_ARM_TYPES = 5

POLICY_NAMES = [
    "HeterogeneousWhittle",
    "HeterogeneousLPPriority",
    "HeterogeneousFTVA",
    "HeterogeneousMyopic",
    "RandomActivation",
    "RoundRobin",
    # "HeterogeneousLPUpdate",  # slow: solves an LP for each new population state.
]

INSTANCE_BUILDERS = {
    "heterogeneous_maintenance": build_heterogeneous_maintenance,
    "heterogeneous_wireless": build_heterogeneous_wireless,
}


# ============================================================
# 3. Plotting
# ============================================================

def safe_filename(name):
    return re.sub(r'[ /\\:]', "_", name)


def plot_metric(summary_df, instance_name, metric, ylabel, suffix, output_dir=OUTPUT_DIR):
    """Plot one line per policy; confidence intervals are intentionally omitted."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    instance_df = summary_df[summary_df["instance"] == instance_name]

    for policy_name, group in instance_df.groupby("policy", sort=True):
        group = group.sort_values("N")
        ax.plot(
            group["N"],
            group[metric],
            marker="o",
            label=policy_name,
        )

    ax.set_xlabel("Number of arms (N)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if "seconds" in metric:
        ax.set_yscale("log")
    plotted_n = sorted(instance_df["N"].unique())
    ax.set_xticks(plotted_n)
    ax.set_xticklabels([str(n) for n in plotted_n])
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(
        Path(output_dir) / f"{safe_filename(instance_name)}_{suffix}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


# ============================================================
# 4. Benchmark
# ============================================================

def run_heterogeneous_benchmark(
    n_values=N_VALUES,
    num_monte_carlo=NUM_MONTE_CARLO,
    simulation_horizon=SIMULATION_HORIZON,
    num_arm_types=NUM_ARM_TYPES,
    policy_names=POLICY_NAMES,
    instance_builders=INSTANCE_BUILDERS,
    output_dir=OUTPUT_DIR,
    make_plots=True,
):
    """
    Run instance -> N -> policy -> seed using explicit per-arm models.

    setup_seconds measures policy preprocessing, especially per-model Whittle
    index computation. simulation_seconds measures online decisions plus arm
    transitions. No cache is used.
    """
    n_values, policy_names = experiment_axes(
        n_values, policy_names, num_monte_carlo=num_monte_carlo,
        simulation_horizon=simulation_horizon,
    )
    if not instance_builders:
        raise ValueError("Select at least one instance builder.")
    if num_arm_types is not None:
        num_arm_types = positive_integer(num_arm_types, "num_arm_types")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_settings(
        output_dir, source_files=["run_heterogeneous_benchmark.py", "benchmark_metadata.py",
                                  "heterogeneous_rmab.py", "bandit_lp.py", "strategies.py",
                                  "simulation_utils.py", "known_model_extra_instances.py", "rmab_instances.py"],
        instances=list(instance_builders), policies=policy_names, N=n_values,
        num_monte_carlo=num_monte_carlo, horizon=simulation_horizon, mc_seed=MC_SEED,
        num_arm_types=num_arm_types, simulation_cache=False,
        setup_repetitions=1,
        setup_scope="policy construction once per instance, N and policy; no prior grouped-model cache",
        simulation_scope="policy reset, initial states, decisions, rewards and transitions; excludes policy construction and file I/O",
        lp_update_cache="reused within one simulation only; cleared on reset",
    )

    setup_rows = []
    raw_rows = []

    for instance_name, builder in instance_builders.items():
        print(f"\nInstance family: {instance_name}")

        for N in n_values:
            environment = builder(
                bandit_lp=bandit_lp,
                N=N,
                num_types=num_arm_types,
            )
            distinct_models = len({arm.hashname() for arm in environment.arms})
            print(
                f"  N={N}, budget={environment.budget}, "
                f"distinct arm models={distinct_models}"
            )

            for policy_name in policy_names:
                try:
                    # Count grouping work for every policy that needs it.
                    if hasattr(environment, "_model_groups_cache"):
                        del environment._model_groups_cache
                    setup_start = time.perf_counter()
                    policy = make_heterogeneous_policy(
                        policy_name,
                        environment,
                    )
                    setup_seconds = time.perf_counter() - setup_start
                    setup_status = "ok"
                    setup_error = ""
                except Exception as exc:
                    policy = None
                    setup_seconds = np.nan
                    setup_status = "failed"
                    setup_error = f"{type(exc).__name__}: {exc}"

                setup_rows.append(
                    {
                        "instance": instance_name,
                        "environment_name": environment.name,
                        "policy": policy_name,
                        "N": N,
                        "S_min": min(arm.S for arm in environment.arms),
                        "S_max": max(arm.S for arm in environment.arms),
                        "alpha": environment.alpha,
                        "budget": environment.budget,
                        "num_distinct_models": distinct_models,
                        "setup_seconds": setup_seconds,
                        "status": setup_status,
                        "error": setup_error,
                    }
                )

                if policy is None:
                    print(f"    {policy_name}: setup failed: {setup_error}")
                    for replication in range(num_monte_carlo):
                        raw_rows.append(dict(
                            instance=instance_name, environment_name=environment.name,
                            policy=policy_name, N=N, alpha=environment.alpha, budget=environment.budget,
                            num_distinct_models=distinct_models, replication=replication,
                            seed=MC_SEED + replication, horizon=simulation_horizon,
                            mean_reward=np.nan, simulation_seconds=np.nan, seconds_per_step=np.nan,
                            status="failed", error=f"Policy setup failed: {setup_error}",
                        ))
                    pd.DataFrame(setup_rows).to_csv(output_dir / "heterogeneous_setup_times.csv", index=False)
                    pd.DataFrame(raw_rows).to_csv(output_dir / "heterogeneous_raw_results.csv", index=False)
                    continue

                failures = 0
                for replication in range(num_monte_carlo):
                    seed = MC_SEED + replication
                    try:
                        simulation_start = time.perf_counter()
                        result = simulate_heterogeneous(
                            environment=environment,
                            policy=policy,
                            horizon=simulation_horizon,
                            seed=seed,
                        )
                        simulation_seconds = (
                            time.perf_counter() - simulation_start
                        )

                        raw_rows.append(
                            {
                                "instance": instance_name,
                                "environment_name": environment.name,
                                "policy": policy_name,
                                "N": N,
                                "alpha": environment.alpha,
                                "budget": environment.budget,
                                "num_distinct_models": distinct_models,
                                "replication": replication,
                                "seed": seed,
                                "horizon": simulation_horizon,
                                "mean_reward": result["mean_reward"],
                                "simulation_seconds": simulation_seconds,
                                "seconds_per_step": (
                                    simulation_seconds / simulation_horizon
                                ),
                                "status": "ok",
                                "error": "",
                            }
                        )
                    except Exception as exc:
                        failures += 1
                        raw_rows.append(
                            {
                                "instance": instance_name,
                                "environment_name": environment.name,
                                "policy": policy_name,
                                "N": N,
                                "alpha": environment.alpha,
                                "budget": environment.budget,
                                "num_distinct_models": distinct_models,
                                "replication": replication,
                                "seed": seed,
                                "horizon": simulation_horizon,
                                "mean_reward": np.nan,
                                "simulation_seconds": np.nan,
                                "seconds_per_step": np.nan,
                                "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

                status_text = "ok" if failures == 0 else f"{failures} failed"
                print(
                    f"    {policy_name}: {status_text}, "
                    f"setup={setup_seconds:.4f}s"
                )
                pd.DataFrame(setup_rows).to_csv(output_dir / "heterogeneous_setup_times.csv", index=False)
                pd.DataFrame(raw_rows).to_csv(output_dir / "heterogeneous_raw_results.csv", index=False)

    setup_df = pd.DataFrame(setup_rows)
    raw_df = pd.DataFrame(raw_rows)

    successful = raw_df[raw_df["status"] == "ok"]
    summary_df = (
        successful.groupby(["instance", "policy", "N"], as_index=False)
        .agg(
            environment_name=("environment_name", "first"),
            alpha=("alpha", "first"),
            budget=("budget", "first"),
            num_distinct_models=("num_distinct_models", "first"),
            mean_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", "std"),
            mean_simulation_seconds=("simulation_seconds", "mean"),
            mean_seconds_per_step=("seconds_per_step", "mean"),
            num_runs=("mean_reward", "size"),
        )
        .merge(
            setup_df[setup_df["status"] == "ok"][
                ["instance", "policy", "N", "setup_seconds"]
            ],
            on=["instance", "policy", "N"],
            how="left",
        )
        .sort_values(["instance", "N", "policy"])
    )
    summary_df["estimated_cold_total_seconds"] = (
        summary_df["setup_seconds"]
        + summary_df["mean_simulation_seconds"]
    )

    setup_df.to_csv(
        output_dir / "heterogeneous_setup_times.csv",
        index=False,
    )
    raw_df.to_csv(
        output_dir / "heterogeneous_raw_results.csv",
        index=False,
    )
    summary_df.to_csv(
        output_dir / "heterogeneous_summary.csv",
        index=False,
    )

    if make_plots and not summary_df.empty:
        for instance_name in summary_df["instance"].unique():
            plot_metric(
                summary_df,
                instance_name,
                "mean_reward",
                "Average reward per arm",
                "reward_vs_N",
                output_dir=output_dir,
            )
            plot_metric(
                summary_df,
                instance_name,
                "setup_seconds",
                "Policy setup time (seconds)",
                "setup_time_vs_N",
                output_dir=output_dir,
            )
            plot_metric(
                summary_df,
                instance_name,
                "mean_simulation_seconds",
                "Simulation time (seconds)",
                "simulation_time_vs_N",
                output_dir=output_dir,
            )

    print(f"\nSaved heterogeneous outputs to:\n{output_dir}")
    return {
        "setup": setup_df,
        "raw": raw_df,
        "summary": summary_df,
    }


if __name__ == "__main__":
    RESULTS = run_heterogeneous_benchmark()
