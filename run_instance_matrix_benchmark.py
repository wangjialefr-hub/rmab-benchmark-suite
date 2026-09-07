r"""
Run the known-model homogeneous RMAB benchmark.

The experiment loop is:

    instance -> N -> policy -> Monte Carlo seed

Outputs are saved to ``instance_matrix_outputs`` next to this script.
"""

import importlib
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and imports
# ============================================================

# Explicit path keeps this file runnable from Jupyter with:
# From Jupyter, run: %run "run_instance_matrix_benchmark.py"
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import bandit_lp
import known_model_extra_instances as instance_module
import make_policy as make_policy_module
import rmab_instances as base_instance_module
import simulation_cache as cache_module

importlib.reload(instance_module)
importlib.reload(make_policy_module)
importlib.reload(base_instance_module)
importlib.reload(cache_module)

from known_model_extra_instances import (
    build_extended_known_model_instance_library,
    extra_instance_metadata_dataframe,
)
from make_policy import POLICY_NAMES as DEFAULT_POLICY_NAMES, make_policy
from paper_config import KNOWN_MODEL_BENCHMARK_INSTANCES
from simulation_cache import cached_lp_upper_bound, cached_simulate_policy


# ============================================================
# 2. Experiment settings
# ============================================================

OUTPUT_DIR = CURRENT_DIR / "instance_matrix_outputs"
CACHE_DIR = CURRENT_DIR / "rmab_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VALUES = [20, 50, 100, 200, 500]
NUM_MONTE_CARLO = 20
MC_SEED = 123
SIMULATION_HORIZON = 200
LP_UPDATE_HORIZON = 20

POLICY_NAMES = list(DEFAULT_POLICY_NAMES)

INSTANCE_NAMES = list(KNOWN_MODEL_BENCHMARK_INSTANCES)


# ============================================================
# 3. Helpers
# ============================================================

plt.rcParams.update(
    {
        "figure.figsize": (7.0, 4.6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
    }
)


def safe_filename(name):
    return re.sub(r"[ /\\:]", "_", name)


def relative_gap(lp_upper_bound, reward):
    if abs(lp_upper_bound) <= 1e-12:
        return np.nan
    return (lp_upper_bound - reward) / abs(lp_upper_bound)


def calc_beta_one_group(group):
    """Fit the descriptive power law Gap(N) = C * N**(-beta).

    This is an empirical finite-range slope, not a proof of an asymptotic
    convergence rate. At least three positive finite gaps are required so that
    the reported slope is not determined by only two points.
    """
    valid = group[
        (group["mean_relative_gap"] > 1e-12)
        & np.isfinite(group["mean_relative_gap"])
        & np.isfinite(group["N"])
    ].sort_values("N")
    if len(valid) < 3:
        return pd.Series(
            {
                "convergence_beta": np.nan,
                "r_squared": np.nan,
                "num_fit_points": len(valid),
            }
        )

    log_n = np.log(valid["N"])
    log_gap = np.log(valid["mean_relative_gap"])
    slope, intercept = np.polyfit(log_n, log_gap, deg=1)

    fitted = slope * log_n + intercept
    ss_tot = np.sum((log_gap - np.mean(log_gap)) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1 - np.sum((log_gap - fitted) ** 2) / ss_tot

    return pd.Series(
        {
            "convergence_beta": -slope,
            "r_squared": r_squared,
            "num_fit_points": len(valid),
        }
    )


def estimate_beta(summary_df):
    grouped = summary_df.groupby(["instance", "policy"])
    try:
        beta_df = grouped.apply(calc_beta_one_group, include_groups=False)
    except TypeError:
        beta_df = grouped.apply(calc_beta_one_group)
    return (
        beta_df.reset_index()
        .sort_values(["instance", "convergence_beta"], ascending=[True, False])
    )


def plot_instance_metric(instance, summary_df, metric, ylabel, log_y, suffix):
    fig, ax = plt.subplots()
    instance_df = summary_df[summary_df["instance"] == instance]

    for policy, policy_df in instance_df.groupby("policy", sort=True):
        policy_df = policy_df.sort_values("N")
        y_values = policy_df[metric].to_numpy()
        if log_y:
            y_values = np.maximum(y_values, 1e-12)
        ax.plot(policy_df["N"], y_values, marker="o", label=policy)

    ax.set_xlabel("Number of arms (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(instance)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{safe_filename(instance)}_{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def make_base_row(spec, instance_name, N, policy, replication, seed, lp_upper_bound):
    bandit = spec.bandit
    return {
        "instance": instance_name,
        "source": spec.source,
        "S": bandit.S,
        "A": bandit.A,
        "alpha": spec.default_alpha,
        "N": N,
        "policy": policy,
        "replication": replication,
        "seed": seed,
        "lp_upper_bound": lp_upper_bound,
    }


# ============================================================
# 4. Build instance library
# ============================================================

instance_library = build_extended_known_model_instance_library(bandit_lp)
missing_instances = [name for name in INSTANCE_NAMES if name not in instance_library]
if missing_instances:
    raise ValueError(f"Unknown instances: {missing_instances}")

instance_metadata_df = pd.DataFrame(extra_instance_metadata_dataframe(instance_library))
instance_metadata_df.to_csv(OUTPUT_DIR / "instance_library.csv", index=False)

print("Instance library:")
print(instance_metadata_df[instance_metadata_df["instance"].isin(INSTANCE_NAMES)])


# ============================================================
# 5. Run simulations
# ============================================================

raw_rows = []

for instance_name in INSTANCE_NAMES:
    spec = instance_library[instance_name]
    bandit = spec.bandit
    alpha = spec.default_alpha
    lp_upper_bound, lp_info = cached_lp_upper_bound(
        bandit=bandit,
        alpha=alpha,
        cache_dir=CACHE_DIR,
    )

    print(
        f"\n{'=' * 72}\n"
        f"Instance: {instance_name} | S={bandit.S} | A={bandit.A} | "
        f"alpha={alpha} | LP bound={lp_upper_bound:.6f} | "
        f"LP hit={lp_info['cache_hit']}"
    )

    for N in N_VALUES:
        print(f"\n  N = {N}")

        for policy_name in POLICY_NAMES:
            hits = 0
            failures = 0

            for replication in range(NUM_MONTE_CARLO):
                seed = MC_SEED + replication
                row = make_base_row(
                    spec,
                    instance_name,
                    N,
                    policy_name,
                    replication,
                    seed,
                    lp_upper_bound,
                )

                try:
                    result = cached_simulate_policy(
                        bandit=bandit,
                        policy_name=policy_name,
                        make_policy=make_policy,
                        initial_state=spec.initial_state,
                        N=N,
                        time_horizon=SIMULATION_HORIZON,
                        seed=seed,
                        alpha=alpha,
                        lp_update_horizon=LP_UPDATE_HORIZON,
                        cache_dir=CACHE_DIR,
                        return_info=True,
                    )
                    mean_reward = result[0]
                    cache_info = result[-1]
                    hits += int(cache_info["cache_hit"])

                    row.update(
                        {
                            "simulation_mean_reward": mean_reward,
                            "gap": lp_upper_bound - mean_reward,
                            "relative_gap": relative_gap(lp_upper_bound, mean_reward),
                            "runtime_seconds": cache_info["runtime_seconds"],
                            "cache_hit": cache_info["cache_hit"],
                            "cache_file": cache_info["cache_file"],
                            "status": "ok",
                            "error": "",
                        }
                    )
                except Exception as exc:
                    failures += 1
                    row.update(
                        {
                            "simulation_mean_reward": np.nan,
                            "gap": np.nan,
                            "relative_gap": np.nan,
                            "runtime_seconds": np.nan,
                            "cache_hit": False,
                            "cache_file": "",
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

                raw_rows.append(row)

            if failures:
                print(f"    {policy_name}: failed ({failures}/{NUM_MONTE_CARLO})")
            elif hits == NUM_MONTE_CARLO:
                print(f"    {policy_name}: cache hit (all)")
            else:
                misses = NUM_MONTE_CARLO - hits
                print(f"    {policy_name}: {hits}/{NUM_MONTE_CARLO} hits, {misses} misses")

raw_df = pd.DataFrame(raw_rows)
raw_df.to_csv(OUTPUT_DIR / "raw_instance_matrix_results.csv", index=False)


# ============================================================
# 6. Aggregate and export
# ============================================================

summary_df = (
    raw_df[raw_df["status"] == "ok"]
    .groupby(["instance", "policy", "N"], as_index=False)
    .agg(
        S=("S", "first"),
        A=("A", "first"),
        alpha=("alpha", "first"),
        lp_upper_bound=("lp_upper_bound", "first"),
        mean_reward=("simulation_mean_reward", "mean"),
        std_reward=("simulation_mean_reward", "std"),
        mean_gap=("gap", "mean"),
        std_gap=("gap", "std"),
        mean_relative_gap=("relative_gap", "mean"),
        std_relative_gap=("relative_gap", "std"),
        mean_runtime=("runtime_seconds", "mean"),
        num_runs=("simulation_mean_reward", "size"),
        cache_hits=("cache_hit", "sum"),
    )
    .sort_values(["instance", "N", "policy"])
)

beta_df = estimate_beta(summary_df)

summary_df.to_csv(OUTPUT_DIR / "summary_instance_matrix.csv", index=False)
beta_df.to_csv(OUTPUT_DIR / "convergence_beta_by_instance.csv", index=False)

max_N = max(N_VALUES)
final_N_df = summary_df[summary_df["N"] == max_N]
final_N_df.to_csv(OUTPUT_DIR / f"summary_final_N{max_N}.csv", index=False)
final_N_df.pivot(index="instance", columns="policy", values="mean_reward").to_csv(
    OUTPUT_DIR / f"pivot_reward_final_N{max_N}.csv"
)
final_N_df.pivot(index="instance", columns="policy", values="mean_relative_gap").to_csv(
    OUTPUT_DIR / f"pivot_relative_gap_final_N{max_N}.csv"
)

print(f"\nSaved outputs to:\n{OUTPUT_DIR}")
print(f"\nConvergence beta preview:\n{beta_df.head(30)}")


# ============================================================
# 7. Plots
# ============================================================

for instance_name in summary_df["instance"].unique():
    plot_instance_metric(
        instance_name,
        summary_df,
        metric="mean_reward",
        ylabel="Average reward",
        log_y=False,
        suffix="reward_vs_N",
    )
    plot_instance_metric(
        instance_name,
        summary_df,
        metric="mean_relative_gap",
        ylabel="Relative gap to LP upper bound",
        log_y=True,
        suffix="relative_gap_vs_N",
    )

print("\nSaved per-instance figures.")
