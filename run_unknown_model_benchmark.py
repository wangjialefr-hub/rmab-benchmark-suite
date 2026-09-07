"""
Benchmark unknown-P,R online-learning baselines.

This script is separate from the known-model benchmark. The hidden environment
uses the true P,R to generate samples, but the online policies update only from
observed states, actions, rewards, and next states.
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

# Explicit path keeps this directly runnable from Jupyter:
# From Jupyter, run: %run "run_unknown_model_benchmark.py"
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import bandit_lp
import known_model_extra_instances as instance_module
import unknown_model_learning as online_module

importlib.reload(instance_module)
importlib.reload(online_module)

from known_model_extra_instances import (
    build_extended_known_model_instance_library,
    extra_instance_metadata_dataframe,
)
from unknown_model_learning import (
    ORACLE_POLICY_NAMES,
    UNKNOWN_POLICY_NAMES,
    make_unknown_model_policy,
    simulate_unknown_model,
)


OUTPUT_DIR = CURRENT_DIR / "unknown_model_outputs"


# ============================================================
# 2. Experiment settings
# ============================================================

INSTANCE_NAMES = [
    "random_S10_seed123",
    "maintenance_S10_a40",
    "wireless_channel_S10_a40",
]

POLICY_NAMES = [
    "KnownWhittleOracle",
    "KnownLPPriorityOracle",
    "OnlinePlugInWhittle",
    "OnlineQLearningIndex",
    "OnlineUCBReward",
    "OnlineRewardGreedy",
    "OnlineRandom",
]

N_VALUES = [20, 50, 100, 200]
NUM_MONTE_CARLO = 5
MC_SEED = 123

# Online learning needs a longer horizon than known-model planning policies,
# because the policy starts without knowing P or R.
SIMULATION_HORIZON = 1000
CURVE_SAMPLE_EVERY = 10


# ============================================================
# 3. Helpers
# ============================================================

def safe_filename(name):
    return re.sub(r'[ /\\:]', "_", name)


def policy_type(policy_name):
    if policy_name in ORACLE_POLICY_NAMES:
        return "known_model_oracle"
    if policy_name in UNKNOWN_POLICY_NAMES:
        return "unknown_model_online"
    return "other"


def plot_metric(summary_df, instance_name, metric, ylabel, suffix):
    """Plot one line per policy; no confidence interval shading."""
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
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{safe_filename(instance_name)}_{suffix}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_learning_curve(curve_df, instance_name, N):
    """Plot average reward over time for one instance and one N."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    instance_df = curve_df[
        (curve_df["instance"] == instance_name)
        & (curve_df["N"] == N)
    ]

    for policy_name, group in instance_df.groupby("policy", sort=True):
        group = group.sort_values("time")
        ax.plot(
            group["time"],
            group["mean_step_reward"],
            label=policy_name,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Average reward per arm")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{safe_filename(instance_name)}_learning_curve_N{N}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


# ============================================================
# 4. Benchmark
# ============================================================

def run_unknown_model_benchmark(
    instance_names=INSTANCE_NAMES,
    policy_names=POLICY_NAMES,
    n_values=N_VALUES,
    num_monte_carlo=NUM_MONTE_CARLO,
    simulation_horizon=SIMULATION_HORIZON,
    curve_sample_every=CURVE_SAMPLE_EVERY,
    output_dir=OUTPUT_DIR,
    make_plots=True,
):
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    instance_library = build_extended_known_model_instance_library(
        bandit_lp,
        include_original=True,
    )
    missing = [name for name in instance_names if name not in instance_library]
    if missing:
        raise ValueError(f"Unknown instances: {missing}")

    metadata_df = pd.DataFrame(
        extra_instance_metadata_dataframe(instance_library)
    )
    metadata_df[metadata_df["instance"].isin(instance_names)].to_csv(
        OUTPUT_DIR / "unknown_model_instance_library.csv",
        index=False,
    )

    raw_rows = []
    curve_rows = []

    for instance_name in instance_names:
        spec = instance_library[instance_name]
        bandit = spec.bandit
        alpha = spec.default_alpha
        print(f"\nInstance: {instance_name}  S={bandit.S}, alpha={alpha}")

        for N in n_values:
            print(f"  N={N}")

            for policy_name in policy_names:
                failures = 0
                step_rewards = []

                for replication in range(num_monte_carlo):
                    seed = MC_SEED + replication
                    start = time.perf_counter()

                    try:
                        policy = make_unknown_model_policy(
                            policy_name,
                            bandit=bandit,
                            N=N,
                            alpha=alpha,
                        )
                        result = simulate_unknown_model(
                            bandit=bandit,
                            policy=policy,
                            initial_state=spec.initial_state,
                            N=N,
                            horizon=simulation_horizon,
                            seed=seed,
                        )
                        seconds = time.perf_counter() - start

                        raw_rows.append(
                            {
                                "instance": instance_name,
                                "policy": policy_name,
                                "policy_type": policy_type(policy_name),
                                "N": N,
                                "S": bandit.S,
                                "A": bandit.A,
                                "alpha": alpha,
                                "budget": int(alpha * N),
                                "replication": replication,
                                "seed": seed,
                                "horizon": simulation_horizon,
                                "mean_reward": result["mean_reward"],
                                "tail_mean_reward": result["tail_mean_reward"],
                                "runtime_seconds": seconds,
                                "status": "ok",
                                "error": "",
                            }
                        )
                        step_rewards.append(result["reward_history"])

                    except Exception as exc:
                        failures += 1
                        raw_rows.append(
                            {
                                "instance": instance_name,
                                "policy": policy_name,
                                "policy_type": policy_type(policy_name),
                                "N": N,
                                "S": bandit.S,
                                "A": bandit.A,
                                "alpha": alpha,
                                "budget": int(alpha * N),
                                "replication": replication,
                                "seed": seed,
                                "horizon": simulation_horizon,
                                "mean_reward": np.nan,
                                "tail_mean_reward": np.nan,
                                "runtime_seconds": np.nan,
                                "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

                if step_rewards:
                    mean_curve = np.mean(np.vstack(step_rewards), axis=0)
                    for t in range(0, simulation_horizon, curve_sample_every):
                        curve_rows.append(
                            {
                                "instance": instance_name,
                                "policy": policy_name,
                                "policy_type": policy_type(policy_name),
                                "N": N,
                                "time": t,
                                "mean_step_reward": mean_curve[t],
                            }
                        )

                status_text = "ok" if failures == 0 else f"{failures} failed"
                print(f"    {policy_name}: {status_text}")

    raw_df = pd.DataFrame(raw_rows)
    curve_df = pd.DataFrame(curve_rows)

    successful = raw_df[raw_df["status"] == "ok"]
    summary_df = (
        successful.groupby(["instance", "policy", "N"], as_index=False)
        .agg(
            policy_type=("policy_type", "first"),
            S=("S", "first"),
            A=("A", "first"),
            alpha=("alpha", "first"),
            budget=("budget", "first"),
            horizon=("horizon", "first"),
            mean_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", "std"),
            mean_tail_reward=("tail_mean_reward", "mean"),
            std_tail_reward=("tail_mean_reward", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            num_runs=("mean_reward", "size"),
        )
        .sort_values(["instance", "N", "policy"])
        .reset_index(drop=True)
    )

    raw_df.to_csv(OUTPUT_DIR / "unknown_model_raw_results.csv", index=False)
    summary_df.to_csv(
        OUTPUT_DIR / "unknown_model_summary.csv",
        index=False,
    )
    curve_df.to_csv(
        OUTPUT_DIR / "unknown_model_learning_curves.csv",
        index=False,
    )

    if make_plots and not summary_df.empty:
        max_N = max(n_values)
        for instance_name in instance_names:
            plot_metric(
                summary_df,
                instance_name,
                "mean_reward",
                "Average reward per arm",
                "reward_vs_N",
            )
            plot_metric(
                summary_df,
                instance_name,
                "mean_tail_reward",
                "Tail average reward per arm",
                "tail_reward_vs_N",
            )
            plot_learning_curve(curve_df, instance_name, max_N)

    print(f"\nSaved unknown-model outputs to:\n{OUTPUT_DIR}")
    return {
        "raw": raw_df,
        "summary": summary_df,
        "learning_curves": curve_df,
    }


if __name__ == "__main__":
    RESULTS = run_unknown_model_benchmark()
