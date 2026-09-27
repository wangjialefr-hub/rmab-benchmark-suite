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
from matplotlib.ticker import NullFormatter
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
from benchmark_metadata import experiment_axes, write_experiment_settings
from simulation_utils import integer_budget


OUTPUT_DIR = CURRENT_DIR / "unknown_model_outputs" / "corrected_v2"


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


def plot_metric(summary_df, instance_name, metric, ylabel, suffix, output_dir=OUTPUT_DIR):
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
    plotted_n = sorted(instance_df["N"].unique())
    ax.set_xticks(plotted_n)
    ax.set_xticklabels([str(n) for n in plotted_n])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(
        Path(output_dir) / f"{safe_filename(instance_name)}_{suffix}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_learning_curve(curve_df, instance_name, N, output_dir=OUTPUT_DIR):
    """Plot per-step reward averaged over seeds at the stored sample times."""
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

    ax.set_xlabel("Completed simulation steps")
    ax.set_ylabel("Mean per-arm reward at sampled steps")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(
        Path(output_dir) / f"{safe_filename(instance_name)}_learning_curve_N{N}.png",
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
    tail_fraction=0.25,
):
    n_values, policy_names = experiment_axes(
        n_values, policy_names, num_monte_carlo=num_monte_carlo,
        simulation_horizon=simulation_horizon, curve_sample_every=curve_sample_every,
    )
    instance_names = list(instance_names)
    if not instance_names or len(set(instance_names)) != len(instance_names):
        raise ValueError("Select distinct, nonempty instance names.")
    if not np.isfinite(tail_fraction) or not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in (0, 1].")
    instance_library = build_extended_known_model_instance_library(
        bandit_lp,
        include_original=True,
    )
    missing = [name for name in instance_names if name not in instance_library]
    if missing:
        raise ValueError(f"Unknown instances: {missing}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_settings(
        output_dir, source_files=["run_unknown_model_benchmark.py", "unknown_model_learning.py",
                                  "benchmark_metadata.py", "simulation_utils.py", "bandit_lp.py",
                                  "strategies.py", "known_model_extra_instances.py", "rmab_instances.py"],
        instances=instance_names, policies=policy_names, N=n_values,
        num_monte_carlo=num_monte_carlo, horizon=simulation_horizon,
        tail_fraction=tail_fraction, curve_sample_every=curve_sample_every, mc_seed=MC_SEED,
        simulation_cache=False,
        feedback="all arm states, deterministic R[s,a] rewards and next states, including passive arms",
        sample_pooling="one shared table over homogeneous arms; N samples per time step",
        initialization="IID arm states from the supplied distribution; a fresh learner for each seed",
        setup_scope="policy construction, including true-model preprocessing for oracles",
        simulation_scope="reset, initial states, decisions, transitions, rewards and learning; excludes file I/O",
        curve_metric="per-step per-arm reward averaged over successful seeds; sampled without smoothing",
        oracle_policies=[name for name in policy_names if name in ORACLE_POLICY_NAMES],
    )

    metadata_df = pd.DataFrame(
        extra_instance_metadata_dataframe(instance_library)
    )
    metadata_df[metadata_df["instance"].isin(instance_names)].to_csv(
        output_dir / "unknown_model_instance_library.csv",
        index=False,
    )

    raw_rows = []
    curve_rows = []

    for instance_name in instance_names:
        spec = instance_library[instance_name]
        bandit = spec.bandit
        requested_alpha = spec.default_alpha
        print(f"\nInstance: {instance_name}  S={bandit.S}, alpha={requested_alpha}", flush=True)

        for N in n_values:
            budget = integer_budget(requested_alpha, N)
            alpha = budget / N
            print(f"  N={N}, budget={budget}", flush=True)

            for policy_name in policy_names:
                failures = 0
                step_rewards = []

                for replication in range(num_monte_carlo):
                    seed = MC_SEED + replication
                    start = time.perf_counter()
                    phase = "setup"
                    row = dict(
                        instance=instance_name, policy=policy_name, policy_type=policy_type(policy_name),
                        N=N, S=bandit.S, A=bandit.A, alpha=alpha, requested_alpha=requested_alpha,
                        budget=budget, replication=replication, seed=seed, horizon=simulation_horizon,
                        tail_fraction=tail_fraction, tail_steps=simulation_horizon - int((1 - tail_fraction) * simulation_horizon),
                        mean_reward=np.nan, tail_mean_reward=np.nan,
                        setup_seconds=np.nan, simulation_seconds=np.nan, runtime_seconds=np.nan,
                        num_index_recomputations=0, num_index_failures=0, num_fallback_decisions=0,
                        last_index_error="", status="failed", failure_phase="", error="",
                    )

                    try:
                        policy = make_unknown_model_policy(
                            policy_name,
                            bandit=bandit,
                            N=N,
                            alpha=alpha,
                        )
                        row["setup_seconds"] = time.perf_counter() - start
                        phase = "simulation"
                        simulation_start = time.perf_counter()
                        result = simulate_unknown_model(
                            bandit=bandit,
                            policy=policy,
                            initial_state=spec.initial_state,
                            N=N,
                            horizon=simulation_horizon,
                            seed=seed,
                            tail_fraction=tail_fraction,
                        )
                        row["simulation_seconds"] = time.perf_counter() - simulation_start
                        row["runtime_seconds"] = row["setup_seconds"] + row["simulation_seconds"]
                        row.update(mean_reward=result["mean_reward"], tail_mean_reward=result["tail_mean_reward"], status="ok")
                        for field in ("num_index_recomputations", "num_index_failures", "num_fallback_decisions", "last_index_error"):
                            row[field] = result[field]
                        step_rewards.append(result["reward_history"])

                    except Exception as exc:
                        failures += 1
                        row.update(error=f"{type(exc).__name__}: {exc}", failure_phase=phase,
                                   runtime_seconds=time.perf_counter() - start)
                        if phase == "simulation":
                            for field in ("num_index_recomputations", "num_index_failures", "num_fallback_decisions", "last_index_error"):
                                row[field] = getattr(policy, field, row[field])
                    raw_rows.append(row)

                if step_rewards:
                    mean_curve = np.mean(np.vstack(step_rewards), axis=0)
                    sampled_steps = sorted(set(range(0, simulation_horizon, curve_sample_every)) | {simulation_horizon - 1})
                    for t in sampled_steps:
                        curve_rows.append(
                            {
                                "instance": instance_name,
                                "policy": policy_name,
                                "policy_type": policy_type(policy_name),
                                "N": N,
                                "time": t + 1,
                                "mean_step_reward": mean_curve[t],
                                "num_runs": len(step_rewards),
                                "complete": len(step_rewards) == num_monte_carlo,
                            }
                        )

                status_text = "ok" if failures == 0 else f"{failures} failed"
                fallback_runs = sum(r["num_index_failures"] > 0 for r in raw_rows[-num_monte_carlo:])
                if fallback_runs:
                    status_text += f", {fallback_runs} runs used an index-solver fallback"
                print(f"    {policy_name}: {status_text}", flush=True)
                pd.DataFrame(raw_rows).to_csv(output_dir / "unknown_model_raw_results.csv", index=False)

    raw_df = pd.DataFrame(raw_rows)
    curve_df = pd.DataFrame(curve_rows, columns=["instance", "policy", "policy_type", "N", "time", "mean_step_reward", "num_runs", "complete"])

    successful = raw_df[raw_df["status"] == "ok"]
    summary_df = (
        successful.groupby(["instance", "policy", "N"], as_index=False)
        .agg(
            policy_type=("policy_type", "first"),
            S=("S", "first"),
            A=("A", "first"),
            alpha=("alpha", "first"),
            requested_alpha=("requested_alpha", "first"),
            budget=("budget", "first"),
            horizon=("horizon", "first"),
            tail_fraction=("tail_fraction", "first"),
            tail_steps=("tail_steps", "first"),
            mean_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", "std"),
            mean_tail_reward=("tail_mean_reward", "mean"),
            std_tail_reward=("tail_mean_reward", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_setup_seconds=("setup_seconds", "mean"),
            mean_simulation_seconds=("simulation_seconds", "mean"),
            total_index_recomputations=("num_index_recomputations", "sum"),
            total_index_failures=("num_index_failures", "sum"),
            total_fallback_decisions=("num_fallback_decisions", "sum"),
            num_runs=("mean_reward", "size"),
        )
        .sort_values(["instance", "N", "policy"])
        .reset_index(drop=True)
    )
    summary_df["complete"] = summary_df["num_runs"].eq(num_monte_carlo)

    raw_df.to_csv(output_dir / "unknown_model_raw_results.csv", index=False)
    summary_df.to_csv(
        output_dir / "unknown_model_summary.csv",
        index=False,
    )
    curve_df.to_csv(
        output_dir / "unknown_model_learning_curves.csv",
        index=False,
    )

    plotted_summary = summary_df[summary_df["complete"]]
    plotted_curves = curve_df[curve_df["complete"].eq(True)]
    if make_plots and not plotted_summary.empty:
        max_N = max(n_values)
        for instance_name in plotted_summary["instance"].unique():
            plot_metric(
                plotted_summary,
                instance_name,
                "mean_reward",
                "Average reward per arm",
                "reward_vs_N",
                output_dir=output_dir,
            )
            plot_metric(
                plotted_summary,
                instance_name,
                "mean_tail_reward",
                "Tail average reward per arm",
                "tail_reward_vs_N",
                output_dir=output_dir,
            )
            if ((plotted_curves["instance"] == instance_name) & (plotted_curves["N"] == max_N)).any():
                plot_learning_curve(plotted_curves, instance_name, max_N, output_dir=output_dir)

    print(f"\nSaved unknown-model outputs to:\n{output_dir}", flush=True)
    return {
        "raw": raw_df,
        "summary": summary_df,
        "learning_curves": curve_df,
    }


if __name__ == "__main__":
    RESULTS = run_unknown_model_benchmark()
