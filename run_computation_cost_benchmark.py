"""
Measure the computational cost of RMAB policies without using simulation cache.

The experiment separates two costs:

1. setup_seconds:
   Time spent constructing a policy, including LP solves, Whittle-index
   computation, and Q-Whittle training.
2. simulation_seconds:
   Time spent making decisions and simulating transitions over a fixed horizon.

This script intentionally bypasses both cache layers used by the reward
benchmark:

- rmab_cache from simulation_cache.py
- computed_values from the teacher's strategies.simulate()

Therefore, running this file never deletes or modifies existing reward caches.
"""

import gc
import importlib
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

# Use an explicit path so this file also works when launched from Jupyter.
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import bandit_lp
import known_model_extra_instances as extra_instances_module
import make_policy as make_policy_module
import strategies

importlib.reload(make_policy_module)
importlib.reload(extra_instances_module)

from known_model_extra_instances import build_extended_known_model_instance_library
from benchmark_metadata import experiment_axes, write_experiment_settings
from simulation_utils import integer_budget


# Computation-cost results are kept separate from the reward benchmark.
OUTPUT_DIR = CURRENT_DIR / "computation_cost_outputs" / "corrected_v2"


# ============================================================
# 2. Experiment settings
# ============================================================

# Start with representative instances. Add more after checking the first run.
INSTANCE_NAMES = [
    "random_S10_seed123",
    "hong_counterexample",
    "conveyor_eg4unif-tb_S8",
    "wireless_channel_S10_a40",
]

POLICY_NAMES = [
    "WhittleIndexStrategy",
    "LPPriorityStrategy",
    "FTVA_Strategy",
    "LPupdateStrategy",
    "Myopic",
    "RandomPriority",
    "RoundRobin",
    "LPRandomized",
    "QWhittleKnownModel",
]

N_VALUES = [20, 100, 500]
NUM_REPETITIONS = 3
SETUP_REPETITIONS = 1
MC_SEED = 123
SIMULATION_HORIZON = 200
LP_UPDATE_HORIZON = 20


# ============================================================
# 3. Uncached simulation
# ============================================================

def simulate_uncached(bandit, policy, initial_state, N, horizon, seed):
    """
    Run one simulation without reading or writing any cache file.

    This reproduces the simulation loop in the teacher code, but deliberately
    omits its computed_values lookup so the measured time is genuine.
    """
    np.random.seed(seed)
    if hasattr(policy, "reset"):
        policy.reset()
    state_x = strategies.round_state_to_integer(initial_state, N)
    rewards = np.zeros(horizon)

    for t in range(horizon):
        next_x, reward, _ = strategies.simulation_step(bandit, policy, state_x, N)

        rewards[t] = reward
        state_x = next_x

    return float(np.mean(rewards))


def clear_qwhittle_memory_cache():
    """
    Remove only Q-Whittle's in-memory preprocessing result.

    This is not the disk-based rmab_cache. Clearing it is necessary to measure
    the true cold-start Q-learning time.
    """
    cache = getattr(make_policy_module, "_QWHITTLE_PRIORITY_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()


def create_policy(policy_name, bandit, alpha, random_seed):
    """Create a fresh policy object using the project's unified factory."""
    return make_policy_module.make_policy(
        policy_name=policy_name,
        bandit=bandit,
        alpha=alpha,
        lp_update_horizon=LP_UPDATE_HORIZON,
        random_seed=random_seed,
    )


def online_state_representation(policy_name):
    """
    Describe how the current implementation represents the N arms online.

    Aggregate-state policies operate on counts/fractions of arms in each state.
    Their measured runtime can be almost independent of N in the homogeneous
    simulator. FTVA and RoundRobin explicitly maintain N individual arm states.
    """
    if policy_name in {"FTVA_Strategy", "RoundRobin"}:
        return "individual_arm"
    return "aggregate_state"


# ============================================================
# 4. Summaries
# ============================================================

def estimate_runtime_scaling(group):
    """
    Fit simulation_time = c * N^gamma on a log-log scale.

    gamma near 1 suggests approximately linear growth in the number of arms.
    """
    valid = group[
        (group["mean_simulation_seconds"] > 0)
        & np.isfinite(group["mean_simulation_seconds"])
        & (group["N"] > 0)
        & np.isfinite(group["N"])
    ].drop_duplicates("N")
    if len(valid) < 2:
        return pd.Series(
            {
                "runtime_scaling_gamma": np.nan,
                "r_squared": np.nan,
                "num_fit_points": len(valid),
            }
        )

    log_n = np.log(valid["N"].to_numpy(dtype=float))
    log_time = np.log(valid["mean_simulation_seconds"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(log_n, log_time, deg=1)
    predicted = slope * log_n + intercept
    ss_total = np.sum((log_time - np.mean(log_time)) ** 2)
    r_squared = (
        np.nan
        if ss_total <= 0
        else 1.0 - np.sum((log_time - predicted) ** 2) / ss_total
    )

    return pd.Series(
        {
            "runtime_scaling_gamma": slope,
            "r_squared": r_squared,
            "num_fit_points": len(valid),
        }
    )


def safe_filename(name):
    return re.sub(r'[ /\\:]', "_", name)


def save_plots(setup_summary, cost_summary, output_dir):
    """Save simple line and bar figures without confidence intervals."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.8),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    for instance_name in cost_summary["instance"].unique():
        instance_cost = cost_summary[cost_summary["instance"] == instance_name]

        fig, ax = plt.subplots()
        for policy_name, group in instance_cost.groupby("policy", sort=True):
            group = group.sort_values("N")
            ax.plot(
                group["N"],
                group["mean_simulation_seconds"],
                marker="o",
                label=policy_name,
            )
        ax.set(
            title=instance_name,
            xlabel="Number of arms (N)",
            ylabel="Simulation time (seconds)",
            xscale="log",
            yscale="log",
        )
        plotted_n_values = sorted(instance_cost["N"].unique())
        ax.set_xticks(plotted_n_values)
        ax.set_xticklabels([str(n) for n in plotted_n_values])
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{safe_filename(instance_name)}_online_time_vs_N.png",
            bbox_inches="tight",
        )
        plt.close(fig)

        instance_setup = setup_summary[
            setup_summary["instance"] == instance_name
        ].sort_values("mean_setup_seconds")
        if not instance_setup.empty:
            fig, ax = plt.subplots()
            labels = instance_setup["policy"]
            if labels.duplicated().any():
                labels = labels + " (alpha=" + instance_setup["alpha"].map(lambda x: f"{x:.4g}") + ")"
            ax.barh(
                labels,
                instance_setup["mean_setup_seconds"],
            )
            ax.set(
                title=instance_name,
                xlabel="Cold policy setup time (seconds)",
                xscale="log",
            )
            fig.tight_layout()
            fig.savefig(
                output_dir / f"{safe_filename(instance_name)}_setup_time.png",
                bbox_inches="tight",
            )
            plt.close(fig)


# ============================================================
# 5. Main experiment
# ============================================================

def run_computation_cost_benchmark(
    instance_names=INSTANCE_NAMES,
    policy_names=POLICY_NAMES,
    n_values=N_VALUES,
    num_repetitions=NUM_REPETITIONS,
    setup_repetitions=SETUP_REPETITIONS,
    simulation_horizon=SIMULATION_HORIZON,
    output_dir=OUTPUT_DIR,
    make_plots=True,
):
    """
    Run a fresh computation-cost experiment and return all result dataframes.

    No simulation cache is accepted as an argument by design. Every simulation
    is recomputed, while the original rmab_cache directory remains untouched.
    """
    n_values, policy_names = experiment_axes(
        n_values, policy_names, num_repetitions=num_repetitions,
        setup_repetitions=setup_repetitions, simulation_horizon=simulation_horizon,
    )
    instance_names = list(instance_names)
    if not instance_names or len(set(instance_names)) != len(instance_names):
        raise ValueError("Select distinct, nonempty instance names.")
    instance_library = build_extended_known_model_instance_library(bandit_lp)
    missing = [name for name in instance_names if name not in instance_library]
    if missing:
        raise ValueError(f"Unknown instances: {missing}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_settings(
        output_dir, source_files=["run_computation_cost_benchmark.py", "benchmark_metadata.py",
                                  "bandit_lp.py", "strategies.py", "make_policy.py",
                                  "simulation_utils.py", "rmab_instances.py", "known_model_extra_instances.py"],
        instances=instance_names, policies=policy_names, N=n_values,
        num_repetitions=num_repetitions, setup_repetitions=setup_repetitions,
        horizon=simulation_horizon, mc_seed=MC_SEED, lp_update_horizon=LP_UPDATE_HORIZON,
        simulation_cache=False, setup_scope="one cold setup per instance, policy and effective alpha",
        simulation_scope="reset, initial state, decisions, rewards and transitions; excludes policy construction and file I/O",
        qwhittle_training=dict(gamma=make_policy_module.QWHITTLE_GAMMA,
                               num_penalties=make_policy_module.QWHITTLE_NUM_PENALTIES,
                               steps_per_penalty=make_policy_module.QWHITTLE_STEPS_PER_PENALTY,
                               seed=make_policy_module.QWHITTLE_TRAINING_SEED),
    )

    # N can change the feasible activation fraction through integer rounding.
    # Measure setup once for each actual fraction, not for an infeasible alpha.
    experiments = []
    for name in instance_names:
        by_alpha = {}
        for N in n_values:
            alpha = integer_budget(instance_library[name].default_alpha, N) / N
            by_alpha.setdefault(alpha, []).append(N)
        experiments.extend((name, alpha, sizes) for alpha, sizes in by_alpha.items())

    setup_rows = []
    simulation_rows = []

    for instance_name, alpha, run_n_values in experiments:
        spec = instance_library[instance_name]
        bandit = spec.bandit

        print(
            f"\nInstance: {instance_name} | "
            f"S={bandit.S} | A={bandit.A} | alpha={alpha}"
        )

        for policy_name in policy_names:
            print(f"  {policy_name}: measuring cold setup...", end="", flush=True)

            setup_succeeded = False
            setup_error = ""
            for setup_rep in range(setup_repetitions):
                try:
                    if policy_name == "QWhittleKnownModel":
                        clear_qwhittle_memory_cache()

                    gc.collect()
                    start = time.perf_counter()
                    create_policy(
                        policy_name,
                        bandit,
                        alpha,
                        random_seed=MC_SEED + setup_rep,
                    )
                    setup_seconds = time.perf_counter() - start
                    setup_succeeded = True

                    setup_rows.append(
                        {
                            "instance": instance_name,
                            "policy": policy_name,
                            "S": bandit.S,
                            "A": bandit.A,
                            "alpha": alpha,
                            "online_state_representation": (
                                online_state_representation(policy_name)
                            ),
                            "setup_replication": setup_rep,
                            "setup_seconds": setup_seconds,
                            "status": "ok",
                            "error": "",
                        }
                    )
                except Exception as exc:
                    setup_error = f"{type(exc).__name__}: {exc}"
                    setup_rows.append(
                        {
                            "instance": instance_name,
                            "policy": policy_name,
                            "S": bandit.S,
                            "A": bandit.A,
                            "alpha": alpha,
                            "online_state_representation": (
                                online_state_representation(policy_name)
                            ),
                            "setup_replication": setup_rep,
                            "setup_seconds": np.nan,
                            "status": "failed",
                            "error": setup_error,
                        }
                    )

            if not setup_succeeded:
                print(f" failed: {setup_error}")
                for N in run_n_values:
                    for rep in range(num_repetitions):
                        simulation_rows.append(dict(
                            instance=instance_name, policy=policy_name, S=bandit.S, A=bandit.A,
                            alpha=alpha, online_state_representation=online_state_representation(policy_name),
                            N=N, replication=rep, seed=MC_SEED + rep, horizon=simulation_horizon,
                            mean_reward=np.nan, simulation_seconds=np.nan, seconds_per_step=np.nan,
                            status="failed", error=f"Policy setup failed: {setup_error}",
                        ))
                pd.DataFrame(setup_rows).to_csv(output_dir / "setup_times_raw.csv", index=False)
                pd.DataFrame(simulation_rows).to_csv(output_dir / "simulation_times_raw.csv", index=False)
                continue

            print(" done; measuring online simulation.")

            for N in run_n_values:
                failures = 0
                for rep in range(num_repetitions):
                    seed = MC_SEED + rep
                    try:
                        # Policy creation is outside the online timer. Its cost
                        # has already been measured separately as setup_seconds.
                        policy = create_policy(
                            policy_name,
                            bandit,
                            alpha,
                            random_seed=seed,
                        )

                        start = time.perf_counter()
                        mean_reward = simulate_uncached(
                            bandit=bandit,
                            policy=policy,
                            initial_state=spec.initial_state,
                            N=N,
                            horizon=simulation_horizon,
                            seed=seed,
                        )
                        simulation_seconds = time.perf_counter() - start

                        simulation_rows.append(
                            {
                                "instance": instance_name,
                                "policy": policy_name,
                                "S": bandit.S,
                                "A": bandit.A,
                                "alpha": alpha,
                                "online_state_representation": (
                                    online_state_representation(policy_name)
                                ),
                                "N": N,
                                "replication": rep,
                                "seed": seed,
                                "horizon": simulation_horizon,
                                "mean_reward": mean_reward,
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
                        simulation_rows.append(
                            {
                                "instance": instance_name,
                                "policy": policy_name,
                                "S": bandit.S,
                                "A": bandit.A,
                                "alpha": alpha,
                                "online_state_representation": (
                                    online_state_representation(policy_name)
                                ),
                                "N": N,
                                "replication": rep,
                                "seed": seed,
                                "horizon": simulation_horizon,
                                "mean_reward": np.nan,
                                "simulation_seconds": np.nan,
                                "seconds_per_step": np.nan,
                                "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

                status = "ok" if failures == 0 else f"{failures} failed"
                print(f"    N={N}: {status}")
            pd.DataFrame(setup_rows).to_csv(output_dir / "setup_times_raw.csv", index=False)
            pd.DataFrame(simulation_rows).to_csv(output_dir / "simulation_times_raw.csv", index=False)

    setup_raw = pd.DataFrame(setup_rows)
    simulation_raw = pd.DataFrame(simulation_rows)

    setup_ok = setup_raw[setup_raw["status"] == "ok"]
    setup_summary = (
        setup_ok.groupby(["instance", "policy", "alpha"], as_index=False)
        .agg(
            S=("S", "first"),
            A=("A", "first"),
            online_state_representation=("online_state_representation", "first"),
            mean_setup_seconds=("setup_seconds", "mean"),
            min_setup_seconds=("setup_seconds", "min"),
            num_setup_runs=("setup_seconds", "size"),
        )
        .sort_values(["instance", "mean_setup_seconds"])
    )

    simulation_ok = simulation_raw[simulation_raw["status"] == "ok"]
    simulation_summary = (
        simulation_ok.groupby(["instance", "policy", "N"], as_index=False)
        .agg(
            S=("S", "first"),
            A=("A", "first"),
            alpha=("alpha", "first"),
            online_state_representation=("online_state_representation", "first"),
            horizon=("horizon", "first"),
            mean_reward=("mean_reward", "mean"),
            mean_simulation_seconds=("simulation_seconds", "mean"),
            min_simulation_seconds=("simulation_seconds", "min"),
            mean_seconds_per_step=("seconds_per_step", "mean"),
            num_runs=("simulation_seconds", "size"),
        )
        .sort_values(["instance", "N", "policy"])
    )

    cost_summary = simulation_summary.merge(
        setup_summary[
            ["instance", "policy", "alpha", "mean_setup_seconds", "min_setup_seconds"]
        ],
        on=["instance", "policy", "alpha"],
        how="left",
    )
    cost_summary["estimated_cold_total_seconds"] = (
        cost_summary["mean_setup_seconds"]
        + cost_summary["mean_simulation_seconds"]
    )

    scaling_rows = []
    for (instance, policy, alpha), group in simulation_summary.groupby(["instance", "policy", "alpha"]):
        scaling_rows.append(dict(instance=instance, policy=policy, alpha=alpha,
                                 **estimate_runtime_scaling(group).to_dict()))
    scaling_df = pd.DataFrame(scaling_rows, columns=[
        "instance", "policy", "alpha", "runtime_scaling_gamma", "r_squared", "num_fit_points",
    ]).sort_values(["instance", "runtime_scaling_gamma"])

    setup_raw.to_csv(output_dir / "setup_times_raw.csv", index=False)
    simulation_raw.to_csv(output_dir / "simulation_times_raw.csv", index=False)
    setup_summary.to_csv(output_dir / "setup_time_summary.csv", index=False)
    cost_summary.to_csv(output_dir / "computation_cost_summary.csv", index=False)
    scaling_df.to_csv(output_dir / "runtime_scaling_by_policy.csv", index=False)

    if make_plots and not cost_summary.empty:
        save_plots(setup_summary, cost_summary, output_dir)

    print(f"\nSaved computation-cost outputs to:\n{output_dir}")
    return {
        "setup_raw": setup_raw,
        "simulation_raw": simulation_raw,
        "setup_summary": setup_summary,
        "cost_summary": cost_summary,
        "runtime_scaling": scaling_df,
    }


if __name__ == "__main__":
    RESULTS = run_computation_cost_benchmark()
