"""
Run and summarize computation-cost experiments across benchmark families.

This script is the "systematic cost" layer. It combines:

1. known-model homogeneous policies;
2. known-model heterogeneous policies;
3. unknown-model online-learning policies.

It never deletes or rewrites the reward cache. Cost outputs are written to a
separate folder.
"""

import importlib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and imports
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import run_computation_cost_benchmark as known_cost_module
import run_heterogeneous_benchmark as heterogeneous_cost_module
import run_unknown_model_benchmark as unknown_cost_module
from paper_config import filter_paper_instances
from simulation_utils import SIMULATION_VERSION

importlib.reload(known_cost_module)
importlib.reload(heterogeneous_cost_module)
importlib.reload(unknown_cost_module)


OUTPUT_DIR = CURRENT_DIR / "computation_cost_suite_outputs" / "corrected_v2"


# ============================================================
# 2. Experiment settings
# ============================================================

KNOWN_MODEL_INSTANCES = [
    "random_S10_seed123",
    "hong_counterexample",
    "conveyor_eg4unif-tb_S8",
    "maintenance_S10_a20",
    "wireless_channel_S10_a40",
    "deadline_S10_a20",
]

KNOWN_MODEL_POLICIES = [
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

HETEROGENEOUS_POLICIES = [
    "HeterogeneousWhittle",
    "HeterogeneousLPPriority",
    "HeterogeneousFTVA",
    "HeterogeneousMyopic",
    "RandomActivation",
    "RoundRobin",
    # "HeterogeneousLPUpdate",  # implemented, but very slow.
]

UNKNOWN_MODEL_INSTANCES = [
    "random_S10_seed123",
    "maintenance_S10_a40",
    "wireless_channel_S10_a40",
]

UNKNOWN_MODEL_POLICIES = [
    "KnownWhittleOracle",
    "KnownLPPriorityOracle",
    "OnlinePlugInWhittle",
    "OnlineQLearningIndex",
    "OnlineUCBReward",
    "OnlineRewardGreedy",
    "OnlineRandom",
]

# Moderate defaults. Increase these for final paper runs.
N_VALUES_KNOWN = [20, 100, 500]
N_VALUES_HETEROGENEOUS = [20, 100, 500]
N_VALUES_UNKNOWN = [20, 100, 200]


# ============================================================
# 3. Normalization helpers
# ============================================================

def normalize_known_cost(cost_summary):
    df = filter_paper_instances(cost_summary.copy())
    return pd.DataFrame(
        {
            "experiment_family": "known_model_homogeneous",
            "model_knowledge": "known_P_R",
            "arm_structure": "homogeneous",
            "instance": df["instance"],
            "policy": df["policy"],
            "N": df["N"],
            "S": df["S"],
            "A": df["A"],
            "alpha": df["alpha"],
            "reward_metric": df["mean_reward"],
            "setup_seconds": df["mean_setup_seconds"],
            "online_seconds": df["mean_simulation_seconds"],
            "seconds_per_step": df["mean_seconds_per_step"],
            "total_seconds": df["estimated_cold_total_seconds"],
            "cost_scope": "setup_plus_uncached_simulation",
        }
    )


def normalize_heterogeneous_cost(summary):
    df = summary.copy()
    return pd.DataFrame(
        {
            "experiment_family": "known_model_heterogeneous",
            "model_knowledge": "known_P_i_R_i",
            "arm_structure": "heterogeneous",
            "instance": df["instance"],
            "policy": df["policy"],
            "N": df["N"],
            "S": np.nan,
            "A": 2,
            "alpha": df["alpha"],
            "reward_metric": df["mean_reward"],
            "setup_seconds": df["setup_seconds"],
            "online_seconds": df["mean_simulation_seconds"],
            "seconds_per_step": df["mean_seconds_per_step"],
            "total_seconds": df["estimated_cold_total_seconds"],
            "cost_scope": "setup_plus_explicit_arm_simulation",
        }
    )


def normalize_unknown_cost(summary):
    df = summary.copy()
    required = {"policy_type", "mean_setup_seconds", "mean_simulation_seconds"}
    if not required.issubset(df.columns):
        raise ValueError("Online timing requires the corrected split timers and oracle labels; rerun old results.")
    if not df["policy_type"].isin(["known_model_oracle", "unknown_model_online"]).all():
        raise ValueError("Online timing contains an unrecognized policy type.")
    oracle = df["policy_type"].eq("known_model_oracle")
    return pd.DataFrame(
        {
            "experiment_family": "unknown_model_online",
            "model_knowledge": np.where(oracle, "known_P_R_oracle", "unknown_P_R"),
            "arm_structure": "homogeneous_environment_explicit_arms",
            "instance": df["instance"],
            "policy": df["policy"],
            "N": df["N"],
            "S": df["S"],
            "A": df["A"],
            "alpha": df["alpha"],
            "reward_metric": df["mean_tail_reward"],
            "setup_seconds": df["mean_setup_seconds"],
            "online_seconds": df["mean_simulation_seconds"],
            "seconds_per_step": df["mean_simulation_seconds"] / df["horizon"],
            "total_seconds": df["mean_runtime_seconds"],
            "cost_scope": "construction_plus_learning_and_explicit_arm_simulation",
        }
    )


def summarize_combined_cost(combined):
    return (
        combined.groupby(
            [
                "experiment_family",
                "model_knowledge",
                "arm_structure",
                "policy",
            ],
            as_index=False,
        )
        .agg(
            mean_setup_seconds=("setup_seconds", "mean"),
            mean_online_seconds=("online_seconds", "mean"),
            mean_total_seconds=("total_seconds", "mean"),
            max_total_seconds=("total_seconds", "max"),
            mean_seconds_per_step=("seconds_per_step", "mean"),
            num_rows=("total_seconds", "size"),
        )
        .sort_values(
            ["experiment_family", "mean_total_seconds"],
            ascending=[True, False],
        )
    )


def write_cost_readme(output_dir):
    text = """# Computation Cost Suite

This folder combines timing results from the explicitly selected experiment families.
The corrected default includes known-model homogeneous and heterogeneous experiments.
Unknown-model timing remains exploratory and requires explicit opt-in.

- `known_model_homogeneous`: algorithms know the shared true P,R.
- `known_model_heterogeneous`: algorithms know each arm type's P_i,R_i.
- `unknown_model_online`: an online-learning protocol including known-model reference oracles. Check `model_knowledge` to distinguish learners from oracles.

Important interpretation:

- `setup_seconds` is preprocessing time, such as LP solving, Whittle-index computation, or Q-Whittle training.
- `online_seconds` is the time spent making decisions and simulating transitions.
- `total_seconds` is setup + online time when both are separately measured.
- For the online protocol, `setup_seconds` measures construction (including true-model oracle preprocessing). `online_seconds` measures reset, interactions and learning; `total_seconds` is their sum. Passive-arm outcomes are observed, and learned tables pool all N arm samples each period.

`HeterogeneousLPUpdate` is implemented but not enabled by default because solving a finite-horizon LP for each new population state can be very slow. Its within-trajectory solve cache is cleared between replications.

These times include environment simulation, not only policy decision time. Average times over different sets of successful instances are not a controlled algorithm ranking; check coverage and raw failure records first.
"""
    (output_dir / "README_computation_cost_suite.md").write_text(
        text,
        encoding="utf-8",
    )


def read_corrected_cost(path):
    """Refuse historical or stale timing files instead of silently combining them."""
    path = Path(path)
    settings_path = path.parent / "experiment_settings.json"
    if not path.is_file() or not settings_path.is_file():
        raise FileNotFoundError(f"Missing corrected timing data or metadata: {path}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if settings.get("simulation_version") != SIMULATION_VERSION or not settings.get("source_sha256"):
        raise ValueError(f"Missing corrected simulator provenance: {settings_path}")
    for name, expected_hash in settings["source_sha256"].items():
        source = (CURRENT_DIR / name).resolve()
        if source.parent != CURRENT_DIR or not source.is_file():
            raise ValueError(f"Unknown timing source file: {name}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Timing was generated by different code ({name}); rerun before collecting it.")
    return pd.read_csv(path)


def collect_existing_computation_cost_outputs(
    output_dir=OUTPUT_DIR, *,
    known_dir=known_cost_module.OUTPUT_DIR,
    heterogeneous_dir=heterogeneous_cost_module.OUTPUT_DIR,
    unknown_dir=None,
):
    """
    Combine cost CSV files that already exist, without rerunning simulations.

    Source folders require matching code fingerprints. Pass None to omit a
    family. Online results require an explicit corrected unknown_dir; no
    historical unknown-model results are imported by this collector.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    if known_dir is not None:
        known_cost = read_corrected_cost(Path(known_dir) / "computation_cost_summary.csv")
        frames.append(normalize_known_cost(known_cost))
    if heterogeneous_dir is not None:
        heterogeneous_cost = read_corrected_cost(Path(heterogeneous_dir) / "heterogeneous_summary.csv")
        frames.append(normalize_heterogeneous_cost(heterogeneous_cost))
    if unknown_dir is not None:
        unknown_cost = read_corrected_cost(Path(unknown_dir) / "unknown_model_summary.csv")
        frames.append(normalize_unknown_cost(unknown_cost))

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    combined.to_csv(
        output_dir / "combined_computation_cost_summary.csv",
        index=False,
    )
    ranking = summarize_combined_cost(combined) if not combined.empty else pd.DataFrame()
    ranking.to_csv(
        output_dir / "combined_computation_cost_ranking.csv",
        index=False,
    )
    write_cost_readme(output_dir)

    print(f"Collected existing computation-cost outputs into:\n{output_dir}")
    return {
        "combined": combined,
        "ranking": ranking,
    }


# ============================================================
# 4. Main suite
# ============================================================

def run_computation_cost_suite(
    output_dir=OUTPUT_DIR,
    run_known=True,
    run_heterogeneous=True,
    run_unknown=False,
):
    if not any((run_known, run_heterogeneous, run_unknown)):
        raise ValueError("Select at least one experiment family.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_frames = []

    if run_known:
        known_results = known_cost_module.run_computation_cost_benchmark(
            instance_names=KNOWN_MODEL_INSTANCES,
            policy_names=KNOWN_MODEL_POLICIES,
            n_values=N_VALUES_KNOWN,
            num_repetitions=3,
            setup_repetitions=1,
            simulation_horizon=200,
            output_dir=output_dir / "known_model_homogeneous",
            make_plots=True,
        )
        normalized_frames.append(
            normalize_known_cost(known_results["cost_summary"])
        )

    if run_heterogeneous:
        heterogeneous_results = heterogeneous_cost_module.run_heterogeneous_benchmark(
            n_values=N_VALUES_HETEROGENEOUS,
            num_monte_carlo=3,
            simulation_horizon=200,
            num_arm_types=5,
            policy_names=HETEROGENEOUS_POLICIES,
            output_dir=output_dir / "known_model_heterogeneous",
            make_plots=True,
        )
        normalized_frames.append(
            normalize_heterogeneous_cost(heterogeneous_results["summary"])
        )

    if run_unknown:
        unknown_results = unknown_cost_module.run_unknown_model_benchmark(
            instance_names=UNKNOWN_MODEL_INSTANCES,
            policy_names=UNKNOWN_MODEL_POLICIES,
            n_values=N_VALUES_UNKNOWN,
            num_monte_carlo=3,
            simulation_horizon=1000,
            curve_sample_every=10,
            output_dir=output_dir / "unknown_model_online",
            make_plots=True,
        )
        normalized_frames.append(
            normalize_unknown_cost(unknown_results["summary"])
        )

    if normalized_frames:
        combined = pd.concat(normalized_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    combined.to_csv(
        output_dir / "combined_computation_cost_summary.csv",
        index=False,
    )
    cost_ranking = summarize_combined_cost(combined) if not combined.empty else pd.DataFrame()
    cost_ranking.to_csv(
        output_dir / "combined_computation_cost_ranking.csv",
        index=False,
    )
    write_cost_readme(output_dir)

    print(f"\nSaved computation-cost suite outputs to:\n{output_dir}")
    return {
        "combined": combined,
        "ranking": cost_ranking,
    }


if __name__ == "__main__":
    RESULTS = run_computation_cost_suite()
