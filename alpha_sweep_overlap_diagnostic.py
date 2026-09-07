r"""
Small alpha-sweep diagnostic for policy-overlap behavior.

This script is intentionally separate from the main benchmark. It does not
modify existing benchmark scripts or cached results. It checks whether changing
alpha separates policies that overlap in the final figures.

Outputs:

    paper_summary_outputs\alpha_sweep_overlap_summary.csv
"""

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import bandit_lp
import strategies
from known_model_extra_instances import build_extended_known_model_instance_library
from make_policy import make_policy


OUTPUT_DIR = CURRENT_DIR / "paper_summary_outputs"

INSTANCE_NAMES = [
    "deadline_S10_a20",
    "maintenance_S10_a20",
    "wireless_channel_S10_a40",
    "yan_gast_example2",
    "yan_gast_example3",
]

# Small diagnostic grid. These values are enough to see whether overlap is
# caused by the selected alpha or by the instance structure.
ALPHA_VALUES = [0.1, 0.2, 0.4, 0.6, 0.8]

# Keep this fast. This is a diagnostic, not the final benchmark.
N = 200
TIME_HORIZON = 100
SEED = 123

POLICY_NAMES = [
    "WhittleIndexStrategy",
    "LPPriorityStrategy",
    "Myopic",
    "LPupdateStrategy",
    "LPRandomized",
]


def policy_signature(policy, bandit, initial_state, N, time_horizon, seed):
    """Return a compact signature of one policy's simulated actions."""
    mean_reward, x_values, reward_values, y_values = strategies.simulate(
        bandit=bandit,
        strategy=policy,
        initial_state=initial_state,
        N=N,
        time=time_horizon,
        seed=seed,
    )
    return {
        "mean_reward": float(mean_reward),
        "x_values": x_values,
        "reward_values": reward_values,
        "y_values": y_values,
        "action_signature": y_values.round(12).tobytes(),
    }


def priority_order(policy):
    if hasattr(policy, "order_of_states"):
        return ",".join(map(str, policy.order_of_states.tolist()))
    return ""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance_library = build_extended_known_model_instance_library(bandit_lp)
    rows = []

    for instance_name in INSTANCE_NAMES:
        spec = instance_library[instance_name]
        bandit = spec.bandit

        for alpha in ALPHA_VALUES:
            trajectories = {}
            errors = {}

            for policy_name in POLICY_NAMES:
                try:
                    policy = make_policy(
                        policy_name=policy_name,
                        bandit=bandit,
                        alpha=alpha,
                        lp_update_horizon=20,
                        random_seed=SEED,
                    )
                    trajectories[policy_name] = policy_signature(
                        policy=policy,
                        bandit=bandit,
                        initial_state=spec.initial_state,
                        N=N,
                        time_horizon=TIME_HORIZON,
                        seed=SEED,
                    )
                    trajectories[policy_name]["priority_order"] = priority_order(policy)
                except Exception as exc:
                    errors[policy_name] = f"{type(exc).__name__}: {exc}"

            for policy_name in POLICY_NAMES:
                traj = trajectories.get(policy_name)
                rows.append(
                    {
                        "instance": instance_name,
                        "alpha": alpha,
                        "policy": policy_name,
                        "status": "ok" if traj is not None else "failed",
                        "error": errors.get(policy_name, ""),
                        "mean_reward": np.nan if traj is None else traj["mean_reward"],
                        "priority_order": "" if traj is None else traj["priority_order"],
                    }
                )

            for policy_a, policy_b in combinations(trajectories.keys(), 2):
                traj_a = trajectories[policy_a]
                traj_b = trajectories[policy_b]
                identical_y = traj_a["action_signature"] == traj_b["action_signature"]
                max_reward_diff = float(
                    np.max(
                        np.abs(
                            traj_a["reward_values"]
                            - traj_b["reward_values"]
                        )
                    )
                )
                rows.append(
                    {
                        "instance": instance_name,
                        "alpha": alpha,
                        "policy": f"{policy_a} == {policy_b}",
                        "status": "pair_check",
                        "error": "",
                        "mean_reward": np.nan,
                        "priority_order": "",
                        "identical_action_trajectory": identical_y,
                        "max_reward_path_diff": max_reward_diff,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "alpha_sweep_overlap_summary.csv", index=False)

    pair_df = df[df["status"] == "pair_check"].copy()
    print("Saved alpha sweep diagnostic to:")
    print(OUTPUT_DIR / "alpha_sweep_overlap_summary.csv")
    print("\nIdentical action pairs by instance/alpha:")
    same = pair_df[pair_df["identical_action_trajectory"] == True]
    if same.empty:
        print("None")
    else:
        print(
            same[
                [
                    "instance",
                    "alpha",
                    "policy",
                    "max_reward_path_diff",
                ]
            ].to_string(index=False)
        )

    print("\nNumber of identical pairs by instance/alpha:")
    counts = (
        pair_df.groupby(["instance", "alpha"])["identical_action_trajectory"]
        .sum()
        .reset_index(name="num_identical_policy_pairs")
    )
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
