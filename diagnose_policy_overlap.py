r"""
Diagnose why some policy curves overlap exactly.

This script does not run new experiments when cache files already exist. It
loads cached trajectories and checks whether two policies produced identical
actions, states, and rewards for the same instance/N/seed.

Outputs are saved under ``paper_summary_outputs`` next to this script.
"""

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import bandit_lp
from known_model_extra_instances import build_extended_known_model_instance_library
from make_policy import POLICY_NAMES, make_policy
from paper_config import KNOWN_MODEL_BENCHMARK_INSTANCES
from simulation_cache import cached_simulate_policy


OUTPUT_DIR = CURRENT_DIR / "paper_summary_outputs"
CACHE_DIR = CURRENT_DIR / "rmab_cache"
SUMMARY_PATH = CURRENT_DIR / "instance_matrix_outputs" / "summary_instance_matrix.csv"

N_TO_CHECK = 500
TIME_HORIZON = 200
LP_UPDATE_HORIZON = 20
SEEDS_TO_CHECK = list(range(123, 143))  # same 20 Monte Carlo seeds as main run


def trajectory_for_policy(spec, policy_name, N, seed):
    """Load one cached trajectory. Return None if that policy failed."""
    try:
        result = cached_simulate_policy(
            bandit=spec.bandit,
            policy_name=policy_name,
            make_policy=make_policy,
            initial_state=spec.initial_state,
            N=N,
            time_horizon=TIME_HORIZON,
            seed=seed,
            alpha=spec.default_alpha,
            lp_update_horizon=LP_UPDATE_HORIZON,
            cache_dir=CACHE_DIR,
            return_info=True,
        )
    except Exception:
        return None

    mean_reward, x_values, reward_values, y_values, cache_info = result
    if not cache_info["cache_hit"]:
        # This should not happen after the final benchmark run. Keep the flag so
        # we do not accidentally over-interpret a freshly recomputed result.
        print(f"Warning: cache miss for {policy_name}, N={N}, seed={seed}")

    return {
        "mean_reward": mean_reward,
        "x_values": x_values,
        "reward_values": reward_values,
        "y_values": y_values,
    }


def max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))


def diagnostic_rows_for_instance(instance_name, spec, policy_names):
    pair_rows = []

    # Load all trajectories once.
    trajectories = {
        seed: {
            policy: trajectory_for_policy(spec, policy, N_TO_CHECK, seed)
            for policy in policy_names
        }
        for seed in SEEDS_TO_CHECK
    }

    for policy_a, policy_b in combinations(policy_names, 2):
        checked = 0
        identical_y = 0
        identical_x = 0
        identical_reward_path = 0
        max_y_diff_all = 0.0
        max_x_diff_all = 0.0
        max_reward_diff_all = 0.0

        for seed in SEEDS_TO_CHECK:
            traj_a = trajectories[seed].get(policy_a)
            traj_b = trajectories[seed].get(policy_b)
            if traj_a is None or traj_b is None:
                continue

            checked += 1
            y_diff = max_abs_diff(traj_a["y_values"], traj_b["y_values"])
            x_diff = max_abs_diff(traj_a["x_values"], traj_b["x_values"])
            r_diff = max_abs_diff(
                traj_a["reward_values"], traj_b["reward_values"]
            )

            max_y_diff_all = max(max_y_diff_all, y_diff)
            max_x_diff_all = max(max_x_diff_all, x_diff)
            max_reward_diff_all = max(max_reward_diff_all, r_diff)

            identical_y += int(y_diff == 0.0)
            identical_x += int(x_diff == 0.0)
            identical_reward_path += int(r_diff == 0.0)

        pair_rows.append(
            {
                "instance": instance_name,
                "N": N_TO_CHECK,
                "time_horizon": TIME_HORIZON,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "num_seeds_checked": checked,
                "num_seeds_identical_y": identical_y,
                "num_seeds_identical_x": identical_x,
                "num_seeds_identical_reward_path": identical_reward_path,
                "identical_y_all_checked_seeds": (
                    checked > 0 and identical_y == checked
                ),
                "identical_full_trajectory_all_checked_seeds": (
                    checked > 0
                    and identical_y == checked
                    and identical_x == checked
                    and identical_reward_path == checked
                ),
                "max_y_diff_all_checked_seeds": max_y_diff_all,
                "max_x_diff_all_checked_seeds": max_x_diff_all,
                "max_reward_diff_all_checked_seeds": max_reward_diff_all,
            }
        )

    group_rows = []
    first_seed = SEEDS_TO_CHECK[0]
    signature_groups = {}
    for policy in policy_names:
        traj = trajectories[first_seed].get(policy)
        if traj is None:
            continue
        signature = traj["y_values"].round(12).tobytes()
        signature_groups.setdefault(signature, []).append(policy)

    for group in signature_groups.values():
        if len(group) > 1:
            group_rows.append(
                {
                    "instance": instance_name,
                    "N": N_TO_CHECK,
                    "seed": first_seed,
                    "group_size": len(group),
                    "identical_action_policies": " = ".join(group),
                }
            )

    return pair_rows, group_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_PATH)
    instance_library = build_extended_known_model_instance_library(bandit_lp)

    all_pair_rows = []
    all_group_rows = []

    for instance_name in KNOWN_MODEL_BENCHMARK_INSTANCES:
        if instance_name not in instance_library:
            continue

        present_policies = [
            policy
            for policy in POLICY_NAMES
            if (
                (summary["instance"] == instance_name)
                & (summary["policy"] == policy)
                & (summary["N"] == N_TO_CHECK)
            ).any()
        ]

        pair_rows, group_rows = diagnostic_rows_for_instance(
            instance_name,
            instance_library[instance_name],
            present_policies,
        )
        all_pair_rows.extend(pair_rows)
        all_group_rows.extend(group_rows)

    pair_df = pd.DataFrame(all_pair_rows)
    group_df = pd.DataFrame(all_group_rows)

    pair_df.to_csv(OUTPUT_DIR / "policy_overlap_pairs.csv", index=False)
    group_df.to_csv(OUTPUT_DIR / "policy_overlap_groups.csv", index=False)

    print("Saved policy-overlap diagnostics to:")
    print(OUTPUT_DIR)
    if not group_df.empty:
        print("\nIdentical action groups at N=500, seed=123:")
        print(group_df.to_string(index=False))


if __name__ == "__main__":
    main()
