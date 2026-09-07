"""Small public interface for the known-model RMAB benchmark.

The full experiment scripts remain available, but this module is the simplest
entry point for users who want to inspect an instance or run one policy.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import bandit_lp
from known_model_extra_instances import (
    build_extended_known_model_instance_library,
    extra_instance_metadata_dataframe,
)
from make_policy import POLICY_NAMES, make_policy
from simulation_cache import cached_lp_upper_bound, cached_simulate_policy


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT_DIR / "rmab_cache"


def available_policies():
    """Return the names accepted by :func:`run_named_experiment`."""
    return tuple(POLICY_NAMES)


def instance_library():
    """Build and return all named homogeneous known-model instances."""
    return build_extended_known_model_instance_library(bandit_lp)


def list_instances():
    """Return a dataframe describing the available named instances."""
    rows = extra_instance_metadata_dataframe(instance_library())
    return pd.DataFrame(rows).sort_values("instance").reset_index(drop=True)


def _run(
    *,
    bandit,
    initial_state,
    policy_name,
    alpha,
    N,
    horizon,
    seed,
    lp_update_horizon,
    cache_dir,
    force_recompute,
):
    result = cached_simulate_policy(
        bandit=bandit,
        policy_name=policy_name,
        make_policy=make_policy,
        initial_state=initial_state,
        N=N,
        time_horizon=horizon,
        seed=seed,
        alpha=alpha,
        lp_update_horizon=lp_update_horizon,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
        return_info=True,
    )
    mean_reward, x_values, reward_values, y_values, cache_info = result
    lp_bound, _ = cached_lp_upper_bound(
        bandit=bandit,
        alpha=alpha,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )
    gap = lp_bound - mean_reward
    relative_gap = np.nan if abs(lp_bound) <= 1e-12 else gap / abs(lp_bound)
    return {
        "policy": policy_name,
        "N": int(N),
        "alpha": float(alpha),
        "horizon": int(horizon),
        "seed": int(seed),
        "mean_reward": float(mean_reward),
        "lp_upper_bound": float(lp_bound),
        "relative_gap": float(relative_gap),
        "cache_hit": bool(cache_info["cache_hit"]),
        "runtime_seconds": float(cache_info["runtime_seconds"]),
        "x_values": x_values,
        "reward_values": reward_values,
        "y_values": y_values,
    }


def run_named_experiment(
    instance_name,
    policy_name,
    *,
    N=100,
    horizon=200,
    seed=123,
    alpha=None,
    lp_update_horizon=20,
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute=False,
):
    """Run one policy on one named homogeneous instance."""
    library = instance_library()
    if instance_name not in library:
        names = ", ".join(sorted(library))
        raise KeyError(f"Unknown instance {instance_name!r}. Available: {names}")
    spec = library[instance_name]
    if alpha is None:
        alpha = spec.default_alpha
    output = _run(
        bandit=spec.bandit,
        initial_state=spec.initial_state,
        policy_name=policy_name,
        alpha=alpha,
        N=N,
        horizon=horizon,
        seed=seed,
        lp_update_horizon=lp_update_horizon,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )
    output["instance"] = instance_name
    return output


def run_custom_experiment(
    P,
    R,
    policy_name,
    *,
    alpha,
    N=100,
    horizon=200,
    seed=123,
    initial_state=None,
    lp_update_horizon=20,
    cache_dir=DEFAULT_CACHE_DIR,
    force_recompute=False,
):
    """Run one policy on user-supplied ``P`` and ``R`` arrays."""
    bandit = bandit_lp.BanditInstance(np.asarray(P, float), np.asarray(R, float))
    if initial_state is None:
        initial_state = np.ones(bandit.S) / bandit.S
    output = _run(
        bandit=bandit,
        initial_state=np.asarray(initial_state, float),
        policy_name=policy_name,
        alpha=alpha,
        N=N,
        horizon=horizon,
        seed=seed,
        lp_update_horizon=lp_update_horizon,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )
    output["instance"] = "custom"
    return output
