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
from simulation_cache import cached_finite_horizon_bound, cached_lp_upper_bound, cached_simulate_policy
from simulation_utils import integer_budget, state_counts


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
    burn_in=0,
    include_finite_horizon_bound=False,
):
    configured_alpha = float(alpha)
    alpha = integer_budget(alpha, N) / N
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
        burn_in=burn_in,
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
    output = {
        "policy": policy_name,
        "N": int(N),
        "alpha": float(alpha),
        "configured_alpha": configured_alpha,
        "active_budget": integer_budget(alpha, N),
        "horizon": int(horizon),
        "seed": None if seed is None else int(seed),
        "burn_in": int(burn_in),
        "total_simulated_steps": int(horizon + burn_in),
        "mean_reward": float(mean_reward),
        "full_horizon_mean_reward": float(np.mean(reward_values)),
        "lp_upper_bound": float(lp_bound),
        "relative_gap": float(relative_gap),
        "lp_reference": "stationary_average_reward",
        "simulation_version": cache_info["simulation_version"],
        "code_hash": cache_info["code_hash"],
        "cache_hit": bool(cache_info["cache_hit"]),
        "runtime_seconds": float(cache_info["runtime_seconds"]),
        "x_values": x_values,
        "reward_values": reward_values,
        "y_values": y_values,
    }
    if include_finite_horizon_bound:
        bound = cached_finite_horizon_bound(
            bandit=bandit, alpha=alpha, initial_state=state_counts(initial_state, N) / N,
            time_horizon=horizon + burn_in, cache_dir=cache_dir,
            force_recompute=force_recompute,
        )
        output["finite_horizon_lp_bound"] = bound
        full_mean = output["full_horizon_mean_reward"]
        output["finite_horizon_relative_gap"] = (
            np.nan if abs(bound) <= 1e-12 else (bound - full_mean) / abs(bound)
        )
    return output


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
    burn_in=0,
    include_finite_horizon_bound=False,
):
    """Run one policy on one named homogeneous instance.

    horizon counts measured steps; burn_in adds unscored startup steps. Returned
    arrays contain both. relative_gap is signed against the stationary LP.
    The optional finite-horizon comparison uses ALL steps, including burn-in.
    Noninteger alpha*N is replaced by floor(alpha*N), including in LP policies.
    """
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
        burn_in=burn_in,
        include_finite_horizon_bound=include_finite_horizon_bound,
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
    burn_in=0,
    include_finite_horizon_bound=False,
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
        burn_in=burn_in,
        include_finite_horizon_bound=include_finite_horizon_bound,
    )
    output["instance"] = "custom"
    return output
