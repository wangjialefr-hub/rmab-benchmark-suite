"""
Simulation cache utilities for RMAB benchmark experiments.

This file is independent from the teacher code. It avoids recomputing the same
simulation run when the combination

    bandit + policy + alpha + N + horizon + seed + initial_state

has already been evaluated.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np


CACHE_VERSION = "simulation-cache-v1"


def _hash_array(array):
    """Return a stable hash for a numpy-compatible array."""
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.view(np.uint8))
    return h.hexdigest()[:16]


def _policy_cache_payload(policy_name):
    """
    Add policy-specific parameters that affect simulation results.

    Most policies are fully identified by policy_name + bandit + alpha. Q-Whittle
    also depends on its Q-learning training configuration, so those values must
    be part of the cache key.
    """
    if str(policy_name) != "QWhittleKnownModel":
        return {}

    import make_policy

    return {
        "qwhittle_gamma": float(make_policy.QWHITTLE_GAMMA),
        "qwhittle_num_penalties": int(make_policy.QWHITTLE_NUM_PENALTIES),
        "qwhittle_steps_per_penalty": int(make_policy.QWHITTLE_STEPS_PER_PENALTY),
        "qwhittle_training_seed": int(make_policy.QWHITTLE_TRAINING_SEED),
    }


def _cache_key(
    *,
    bandit,
    policy_name,
    alpha,
    initial_state,
    N,
    time_horizon,
    seed,
    lp_update_horizon,
):
    """Create a stable cache key for one simulation setting."""
    payload = {
        "cache_version": CACHE_VERSION,
        "bandit_hash": bandit.hashname(),
        "policy_name": str(policy_name),
        "alpha": float(alpha),
        "initial_state_hash": _hash_array(initial_state),
        "N": str(N),
        "time_horizon": int(time_horizon),
        "seed": None if seed is None else int(seed),
        "lp_update_horizon": int(lp_update_horizon),
    }
    payload.update(_policy_cache_payload(policy_name))

    encoded = json.dumps(payload, sort_keys=True).encode()
    key = hashlib.sha256(encoded).hexdigest()[:24]
    return key, payload


def cached_lp_upper_bound(
    *,
    bandit,
    alpha,
    cache_dir="rmab_cache",
    force_recompute=False,
):
    """
    Cache bandit.relaxed_lp_average_reward(alpha)[0].
    """
    cache_dir = Path(cache_dir) / "lp_upper_bounds"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cache_version": CACHE_VERSION,
        "bandit_hash": bandit.hashname(),
        "alpha": float(alpha),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    cache_file = cache_dir / f"lp_bound_{key}.npz"

    if cache_file.exists() and not force_recompute:
        data = np.load(cache_file, allow_pickle=False)
        return float(data["lp_upper_bound"]), {
            "cache_hit": True,
            "cache_file": str(cache_file),
        }

    start = time.perf_counter()
    lp_upper_bound = float(bandit.relaxed_lp_average_reward(alpha)[0])
    runtime_seconds = time.perf_counter() - start

    np.savez_compressed(
        cache_file,
        lp_upper_bound=np.array(lp_upper_bound),
        runtime_seconds=np.array(runtime_seconds),
        metadata=np.array(json.dumps(payload, sort_keys=True)),
    )

    return lp_upper_bound, {
        "cache_hit": False,
        "cache_file": str(cache_file),
        "runtime_seconds": runtime_seconds,
    }


def cached_simulate_policy(
    *,
    bandit,
    policy_name,
    make_policy,
    initial_state,
    N,
    time_horizon,
    seed,
    alpha,
    lp_update_horizon=20,
    cache_dir="rmab_cache",
    force_recompute=False,
    verbose=False,
    return_info=False,
):
    """
    Run strategies.simulate with caching.

    Parameters
    ----------
    make_policy:
        A factory function with signature:

            make_policy(policy_name, bandit, alpha, lp_update_horizon=...)

        The make_policy.py file created for this project has this signature.

    return_info:
        If False, return the same four values as strategies.simulate:

            mean_reward, x_values, reward_values, y_values

        If True, additionally return a cache_info dictionary.
    """
    import strategies

    cache_dir = Path(cache_dir) / "simulations"
    cache_dir.mkdir(parents=True, exist_ok=True)

    key, payload = _cache_key(
        bandit=bandit,
        policy_name=policy_name,
        alpha=alpha,
        initial_state=initial_state,
        N=N,
        time_horizon=time_horizon,
        seed=seed,
        lp_update_horizon=lp_update_horizon,
    )
    cache_file = cache_dir / f"sim_{key}.npz"

    if cache_file.exists() and not force_recompute:
        if verbose:
            print(f"[cache hit] {policy_name}, N={N}, seed={seed}")
        data = np.load(cache_file, allow_pickle=False)
        result = (
            float(data["mean_reward"]),
            data["x_values"],
            data["reward_values"],
            data["y_values"],
        )
        info = {
            "cache_hit": True,
            "cache_file": str(cache_file),
            "runtime_seconds": float(data["runtime_seconds"]),
        }
        return (*result, info) if return_info else result

    if verbose:
        reason = "force recompute" if force_recompute else "cache miss"
        print(f"[{reason}] {policy_name}, N={N}, seed={seed}")

    policy = make_policy(
        policy_name=policy_name,
        bandit=bandit,
        alpha=alpha,
        lp_update_horizon=lp_update_horizon,
    )

    # The reference simulator writes its own cache to this relative folder.
    # A fresh clone does not contain generated directories, so create it here.
    Path("computed_values").mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    mean_reward, x_values, reward_values, y_values = strategies.simulate(
        bandit=bandit,
        strategy=policy,
        initial_state=initial_state,
        N=N,
        time=time_horizon,
        seed=seed,
    )
    runtime_seconds = time.perf_counter() - start

    np.savez_compressed(
        cache_file,
        mean_reward=np.array(float(mean_reward)),
        x_values=x_values,
        reward_values=reward_values,
        y_values=y_values,
        runtime_seconds=np.array(runtime_seconds),
        metadata=np.array(json.dumps(payload, sort_keys=True)),
    )

    result = (
        float(mean_reward),
        x_values,
        reward_values,
        y_values,
    )
    info = {
        "cache_hit": False,
        "cache_file": str(cache_file),
        "runtime_seconds": runtime_seconds,
    }
    return (*result, info) if return_info else result
