"""Run a quick end-to-end check after installing the project."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from benchmark_api import run_named_experiment


def main():
    """Run one small experiment twice and verify deterministic caching."""
    settings = {
        "instance_name": "random_S10_seed123",
        "policy_name": "Myopic",
        "N": 20,
        "horizon": 20,
        "seed": 123,
    }

    original_directory = Path.cwd()
    with TemporaryDirectory(prefix="rmab_smoke_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        os.chdir(temporary_path)
        try:
            first = run_named_experiment(
                **settings,
                cache_dir=temporary_path / "rmab_cache",
            )
            second = run_named_experiment(
                **settings,
                cache_dir=temporary_path / "rmab_cache",
            )
        finally:
            os.chdir(original_directory)

    assert np.isfinite(first["mean_reward"])
    assert not first["cache_hit"]
    assert second["cache_hit"]
    assert np.isclose(first["mean_reward"], second["mean_reward"])

    print("Smoke test passed.")
    print(f"Mean reward: {first['mean_reward']:.6f}")
    print("Cache behavior: miss -> hit")


if __name__ == "__main__":
    main()
