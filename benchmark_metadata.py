"""Small shared checks and provenance records for uncached benchmarks."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from simulation_utils import SIMULATION_VERSION


ROOT = Path(__file__).resolve().parent


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def experiment_axes(n_values, policies, **counts):
    """Reject empty/duplicate axes before an experiment writes any results."""
    n_values, policies = list(n_values), list(policies)
    if not n_values or not policies:
        raise ValueError("Select at least one N and one policy.")
    n_values = [positive_integer(n, "N") for n in n_values]
    if len(set(n_values)) != len(n_values) or len(set(policies)) != len(policies):
        raise ValueError("N values and policies must not contain duplicates.")
    for name, value in counts.items():
        positive_integer(value, name)
    return n_values, policies


def write_experiment_settings(output_dir, *, source_files, **settings):
    """Record code and protocol separately from measured policy/simulation time."""
    metadata = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, simulation_version=SIMULATION_VERSION,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in source_files},
        **settings,
    )
    path = Path(output_dir) / "experiment_settings.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
