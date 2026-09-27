"""Read one corrected experiment without silently falling back to old results."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_policy import POLICY_NAMES
from paper_config import KNOWN_MODEL_BENCHMARK_INSTANCES, filter_paper_instances
from run_instance_matrix_benchmark import estimate_beta
from simulation_cache import simulation_code_hash
from simulation_utils import SIMULATION_VERSION


ROOT = Path(__file__).resolve().parent
CORRECTED_DATA = ROOT / "instance_matrix_outputs" / "corrected_v2"
CORRECTED_REPORT = ROOT / "paper_summary_outputs" / "corrected_v2"


def load_corrected_results(folder=CORRECTED_DATA):
    """Validate provenance and summaries; retain only complete replicate groups.

    Small validation sweeps are allowed, but explicitly labeled preliminary.
    Missing raw rows indicate an interrupted run, not an absent algorithm.
    """
    folder = Path(folder).resolve()
    files = {name: folder / name for name in (
        "experiment_settings.json", "raw_instance_matrix_results.csv", "summary_instance_matrix.csv",
    )}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing corrected result: {path}. Select a completed run; no historical fallback is used.")
    settings = json.loads(files["experiment_settings.json"].read_text(encoding="utf-8"))
    if (settings.get("simulation_version") != SIMULATION_VERSION
            or settings.get("code_hash") != simulation_code_hash()):
        raise ValueError("Results do not match the current simulator. Rerun the benchmark before reporting them.")
    raw = pd.read_csv(files["raw_instance_matrix_results.csv"])
    summary = pd.read_csv(files["summary_instance_matrix.csv"])
    keys = ["instance", "policy", "N"]
    run_keys = keys + ["replication"]
    for frame in (raw, summary):
        for column in ("simulation_version", "code_hash"):
            if column not in frame or not frame[column].eq(settings[column]).all():
                raise ValueError(f"Mixed or missing {column} in {folder}.")
    expected = pd.MultiIndex.from_product(
        [settings["instances"], settings["policies"], settings["N"], range(settings["num_monte_carlo"])],
        names=run_keys,
    )
    actual = pd.MultiIndex.from_frame(raw[run_keys])
    if actual.has_duplicates or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError("Raw results have duplicate, missing, or unexpected experiment keys.")
    if not raw["status"].isin(["ok", "failed"]).all():
        raise ValueError("Unrecognized simulation status.")
    if not raw["seed"].eq(settings["mc_seed"] + raw["replication"]).all():
        raise ValueError("Raw seeds do not match the declared replication protocol.")
    for column in ("horizon", "burn_in"):
        if not raw[column].eq(settings[column]).all():
            raise ValueError(f"Raw {column} does not match the experiment settings.")
    successful = raw[raw["status"] == "ok"]
    if successful.empty:
        raise ValueError("No successful simulations are available.")
    if not np.isfinite(successful[["simulation_mean_reward", "lp_upper_bound"]]).all().all():
        raise ValueError("Successful rows must contain finite rewards and LP references.")
    lp = successful["lp_upper_bound"].to_numpy(dtype=float)
    reward = successful["simulation_mean_reward"].to_numpy()
    gap = np.divide(lp - reward, np.abs(lp), out=np.full_like(lp, np.nan), where=np.abs(lp) > 1e-12)
    if not np.allclose(successful["relative_gap"], gap, rtol=1e-10, atol=1e-12, equal_nan=True):
        raise ValueError("Stored gaps do not equal the signed difference to the stationary LP.")
    recomputed = successful.groupby(keys).agg(
        mean_reward=("simulation_mean_reward", "mean"),
        std_reward=("simulation_mean_reward", "std"),
        mean_relative_gap=("relative_gap", "mean"), std_relative_gap=("relative_gap", "std"),
        num_runs=("status", "size"),
    ).sort_index()
    stored = summary.set_index(keys).sort_index()
    if stored.index.has_duplicates or not stored.index.equals(recomputed.index):
        raise ValueError("Summary groups do not match successful raw runs.")
    for column in recomputed:
        if not np.allclose(stored[column], recomputed[column], rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Summary {column} does not match the raw results.")
    complete = summary["num_runs"] == settings["num_monte_carlo"]
    excluded = summary.loc[~complete, keys + ["num_runs"]].to_dict("records")
    summary = filter_paper_instances(summary[complete].copy())
    if summary.empty:
        raise ValueError("No complete replicate groups are available for comparison.")
    if not summary["N"].eq(max(settings["N"])).any():
        raise ValueError("No complete group at the largest planned N; do not report a smaller N as the final result.")
    summary.attrs["policies"] = settings["policies"]
    beta = estimate_beta(summary)
    paper_grid_complete = (
        set(settings["instances"]) == set(KNOWN_MODEL_BENCHMARK_INSTANCES)
        and set(settings["policies"]) == set(POLICY_NAMES)
        and set(settings["N"]) == {20, 50, 100, 200, 500}
        and settings["num_monte_carlo"] >= 20 and raw["status"].eq("ok").all()
    )
    manifest = {
        "source_directory": str(folder), "settings": settings,
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        "report_source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "report_data.py", "generate_paper_figures.py", "generate_benchmark_findings.py",
            "run_instance_matrix_benchmark.py", "check_paper_readiness.py",
        )},
        "attempted_runs": len(raw), "failed_runs": int(raw["status"].eq("failed").sum()),
        "excluded_incomplete_groups": excluded, "paper_grid_complete": bool(paper_grid_complete),
        "report_scope": "complete default grid" if paper_grid_complete else "preliminary subset",
        "extensions": "No historical timing, heterogeneous, or learning results were loaded.",
    }
    return summary, beta, manifest


def write_manifest(output_dir, manifest):
    path = Path(output_dir) / "report_inputs.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def complete_reward_ranks(summary, N):
    """Rank on shared instances only; missing policies never gain a rank advantage."""
    frame = summary[summary["N"] == N]
    values = frame.pivot(index="instance", columns="policy", values="mean_reward")
    policies = summary.attrs.get("policies", sorted(summary["policy"].unique()))
    values = values.reindex(columns=policies).replace([np.inf, -np.inf], np.nan)
    return values.dropna(how="any").rank(axis=1, ascending=False, method="average")
