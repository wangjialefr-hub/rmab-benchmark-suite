"""Small, uncached verification of timing and heterogeneous experiment runners.

Run from Jupyter: %run "verify_extensions.py"
This is a functional check, not the full paper's performance experiment.
"""

import argparse
import json
from pathlib import Path

from heterogeneous_rmab import POLICY_CLASSES
from run_computation_cost_benchmark import run_computation_cost_benchmark, POLICY_NAMES
from run_heterogeneous_benchmark import run_heterogeneous_benchmark


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "verification_outputs" / "corrected_extensions_v2"


def main(output_dir=OUTPUT):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    # True cold Q-Whittle setup retrains 41 penalty tables. It is excluded here,
    # not replaced with a cached model or shortened training labeled as cold.
    cost = run_computation_cost_benchmark(
        instance_names=["random_S10_seed123", "deadline_S10_a20", "maintenance_S10_a20"],
        policy_names=[name for name in POLICY_NAMES if name != "QWhittleKnownModel"],
        n_values=[20, 100], num_repetitions=2, setup_repetitions=2,
        simulation_horizon=20, output_dir=folder / "timing", make_plots=True,
    )
    outcomes = dict(
        scope="small uncached functional verification, not publication results",
        timing_runs=len(cost["simulation_raw"]),
        timing_failures=int(cost["simulation_raw"]["status"].ne("ok").sum()),
        timing_setup_runs=len(cost["setup_raw"]),
        timing_setup_failures=int(cost["setup_raw"]["status"].ne("ok").sum()),
        qwhittle_timing="not evaluated; no shortened or cached cold training",
    )
    for name, types in (("five_types", 5), ("unique_models", None)):
        result = run_heterogeneous_benchmark(
            n_values=[10, 20], num_monte_carlo=2, simulation_horizon=10,
            num_arm_types=types, policy_names=list(POLICY_CLASSES),
            output_dir=folder / name, make_plots=True,
        )
        outcomes[name] = dict(
            runs=len(result["raw"]), failures=int(result["raw"]["status"].ne("ok").sum()),
            setup_failures=int(result["setup"]["status"].ne("ok").sum()),
            distinct_models=result["setup"][["instance", "N", "num_distinct_models"]]
                .drop_duplicates().to_dict("records"),
        )
    (folder / "verification_summary.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    report = (
        "# Timing and Heterogeneous Verification\n\n"
        "This is a small functional check, not a full performance study.\n\n"
        f"Timing: {outcomes['timing_runs']} fresh simulations, {outcomes['timing_failures']} failures; "
        f"{outcomes['timing_setup_runs']} cold policy constructions, {outcomes['timing_setup_failures']} failures.\n\n"
        "Timing protocol: three instances, eight policies, N=20/100, T=20, two repetitions; "
        "Q-Whittle cold training is excluded. Setup and simulation are measured separately. "
        "Simulation includes initialization, decisions, rewards and transitions, not file I/O.\n\n"
    )
    for name in ("five_types", "unique_models"):
        row = outcomes[name]
        report += f"{name}: {row['runs']} simulations, {row['failures']} failures, {row['setup_failures']} setup failures.\n\n"
    report += (
        "Heterogeneous protocol: maintenance and wireless, all seven policies (including LP-Update), "
        "N=10/20, T=10, two seeds. Setup is measured once per configuration; "
        "LP-Update clears its previous-trajectory solves on reset.\n\n"
        "Plots and raw CSVs are in timing/, five_types/ and unique_models/. "
        "Each directory records its protocol and code hashes in experiment_settings.json.\n\n"
        "Do not infer asymptotic complexity or a policy recommendation from two N values, "
        "short trajectories or a single heterogeneous setup measurement. All trajectories here "
        "are recomputed without accessing the existing reward cache. Historical results are untouched.\n"
    )
    (folder / "RESULTS.md").write_text(report, encoding="utf-8")
    print(f"\nExtension verification: {folder / 'RESULTS.md'}", flush=True)
    if (outcomes["timing_failures"] or outcomes["timing_setup_failures"]
            or any(outcomes[name]["failures"] or outcomes[name]["setup_failures"]
                   for name in ("five_types", "unique_models"))):
        raise RuntimeError("Some runs failed. Their errors are retained in the raw CSVs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.output_dir)
