"""Run a short, uncached check of online learners and known-model references.

Jupyter: %run "verify_online_learning.py"
This checks execution, observability and reporting, not learning convergence.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from run_unknown_model_benchmark import run_unknown_model_benchmark, POLICY_NAMES
from run_computation_cost_suite import collect_existing_computation_cost_outputs


OUTPUT = Path(__file__).resolve().parent / "verification_outputs" / "corrected_online_v2"


def main(output_dir=OUTPUT):
    folder = Path(output_dir)
    result = run_unknown_model_benchmark(
        instance_names=["random_S10_seed123", "maintenance_S10_a40",
                        "wireless_channel_S10_a40", "deadline_S10_a20"],
        policy_names=POLICY_NAMES, n_values=[20, 50], num_monte_carlo=2,
        simulation_horizon=200, curve_sample_every=10, output_dir=folder,
        tail_fraction=0.25, make_plots=True,
    )
    raw, summary = result["raw"], result["summary"]
    successful = raw[raw["status"].eq("ok")]
    if len(raw) != 112:
        raise AssertionError("Missing rows in the planned online verification matrix.")
    if not np.isfinite(successful[["mean_reward", "tail_mean_reward"]]).all().all():
        raise AssertionError("A successful online run has a nonfinite reward.")
    np.testing.assert_allclose(successful["runtime_seconds"],
                               successful["setup_seconds"] + successful["simulation_seconds"])
    failures = int(raw["status"].ne("ok").sum())
    plugin = successful[successful["policy"].eq("OnlinePlugInWhittle")]
    metadata = dict(
        scope="short online-learning verification, not a convergence or performance claim",
        runs=len(raw), failures=failures,
        complete_groups=int(summary["complete"].sum()),
        plugin_recomputations=int(plugin["num_index_recomputations"].sum()),
        plugin_solver_failures=int(plugin["num_index_failures"].sum()),
        plugin_fallback_decisions=int(plugin["num_fallback_decisions"].sum()),
    )
    (folder / "verification_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fallback_table = plugin.groupby("instance").agg(
        runs=("status", "size"), recomputations=("num_index_recomputations", "sum"),
        index_failures=("num_index_failures", "sum"), fallback_decisions=("num_fallback_decisions", "sum"),
    )
    report = (
        "# Online-Learning Verification\n\n"
        f"Completed simulations: {len(raw)}; failed: {failures}.\n\n"
        "Protocol: four instances, five learners and two known-model references; N=20/50, "
        "T=200, two independent evaluation seeds. Each seed starts a new learner. "
        "Tail reward uses the last 50 steps. Learning curves show mean reward at "
        "sampled steps, not cumulative reward, smoothed reward or regret.\n\n"
        "The learner observes all arms' states, deterministic R[s,a] rewards and "
        "transitions, including passive arms. Homogeneous arms share learned tables, "
        "so each period supplies N samples. This is not a partially observed benchmark.\n\n"
        "## Plug-in Whittle Diagnostics\n\n"
        + fallback_table.to_markdown() + "\n\n"
        "A failed estimated-model index computation uses reward-advantage fallback; "
        "these events are reported rather than interpreted as pure Whittle decisions. "
        "Index recomputation counts include initialization and the update after the last step.\n\n"
        "## Interpretation\n\n"
        "Plug-in Whittle is model-based online learning. OnlineQLearningIndex is a "
        "discounted Q-difference ranking heuristic, not a Whittle-index estimator. "
        "KnownWhittleOracle and KnownLPPriorityOracle use the true model and remain "
        "labeled as references in timing summaries.\n\n"
        "Setup and simulation timers are separate. Simulation includes reset, environment "
        "transitions and learning; it is not pure decision time. No reward caches were used. "
        "Short horizons and two seeds are inadequate for a final policy recommendation.\n"
    )
    (folder / "RESULTS.md").write_text(report, encoding="utf-8")
    collect_existing_computation_cost_outputs(
        folder / "cost_summary", known_dir=None, heterogeneous_dir=None, unknown_dir=folder,
    )
    print(f"\nOnline verification: {folder / 'RESULTS.md'}", flush=True)
    if failures:
        raise RuntimeError("Some online simulations failed; inspect the raw error and failure_phase columns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.output_dir)
