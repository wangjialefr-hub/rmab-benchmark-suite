"""Reproduce the simulator audit without overwriting historical experiments.

Jupyter: %run "verify_corrections.py"
Short check: %run "verify_corrections.py" --quick
"""

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from benchmark_api import instance_library
from make_policy import (
    POLICY_NAMES, make_policy, QWHITTLE_GAMMA, QWHITTLE_NUM_PENALTIES,
    QWHITTLE_STEPS_PER_PENALTY, QWHITTLE_TRAINING_SEED,
)
from run_instance_matrix_benchmark import run_benchmark
from simulation_cache import cached_finite_horizon_bound, simulation_code_hash
from simulation_utils import SIMULATION_VERSION
import strategies


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "verification_outputs" / "corrected_v2"


def deadline_reference_check(folder):
    """Deterministic example: separate startup bias from a simulator defect."""
    spec = instance_library()["deadline_S10_a20"]
    stationary = spec.bandit.relaxed_lp_average_reward(spec.default_alpha)[0]
    rows = []
    for horizon in (20, 50, 100, 200, 500):
        finite = cached_finite_horizon_bound(
            bandit=spec.bandit, alpha=spec.default_alpha, initial_state=spec.initial_state,
            time_horizon=horizon, cache_dir=folder / "cache",
        )
        mean, x, rewards, y = strategies.simulate(
            spec.bandit, make_policy("Myopic", spec.bandit, spec.default_alpha),
            spec.initial_state, 500, horizon, seed=123, use_cache=False,
        )
        residual = float(np.max(np.abs(x[1:] - np.einsum("tsa,sak->tk", y[:-1], spec.bandit.P))))
        if residual > 1e-10 or abs(mean - finite) > 1e-8:
            raise AssertionError("Deterministic deadline trajectory does not match its finite LP.")
        rows.append(dict(horizon=horizon, stationary_lp=stationary, finite_horizon_lp=finite,
                         observed_reward=mean, stationary_relative_gap=(stationary - mean) / abs(stationary),
                         finite_horizon_relative_gap=(finite - mean) / abs(finite),
                         reward_after_step_20=float(np.mean(rewards[20:])) if horizon > 20 else np.nan,
                         max_transition_residual=residual))
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "deadline_reference_check.csv", index=False)
    return frame


def main(quick=False, output_dir=None, with_qwhittle=False):
    # Keep a quick check from overwriting the longer verification evidence.
    default = OUTPUT.with_name("corrected_v2_quick") if quick else OUTPUT
    if with_qwhittle:
        default = default.with_name(default.name + "_with_qwhittle")
    folder = Path(output_dir) if output_dir is not None else default
    folder.mkdir(parents=True, exist_ok=True)
    cache_dir = folder / "cache" if output_dir is not None else OUTPUT / "cache"
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with contextlib.redirect_stdout(log):
        test_result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (folder / "unit_tests.txt").write_text(log.getvalue(), encoding="utf-8")
    if not test_result.wasSuccessful():
        raise RuntimeError(f"Regression tests failed. Read {folder / 'unit_tests.txt'}")
    print(f"Passed {test_result.testsRun} regression tests.", flush=True)

    # Q-Whittle training is deliberately not shortened or silently substituted.
    policies = [p for p in POLICY_NAMES if p != "QWhittleKnownModel"]
    smoke = run_benchmark(
        policy_names=policies, n_values=[20], num_monte_carlo=1, simulation_horizon=30,
        include_finite_horizon_bound=False, make_plots=False,
        output_dir=folder / "library_smoke", cache_dir=cache_dir,
    )
    targeted_settings = dict(
        instance_names=["deadline_S10_a20", "maintenance_S10_a20", "random_S10_seed123"],
        policy_names=POLICY_NAMES if with_qwhittle else policies,
        n_values=[20, 100] if quick else [20, 100, 500],
        num_monte_carlo=2 if quick else 3, simulation_horizon=60 if quick else 200,
        cache_dir=cache_dir,
    )
    checks = run_benchmark(**targeted_settings, output_dir=folder / "targeted")
    # A replay must use every cached trajectory and reproduce every reward.
    with contextlib.redirect_stdout(io.StringIO()):
        replay = run_benchmark(**targeted_settings, output_dir=folder / "cache_replay", make_plots=False)
    if not replay["raw"]["cache_hit"].all():
        raise AssertionError("The verification replay did not reuse every cached trajectory.")
    columns = ["instance", "policy", "N", "replication", "seed", "simulation_mean_reward"]
    pd.testing.assert_frame_equal(checks["raw"][columns], replay["raw"][columns], check_exact=True)
    print(f"Verified {len(replay['raw'])} cache hits with identical rewards.", flush=True)
    evidence = deadline_reference_check(folder)
    old_path = ROOT / "instance_matrix_outputs" / "summary_instance_matrix.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        keys = ["instance", "policy", "N"]
        comparison = checks["summary"].merge(
            old[keys + ["mean_reward", "num_runs"]], on=keys, how="left", suffixes=("_corrected", "_historical"),
        )
        comparison.to_csv(folder / "historical_comparison.csv", index=False)

    failed_smoke = smoke["raw"][smoke["raw"]["status"] != "ok"]
    failed_targeted = checks["raw"][checks["raw"]["status"] != "ok"]
    metadata = dict(completed_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__, simulation_version=SIMULATION_VERSION,
                    code_hash=simulation_code_hash(), quick=quick, tests_passed=test_result.testsRun,
                    smoke_runs=len(smoke["raw"]), smoke_failures=len(failed_smoke),
                    targeted_runs=len(checks["raw"]), targeted_failures=len(failed_targeted),
                    replay_cache_hits=int(replay["raw"]["cache_hit"].sum()),
                    excluded_policy=None if with_qwhittle else "QWhittleKnownModel (not evaluated)",
                    qwhittle_scope="three targeted instances; one fixed training seed" if with_qwhittle else "not evaluated")
    if with_qwhittle:
        metadata["qwhittle_training"] = dict(
            gamma=QWHITTLE_GAMMA, num_penalties=QWHITTLE_NUM_PENALTIES,
            steps_per_penalty=QWHITTLE_STEPS_PER_PENALTY, training_seed=QWHITTLE_TRAINING_SEED,
        )
    (folder / "verification_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ftva = checks["summary"][checks["summary"]["policy"] == "FTVA_Strategy"]
    report = (
        "# Verification Results\n\n"
        f"Completed UTC: {metadata['completed_utc']}\n\n"
        f"Code fingerprint: {metadata['code_hash']}\n\n"
        f"Regression tests: {test_result.testsRun} passed.\n\n"
        f"Library smoke run: {len(smoke['raw'])} simulations, {len(failed_smoke)} failures.\n\n"
        f"Targeted run: {len(checks['raw'])} simulations, {len(failed_targeted)} failures.\n\n"
        f"Cache replay: {metadata['replay_cache_hits']} hits; every per-seed reward is identical.\n\n"
        "## Deadline Reference Check\n\n```text\n"
        + evidence.to_string(index=False) + "\n```\n\n"
        "The finite LP and observed deterministic reward agree. The negative stationary difference\n"
        "reflects startup rewards; it is not an upper-bound violation. Burn-in length 20 resolves\n"
        "this particular Myopic example, not every policy or instance.\n\n"
        "## FTVA Results\n\n```text\n"
        + ftva[["instance", "N", "mean_reward", "mean_relative_gap", "num_runs"]].to_string(index=False)
        + "\n```\n\n"
        "Passing these checks establishes valid allocations and runnable policies, not an\n"
        "asymptotic-optimality theorem. The full 20-seed paper matrix was not rerun here.\n"
        f"Q-Whittle scope: {metadata['qwhittle_scope']}. Historical comparisons may use different horizons and\n"
        "replication counts; they are a diagnostic, not a paired statistical comparison.\n"
    )
    if with_qwhittle:
        learned = checks["summary"][checks["summary"]["policy"] == "QWhittleKnownModel"]
        report += (
            "\n## Q-Whittle Check\n\n```text\n"
            + learned[["instance", "N", "mean_reward", "mean_relative_gap", "num_runs"]].to_string(index=False)
            + f"\n```\n\nTraining configuration: {metadata['qwhittle_training']}. "
            "The three evaluation seeds do not measure variability across independent training runs.\n"
        )
    (folder / "RESULTS.md").write_text(report, encoding="utf-8")
    print(f"\nVerification results: {folder / 'RESULTS.md'}", flush=True)
    if len(failed_smoke) or len(failed_targeted):
        raise RuntimeError("Some benchmark runs failed; inspect the saved policy_run_status.csv files.")
    # These summaries read only this verified subset and label its limited scope.
    from generate_benchmark_findings import generate_benchmark_findings
    from generate_paper_figures import generate_paper_figures
    from check_paper_readiness import check_paper_readiness
    preview = folder / "report_preview"
    generate_benchmark_findings(preview, known_model_dir=folder / "targeted")
    generate_paper_figures(preview, known_model_dir=folder / "targeted")
    check_paper_readiness(preview, known_model_dir=folder / "targeted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--with-qwhittle", action="store_true", help="Also evaluate full-training Q-Whittle on the three targeted instances.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    main(quick=args.quick, output_dir=args.output_dir, with_qwhittle=args.with_qwhittle)
