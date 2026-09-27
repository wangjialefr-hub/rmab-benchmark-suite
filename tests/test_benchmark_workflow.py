"""Check public API, reporting, and the heterogeneous FTVA completion."""

import contextlib
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bandit_lp
import strategies
from benchmark_api import run_named_experiment
from heterogeneous_rmab import (
    HeterogeneousFTVAPolicy, build_heterogeneous_maintenance, simulate_heterogeneous,
    solve_heterogeneous_average_lp,
)
from known_model_extra_instances import deadline_scheduling_instance
from make_policy import make_policy
from run_computation_cost_benchmark import simulate_uncached
from run_instance_matrix_benchmark import calc_beta_one_group, plot_instance_metric, run_benchmark


class WorkflowTests(unittest.TestCase):
    def test_api_burn_in_and_finite_reference_have_distinct_targets(self):
        with TemporaryDirectory() as cache:
            result = run_named_experiment(
                "deadline_S10_a20", "Myopic", N=20, horizon=40, burn_in=20,
                cache_dir=cache, include_finite_horizon_bound=True,
            )
        self.assertEqual(result["total_simulated_steps"], 60)
        self.assertEqual(len(result["reward_values"]), 60)
        self.assertAlmostEqual(result["mean_reward"], 0.09)
        self.assertGreater(result["full_horizon_mean_reward"], 0.09)
        self.assertAlmostEqual(result["finite_horizon_relative_gap"], 0.0)

    def test_api_noninteger_budget_and_no_seed(self):
        with TemporaryDirectory() as cache:
            settings = dict(instance_name="deadline_S10_a20", policy_name="Myopic",
                            N=23, horizon=10, seed=None, cache_dir=cache)
            first = run_named_experiment(**settings)
            second = run_named_experiment(**settings)
            self.assertEqual(list((Path(cache) / "simulations").glob("*.npz")), [])
        self.assertEqual(first["active_budget"], 4)
        self.assertAlmostEqual(first["alpha"], 4 / 23)
        np.testing.assert_allclose(first["y_values"][:, :, 1].sum(axis=1), 4 / 23)
        self.assertFalse(first["cache_hit"])
        self.assertFalse(second["cache_hit"])

    def test_cost_loop_agrees_with_reward_loop(self):
        spec = deadline_scheduling_instance(bandit_lp, state_count=10, alpha=0.2)
        for name in ("Myopic", "LPRandomized", "FTVA", "RoundRobin", "LPUpdate"):
            with self.subTest(policy=name):
                policy = make_policy(name, spec.bandit, 0.2, lp_update_horizon=5)
                mean, *_ = strategies.simulate(spec.bandit, policy, spec.initial_state,
                                               23, 30, seed=123, use_cache=False)
                cost_mean = simulate_uncached(spec.bandit, policy, spec.initial_state, 23, 30, 123)
                self.assertEqual(mean, cost_mean)

    def test_runner_records_failures_and_reuses_successful_runs(self):
        with TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            root = Path(root)
            settings = dict(instance_names=["deadline_S10_a20"], policy_names=["Myopic", "unknown_policy"],
                            n_values=[20, 50], num_monte_carlo=2, simulation_horizon=15,
                            cache_dir=root / "cache", output_dir=root / "results", make_plots=False)
            first = run_benchmark(**settings)
            second = run_benchmark(**settings)
            self.assertEqual(len(first["raw"]), 8)
            self.assertEqual((first["raw"]["status"] == "failed").sum(), 4)
            self.assertEqual(second["summary"]["cache_hits"].sum(), 4)
            self.assertTrue((root / "results" / "policy_run_status.csv").exists())
            self.assertTrue((root / "results" / "experiment_settings.json").exists())

    def test_beta_excludes_nonpositive_and_uncertain_gaps(self):
        group = pd.DataFrame({"N": [10, 20, 40, 80, 160],
                              "mean_relative_gap": [1.0, 0.5, 0.25, -0.01, 1e-5],
                              "std_relative_gap": [0.0, 0.0, 0.0, 0.0, 0.1], "num_runs": 20})
        result = calc_beta_one_group(group)
        self.assertEqual(result["num_fit_points"], 3)
        self.assertEqual(result["num_excluded_points"], 2)
        self.assertAlmostEqual(result["convergence_beta"], 1.0)
        group.loc[2, "mean_relative_gap"] = 0
        self.assertTrue(np.isnan(calc_beta_one_group(group)["convergence_beta"]))

    def test_plot_preserves_negative_values(self):
        frame = pd.DataFrame({"instance": ["example"] * 3, "policy": ["Myopic"] * 3,
                              "N": [20, 100, 500], "mean_relative_gap": [0.1, 0, -0.01]})
        with TemporaryDirectory() as folder, patch("matplotlib.pyplot.close"):
            path = plot_instance_metric("example", frame, "mean_relative_gap", "Signed gap", True,
                                        "signed_gap", output_dir=folder)
            fig = plt.gcf()
            ax = fig.axes[0]
            self.assertEqual(ax.get_yscale(), "symlog")
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.1, 0, -0.01])
            self.assertGreater(path.stat().st_size, 1000)
        plt.close(fig)


class HeterogeneousTests(unittest.TestCase):
    def test_ftva_and_lp_use_actual_budget(self):
        environment = build_heterogeneous_maintenance(bandit_lp, N=23, state_count=10, num_types=3)
        solution = solve_heterogeneous_average_lp(environment)
        mass = sum(g["weight"] * y[:, 1].sum()
                   for g, y in zip(solution["groups"], solution["y_by_group"]))
        self.assertAlmostEqual(mass, 4 / 23, places=6)
        policy = HeterogeneousFTVAPolicy(environment)
        self.assertTrue(any(np.any(x == 0) for x in policy.x_by_group))
        for probabilities, x in zip(policy.pi_by_group, policy.x_by_group):
            np.testing.assert_allclose(probabilities[x == 0], 0.5)
        result = simulate_heterogeneous(environment, policy, horizon=30, seed=123)
        self.assertTrue(np.isfinite(result["mean_reward"]))
        repeat = simulate_heterogeneous(environment, policy, horizon=30, seed=123)
        np.testing.assert_array_equal(result["reward_history"], repeat["reward_history"])


if __name__ == "__main__":
    unittest.main()
