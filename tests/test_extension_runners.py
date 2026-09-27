"""Regression checks for uncached timing and heterogeneous experiment runners."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bandit_lp
import heterogeneous_rmab as hr
import run_computation_cost_benchmark as cost
import run_heterogeneous_benchmark as hetero
import run_computation_cost_suite as suite
from known_model_extra_instances import deadline_scheduling_instance


class ExtensionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.spec = deadline_scheduling_instance(bandit_lp, state_count=10, alpha=0.2)

    def test_cost_preserves_all_setup_failures(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), patch.object(
            cost, "build_extended_known_model_instance_library", return_value={"deadline": self.spec}
        ):
            result = cost.run_computation_cost_benchmark(
                instance_names=["deadline"], policy_names=["not_a_policy"], n_values=[20, 50],
                num_repetitions=2, setup_repetitions=1, simulation_horizon=5,
                output_dir=folder, make_plots=False,
            )
            self.assertEqual(len(result["simulation_raw"]), 4)
            self.assertTrue(result["simulation_raw"]["status"].eq("failed").all())
            self.assertTrue(result["cost_summary"].empty)
            self.assertTrue(result["runtime_scaling"].empty)
            self.assertTrue((Path(folder) / "simulation_times_raw.csv").is_file())
            self.assertTrue((Path(folder) / "experiment_settings.json").is_file())

    def test_cost_setup_matches_actual_integer_budget(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), patch.object(
            cost, "build_extended_known_model_instance_library", return_value={"deadline": self.spec}
        ):
            result = cost.run_computation_cost_benchmark(
                instance_names=["deadline"], policy_names=["Myopic"], n_values=[20, 23, 40],
                num_repetitions=1, setup_repetitions=1, simulation_horizon=8,
                output_dir=folder, make_plots=False,
            )
            self.assertEqual(len(result["setup_raw"]), 2)
            self.assertTrue(result["simulation_raw"]["status"].eq("ok").all())
            by_n = result["cost_summary"].set_index("N")
            self.assertAlmostEqual(by_n.loc[23, "alpha"], 4 / 23)
            self.assertEqual(set(result["runtime_scaling"]["num_fit_points"]), {1, 2})
            metadata = json.loads((Path(folder) / "experiment_settings.json").read_text())
            self.assertFalse(metadata["simulation_cache"])
            self.assertIn("simulation_utils.py", metadata["source_sha256"])
            combined = suite.collect_existing_computation_cost_outputs(
                Path(folder) / "combined", known_dir=folder, heterogeneous_dir=None,
            )
            self.assertEqual(len(combined["combined"]), 3)
            metadata["source_sha256"]["simulation_utils.py"] = "incorrect"
            (Path(folder) / "experiment_settings.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "different code"):
                suite.read_corrected_cost(Path(folder) / "computation_cost_summary.csv")

    def test_heterogeneous_preserves_all_setup_failures(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            result = hetero.run_heterogeneous_benchmark(
                n_values=[10], policy_names=["not_a_policy"], num_monte_carlo=2,
                simulation_horizon=5, num_arm_types=2,
                instance_builders={"maintenance": hr.build_heterogeneous_maintenance},
                output_dir=folder, make_plots=False,
            )
            self.assertEqual(len(result["raw"]), 2)
            self.assertTrue(result["raw"]["status"].eq("failed").all())
            self.assertTrue(result["summary"].empty)
            self.assertEqual(len(pd.read_csv(Path(folder) / "heterogeneous_raw_results.csv")), 2)

    def test_runner_rejects_empty_or_invalid_protocol(self):
        for runner in (cost.run_computation_cost_benchmark, hetero.run_heterogeneous_benchmark):
            for settings in (dict(n_values=[]), dict(n_values=[20, 20]), dict(n_values=[0]),
                             dict(n_values=[1.5]), dict(policy_names=[]), dict(simulation_horizon=0)):
                with self.subTest(runner=runner.__name__, settings=settings), self.assertRaises(ValueError):
                    runner(**settings)

    def test_lpupdate_reset_does_not_reuse_previous_trajectory_solves(self):
        environment = hr.build_heterogeneous_maintenance(bandit_lp, N=10, state_count=5, num_types=2)
        policy = hr.HeterogeneousLPUpdatePolicy(environment, time_horizon=3)
        states = np.zeros(environment.N, dtype=int)
        with patch.object(hr, "solve_heterogeneous_finite_lp", wraps=hr.solve_heterogeneous_finite_lp) as solve:
            policy.reset(123)
            first = policy.select_actions(states)
            policy.select_actions(states)
            self.assertEqual(solve.call_count, 1)
            policy.reset(123)
            repeat = policy.select_actions(states)
            self.assertEqual(solve.call_count, 2)
        np.testing.assert_array_equal(first, repeat)

    def test_simulator_rejects_nonbinary_actions_before_integer_cast(self):
        environment = hr.build_heterogeneous_maintenance(bandit_lp, N=10, num_types=2)
        for value in (0.5, -1, 2, np.nan):
            actions = np.zeros(10)
            actions[0] = value
            policy = SimpleNamespace(environment=environment, reset=lambda seed: None,
                                     select_actions=lambda states: actions)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "binary"):
                hr.simulate_heterogeneous(environment, policy, horizon=2, seed=123)

    def test_custom_heterogeneous_n_ticks(self):
        frame = pd.DataFrame(dict(instance=["test"] * 2, policy=["RoundRobin"] * 2,
                                  N=[13, 27], mean_reward=[1.0, 1.1]))
        with TemporaryDirectory() as folder, patch("matplotlib.pyplot.close"):
            hetero.plot_metric(frame, "test", "mean_reward", "Reward", "reward", output_dir=folder)
            figure = plt.gcf()
            np.testing.assert_array_equal(figure.axes[0].get_xticks(), [13, 27])
            self.assertTrue((Path(folder) / "test_reward.png").is_file())
        plt.close(figure)

    def test_defaults_do_not_overwrite_historical_outputs(self):
        self.assertEqual(cost.OUTPUT_DIR.name, "corrected_v2")
        self.assertEqual(hetero.OUTPUT_DIR.name, "corrected_v2")
        self.assertEqual(suite.OUTPUT_DIR.name, "corrected_v2")

    def test_collector_rejects_unversioned_csv(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "computation_cost_summary.csv"
            pd.DataFrame(dict(policy=["Myopic"], mean_reward=[1.0])).to_csv(path, index=False)
            with self.assertRaisesRegex(FileNotFoundError, "metadata"):
                suite.read_corrected_cost(path)

    def test_no_selected_timing_families_is_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            suite.run_computation_cost_suite(run_known=False, run_heterogeneous=False, run_unknown=False)


if __name__ == "__main__":
    unittest.main()
