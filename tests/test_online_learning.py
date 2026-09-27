"""Check model access, online updates, diagnostics and timing boundaries."""

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
import numpy as np
import pandas as pd

import bandit_lp
import unknown_model_learning as online
import run_unknown_model_benchmark as runner
from known_model_extra_instances import deadline_scheduling_instance
from run_computation_cost_suite import normalize_unknown_cost


class DimensionsOnly:
    S, A = 2, 2

    @property
    def P(self):
        raise AssertionError("Learner read the hidden transition model")

    @property
    def R(self):
        raise AssertionError("Learner read the hidden reward model")


class OnlineLearningTests(unittest.TestCase):
    def setUp(self):
        self.spec = deadline_scheduling_instance(bandit_lp, state_count=10, alpha=0.2)

    def test_unknown_factories_and_updates_do_not_read_true_model(self):
        hidden = DimensionsOnly()
        for name in online.UNKNOWN_POLICY_NAMES:
            with self.subTest(policy=name):
                policy = online.make_unknown_model_policy(name, hidden, N=10, alpha=0.4)
                policy.reset(123)
                states = np.arange(10) % 2
                actions = policy.select_actions(states)
                self.assertEqual(actions.sum(), 4)
                policy.update(states, actions, np.ones(10), 1 - states)
                self.assertEqual(policy.t, 1)

    def test_q_update_is_the_documented_discounted_sample_update(self):
        policy = online.OnlineQLearningIndexPolicy(2, 2, 1, 1, gamma=0.5)
        policy.reset(123)
        policy.Q[:] = [[1, 2], [4, 1]]
        policy.update([0], [1], [3.0], [1])
        # First visit: eta=0.5, target=3+0.5*4=5, updated value=2+0.5*(5-2).
        self.assertEqual(policy.Q[0, 1], 3.5)
        self.assertEqual(policy.visit_count[0, 1], 1)
        self.assertEqual(policy.t, 1)

    def test_all_arm_samples_are_observed_and_reset_between_runs(self):
        policy = online.OnlineQLearningIndexPolicy(10, 2, 10, 0.4)
        settings = dict(bandit=self.spec.bandit, policy=policy, initial_state=self.spec.initial_state,
                        N=10, horizon=4, seed=123)
        first = online.simulate_unknown_model(**settings)
        self.assertEqual(policy.visit_count[:, 1].sum(), 16)
        self.assertEqual(policy.visit_count[:, 0].sum(), 24)
        first_q = policy.Q.copy()
        second = online.simulate_unknown_model(**settings)
        np.testing.assert_array_equal(first["reward_history"], second["reward_history"])
        np.testing.assert_array_equal(first_q, policy.Q)
        self.assertEqual(policy.visit_count.sum(), 40)

    def test_budget_snaps_float_near_integer_and_oracle_uses_effective_alpha(self):
        policy = online.OnlineRandomPolicy(2, 2, 100, 0.58)
        self.assertEqual(policy.budget, 58)
        with patch.object(online.strategies, "LPPriorityStrategy", return_value=SimpleNamespace(lp_index=np.zeros(10))) as create:
            policy = online.KnownLPPriorityOraclePolicy(self.spec.bandit, N=23, alpha=0.2)
            self.assertEqual(policy.budget, 4)
            self.assertEqual(create.call_args.args[1], 4 / 23)

    def test_fallback_is_counted_and_reported(self):
        policy = online.OnlinePlugInWhittlePolicy(10, 2, 10, 0.2,
                                                recompute_interval=1, epsilon_start=0, epsilon_min=0)
        with patch.object(online.strategies, "WhittleIndexStrategy", side_effect=RuntimeError("forced index failure")):
            result = online.simulate_unknown_model(self.spec.bandit, policy, self.spec.initial_state, 10, 3, 123)
        self.assertEqual(result["num_index_recomputations"], 4)  # reset and three updates
        self.assertEqual(result["num_index_failures"], 4)
        self.assertEqual(result["num_fallback_decisions"], 3)
        self.assertIn("forced index failure", result["last_index_error"])

    def test_fallback_does_not_access_true_model_and_estimates_use_samples(self):
        policy = online.OnlinePlugInWhittlePolicy(2, 2, 2, 0.5, recompute_interval=1)
        with patch.object(online.strategies, "WhittleIndexStrategy", side_effect=RuntimeError("forced")):
            policy.reset(123)
            policy.update([0, 1], [0, 1], [2.0, 4.0], [1, 0])
        P, R = policy._estimate_model()
        np.testing.assert_allclose(P[0, 0], [1 / 3, 2 / 3])
        np.testing.assert_allclose(P[1, 1], [2 / 3, 1 / 3])
        self.assertEqual(R[0, 0], 1.0)  # one zero-valued reward prior and one sample
        self.assertEqual(R[1, 1], 2.0)

    def test_simulator_rejects_nonbinary_actions(self):
        for value in (0.5, -1, 2, np.nan):
            policy = online.OnlineRandomPolicy(10, 2, 10, 0.2)
            actions = np.zeros(10)
            actions[0] = value
            with patch.object(policy, "select_actions", return_value=actions), self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "binary"):
                    online.simulate_unknown_model(self.spec.bandit, policy, self.spec.initial_state, 10, 2, 123)

    def test_simulator_protocol_and_tail_are_validated(self):
        policy = online.OnlineRandomPolicy(10, 2, 10, 0.2)
        settings = dict(bandit=self.spec.bandit, policy=policy, initial_state=self.spec.initial_state,
                        N=10, horizon=7, seed=None, tail_fraction=0.25)
        result = online.simulate_unknown_model(**settings)
        self.assertEqual(result["tail_start"], 5)
        self.assertEqual(result["tail_steps"], 2)
        self.assertEqual(result["tail_mean_reward"], np.mean(result["reward_history"][-2:]))
        for updates in (dict(horizon=0), dict(tail_fraction=0), dict(tail_fraction=1.1),
                        dict(N=20), dict(initial_state=np.ones(10))):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                online.simulate_unknown_model(**(settings | updates))

    def test_learner_hyperparameters_are_validated(self):
        for values in (dict(recompute_interval=0), dict(recompute_interval=1.5), dict(transition_prior=0),
                       dict(reward_prior_count=-1), dict(epsilon_min=0.9)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                online.OnlinePlugInWhittlePolicy(2, 2, 10, 0.4, **values)
        with self.assertRaises(ValueError):
            online.OnlineQLearningIndexPolicy(2, 2, 10, 0.4, gamma=1)

    def run_small(self, folder, policies, **settings):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
            runner, "build_extended_known_model_instance_library", return_value={"deadline": self.spec}
        ):
            return runner.run_unknown_model_benchmark(
                instance_names=["deadline"], policy_names=policies, n_values=[23],
                num_monte_carlo=2, simulation_horizon=7, curve_sample_every=3,
                output_dir=folder, make_plots=False, **settings,
            )

    def test_runner_records_failures_split_timers_and_protocol(self):
        with TemporaryDirectory() as folder:
            result = self.run_small(folder, ["OnlineRandom", "not_a_policy"])
            self.assertEqual(len(result["raw"]), 4)
            self.assertEqual(result["raw"]["status"].eq("failed").sum(), 2)
            valid = result["raw"].query("status == 'ok'")
            np.testing.assert_allclose(valid["runtime_seconds"], valid["setup_seconds"] + valid["simulation_seconds"])
            self.assertTrue(valid["budget"].eq(4).all())
            self.assertTrue(valid["alpha"].eq(4 / 23).all())
            self.assertEqual(result["learning_curves"]["time"].tolist(), [1, 4, 7])
            metadata = json.loads((Path(folder) / "experiment_settings.json").read_text())
            self.assertIn("including passive arms", metadata["feedback"])
            self.assertTrue(result["summary"]["complete"].all())

    def test_all_failed_run_still_writes_readable_csvs(self):
        with TemporaryDirectory() as folder:
            result = self.run_small(folder, ["not_a_policy"])
            self.assertTrue(result["summary"].empty)
            self.assertTrue(result["learning_curves"].empty)
            self.assertEqual(len(pd.read_csv(Path(folder) / "unknown_model_raw_results.csv")), 2)
            self.assertIn("time", pd.read_csv(Path(folder) / "unknown_model_learning_curves.csv"))

    def test_incomplete_group_is_marked_not_silently_complete(self):
        calls = 0
        original = runner.simulate_unknown_model

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("forced simulation failure")
            return original(*args, **kwargs)

        with TemporaryDirectory() as folder, patch.object(runner, "simulate_unknown_model", side_effect=fail_once):
            result = self.run_small(folder, ["OnlineRandom"])
        self.assertFalse(result["summary"]["complete"].any())
        self.assertFalse(result["learning_curves"]["complete"].any())
        self.assertEqual(result["raw"].iloc[0]["failure_phase"], "simulation")

    def test_oracle_cost_classification_and_old_timer_rejection(self):
        frame = pd.DataFrame(dict(
            instance=["test"] * 2, policy=["KnownWhittleOracle", "OnlineRandom"],
            policy_type=["known_model_oracle", "unknown_model_online"],
            N=20, S=2, A=2, alpha=0.2, mean_tail_reward=1.0, horizon=10,
            mean_setup_seconds=[2.0, 0.1], mean_simulation_seconds=[3.0, 4.0], mean_runtime_seconds=[5.0, 4.1],
        ))
        normalized = normalize_unknown_cost(frame)
        self.assertEqual(normalized["model_knowledge"].tolist(), ["known_P_R_oracle", "unknown_P_R"])
        np.testing.assert_allclose(normalized["online_seconds"], [3, 4])
        np.testing.assert_allclose(normalized["seconds_per_step"], [0.3, 0.4])
        with self.assertRaisesRegex(ValueError, "corrected"):
            normalize_unknown_cost(frame.drop(columns="mean_setup_seconds"))

    def test_unknown_runner_defaults_and_invalid_axes(self):
        self.assertEqual(runner.OUTPUT_DIR.name, "corrected_v2")
        for settings in (dict(n_values=[]), dict(n_values=[20, 20]), dict(curve_sample_every=0),
                         dict(tail_fraction=0), dict(num_monte_carlo=0)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                runner.run_unknown_model_benchmark(**settings)


if __name__ == "__main__":
    unittest.main()
