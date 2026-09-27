"""Regression tests for the September 2026 simulator/FTVA corrections.

Run from the repository root: python -m unittest discover -s tests -v
"""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bandit_lp
import strategies
from known_model_extra_instances import (
    deadline_scheduling_instance, maintenance_degradation_instance,
)
from make_policy import make_policy
from simulation_cache import cached_simulate_policy
from simulation_utils import activation_probabilities, integer_action_counts, state_counts


def deadline():
    return deadline_scheduling_instance(bandit_lp, state_count=10, alpha=0.2)


class AllocationTests(unittest.TestCase):
    def test_near_integer_actions_and_reward_use_same_counts(self):
        spec = deadline()
        policy = make_policy("Myopic", spec.bandit, 0.2)
        y = policy.next_y(spec.initial_state, 500)
        next_x, reward, executed = spec.bandit.next_x_from_y(y, 500, return_y=True)
        self.assertEqual(int(round(500 * executed[:, 1].sum())), 100)
        np.testing.assert_allclose(executed.sum(axis=1), spec.initial_state)
        np.testing.assert_allclose(next_x, np.einsum("sa,sak->k", executed, spec.bandit.P))
        self.assertAlmostEqual(reward, np.sum(executed * spec.bandit.R))
        self.assertAlmostEqual(next_x[0], 0.2)
        self.assertAlmostEqual(next_x[9], 0.0)

    def test_fractional_allocations_preserve_budget_and_populations(self):
        rng = np.random.default_rng(42)
        for N in (1, 7, 20, 23, 500, 5000):
            for _ in range(30):
                counts = rng.multinomial(N, np.full(10, 0.1))
                x = counts / N
                active = x * rng.random(10)
                y = np.column_stack((x - active, active))
                actual = integer_action_counts(y, N)
                np.testing.assert_array_equal(actual.sum(axis=1), counts)
                self.assertEqual(actual[:, 1].sum(), int(np.floor(N * active.sum())))
                self.assertTrue(np.all(actual >= 0))
                self.assertTrue(np.all(np.abs(actual[:, 1] - N * active) <= 1 + 1e-8))

    def test_initial_rounding_preserves_empirical_counts(self):
        counts = np.array([0, 1, 7, 11, 4])
        np.testing.assert_array_equal(state_counts(counts / 23, 23), counts)
        self.assertEqual(state_counts(np.full(10, 0.1), 23).sum(), 23)

    def test_invalid_models_and_allocations_fail_clearly(self):
        with self.assertRaises(ValueError):
            bandit_lp.BanditInstance(np.zeros((2, 2, 2)), np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            integer_action_counts(np.array([[0.8, -0.1], [0.2, 0.1]]), 20)
        with self.assertRaises(ValueError):
            state_counts([0.2, 0.2], 20)

    def test_rounded_literature_probabilities_are_normalized(self):
        P = np.full((2, 2, 2), 0.499999995)
        model = bandit_lp.BanditInstance(P, np.zeros((2, 2)))
        np.testing.assert_allclose(model.P.sum(axis=2), 1, atol=1e-15, rtol=0)
        self.assertGreater(model.transition_row_error, 1e-9)

    def test_infeasible_lp_is_not_returned_as_a_valid_solution(self):
        with self.assertRaisesRegex(RuntimeError, "Infeasible"):
            deadline().bandit.relaxed_lp_average_reward(1.5)


class FTVATests(unittest.TestCase):
    def test_zero_support_completion_matches_equation_8(self):
        with np.errstate(divide="raise", invalid="raise"):
            np.testing.assert_allclose(
                activation_probabilities([[0, 0], [0.3, 0.1], [0, 0.2]]),
                [0.5, 0.25, 1.0],
            )

    def test_deadline_and_maintenance_ftva_are_valid(self):
        specs = [deadline(), maintenance_degradation_instance(bandit_lp, state_count=10, alpha=0.2)]
        for spec in specs:
            with self.subTest(instance=spec.name), np.errstate(divide="raise", invalid="raise"):
                policy = make_policy("FTVA", spec.bandit, 0.2)
                zero_states = policy.x_star == 0
                self.assertTrue(np.any(zero_states))
                np.testing.assert_allclose(policy.pi_star[zero_states], 0.5)
                kernel = ((1 - policy.pi_star[:, None]) * spec.bandit.P[:, 0]
                          + policy.pi_star[:, None] * spec.bandit.P[:, 1])
                np.testing.assert_allclose(policy.x_star @ kernel, policy.x_star, atol=1e-7)
                mean, x, rewards, y = strategies.simulate(
                    spec.bandit, policy, spec.initial_state, 23, 40, seed=123, use_cache=False,
                )
                self.assertTrue(np.isfinite(mean))
                np.testing.assert_allclose(y.sum(axis=2), x, atol=1e-12)
                np.testing.assert_allclose(y[:, :, 1].sum(axis=1), 4 / 23)
                np.testing.assert_allclose(rewards, np.einsum("tsa,sa->t", y, spec.bandit.R))

    def test_stateful_policies_restart_reproducibly(self):
        spec = deadline()
        for name in ("FTVA", "RoundRobin"):
            policy = make_policy(name, spec.bandit, 0.2)
            first = strategies.simulate(spec.bandit, policy, spec.initial_state, 23, 30, seed=7, use_cache=False)
            second = strategies.simulate(spec.bandit, policy, spec.initial_state, 23, 30, seed=7, use_cache=False)
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)

    def test_import_does_not_change_numpy_error_settings(self):
        code = "import numpy as np; before=np.geterr(); import strategies; assert np.geterr()==before"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class TrajectoryTests(unittest.TestCase):
    def test_deterministic_deadline_matches_finite_lp_and_explains_negative_gap(self):
        spec = deadline()
        g_stationary = spec.bandit.relaxed_lp_average_reward(0.2)[0]
        g_finite = spec.bandit.relaxed_lp_finite_time(0.2, spec.initial_state, 200)[0] / 200
        policy = make_policy("Myopic", spec.bandit, 0.2)
        mean, x, rewards, y = strategies.simulate(
            spec.bandit, policy, spec.initial_state, 500, 200, seed=123, use_cache=False,
        )
        self.assertAlmostEqual(g_stationary, 0.09)
        self.assertAlmostEqual(g_finite, 0.091175)
        self.assertAlmostEqual(mean, g_finite)
        self.assertGreater(mean, g_stationary)
        self.assertAlmostEqual(np.mean(rewards[20:]), g_stationary)
        np.testing.assert_allclose(x[1:], np.einsum("tsa,sak->tk", y[:-1], spec.bandit.P), atol=1e-12)

    def test_all_fast_policies_use_valid_actions(self):
        spec = deadline()
        for name in ("Whittle", "LPPriority", "FTVA", "LPUpdate", "Myopic", "RandomPriority", "RoundRobin", "LPRandomized"):
            with self.subTest(policy=name):
                policy = make_policy(name, spec.bandit, 0.2, lp_update_horizon=5)
                _, x, r, y = strategies.simulate(spec.bandit, policy, spec.initial_state, 20, 25, seed=123, use_cache=False)
                np.testing.assert_allclose(y.sum(axis=2), x, atol=1e-12)
                np.testing.assert_allclose(y[:, :, 1].sum(axis=1), 0.2, atol=1e-12)
                np.testing.assert_allclose(20 * y, np.rint(20 * y), atol=1e-12)
                np.testing.assert_allclose(r, np.einsum("tsa,sa->t", y, spec.bandit.R))
                np.testing.assert_allclose(x[1:], np.einsum("tsa,sak->tk", y[:-1], spec.bandit.P), atol=1e-12)


class CacheTests(unittest.TestCase):
    def test_miss_hit_force_and_burn_in(self):
        spec = deadline()
        with TemporaryDirectory() as cache:
            settings = dict(bandit=spec.bandit, policy_name="Myopic", make_policy=make_policy,
                            initial_state=spec.initial_state, N=20, time_horizon=20,
                            seed=123, alpha=0.2, cache_dir=cache, return_info=True)
            with patch("strategies.simulate", wraps=strategies.simulate) as simulate:
                first = cached_simulate_policy(**settings)
                second = cached_simulate_policy(**settings)
                forced = cached_simulate_policy(**settings, force_recompute=True)
                self.assertEqual(simulate.call_count, 2)
                self.assertFalse(simulate.call_args.kwargs["use_cache"])
            self.assertFalse(first[-1]["cache_hit"])
            self.assertTrue(second[-1]["cache_hit"])
            self.assertFalse(forced[-1]["cache_hit"])
            self.assertEqual(first[0], forced[0])
            tail = cached_simulate_policy(**settings, burn_in=20)
            self.assertFalse(tail[-1]["cache_hit"])
            self.assertEqual(len(tail[2]), 40)
            self.assertAlmostEqual(tail[0], 0.09)
            self.assertNotEqual(tail[-1]["cache_file"], first[-1]["cache_file"])

    def test_direct_simulator_cache_is_versioned_and_bypassable(self):
        spec = deadline()
        policy = make_policy("Myopic", spec.bandit, 0.2)
        original = Path.cwd()
        with TemporaryDirectory() as folder:
            os.chdir(folder)
            try:
                settings = dict(bandit=spec.bandit, strategy=policy, initial_state=spec.initial_state,
                                N=20, time=10, seed=123)
                first = strategies.simulate(**settings)
                with patch.object(policy, "next_y", side_effect=RuntimeError("executed")):
                    cached = strategies.simulate(**settings)
                    self.assertEqual(first[0], cached[0])
                    with self.assertRaisesRegex(RuntimeError, "executed"):
                        strategies.simulate(**settings, force_recompute=True)
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
