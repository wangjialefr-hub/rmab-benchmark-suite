"""Reporting must not hide signs, mix versions, or reward missing experiments."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_paper_readiness import check_paper_readiness
from generate_benchmark_findings import aggregate_known_model_recommendation, generate_benchmark_findings
from generate_paper_draft import generate_paper_draft
from generate_paper_figures import generate_paper_figures, normalized_score, plot_known_model_gap_heatmap
from report_data import complete_reward_ranks, load_corrected_results
from run_instance_matrix_benchmark import run_benchmark


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source"
        with contextlib.redirect_stdout(io.StringIO()):
            run_benchmark(instance_names=["deadline_S10_a20"], policy_names=["Myopic", "FTVA_Strategy"],
                          n_values=[20, 40, 80], num_monte_carlo=2, simulation_horizon=15,
                          include_finite_horizon_bound=False, make_plots=False,
                          output_dir=cls.source, cache_dir=cls.root / "cache")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def copy_source(self, folder):
        return Path(shutil.copytree(self.source, Path(folder) / "data"))

    def test_corrected_subset_is_explicitly_preliminary(self):
        summary, beta, manifest = load_corrected_results(self.source)
        self.assertEqual(manifest["report_scope"], "preliminary subset")
        self.assertFalse(manifest["paper_grid_complete"])
        self.assertEqual(manifest["attempted_runs"], 12)
        self.assertTrue((summary["mean_relative_gap"] < 0).any())
        self.assertEqual(len(beta), 2)

    def test_legacy_folder_has_no_silent_fallback(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_corrected_results(folder)

    def test_mismatched_code_is_rejected(self):
        with TemporaryDirectory() as folder:
            source = self.copy_source(folder)
            path = source / "experiment_settings.json"
            settings = json.loads(path.read_text())
            settings["code_hash"] = "old-code"
            path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, "current simulator"):
                load_corrected_results(source)

    def test_changed_summary_is_rejected(self):
        with TemporaryDirectory() as folder:
            source = self.copy_source(folder)
            path = source / "summary_instance_matrix.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "mean_reward"] += 0.1
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "mean_reward"):
                load_corrected_results(source)

    def test_duplicate_runs_are_rejected(self):
        with TemporaryDirectory() as folder:
            source = self.copy_source(folder)
            path = source / "raw_instance_matrix_results.csv"
            frame = pd.read_csv(path)
            pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "experiment keys"):
                load_corrected_results(source)

    def test_partial_replication_group_is_not_ranked_as_complete(self):
        with TemporaryDirectory() as folder:
            source = self.copy_source(folder)
            raw_path = source / "raw_instance_matrix_results.csv"
            raw = pd.read_csv(raw_path)
            affected = (raw["policy"] == "FTVA_Strategy") & (raw["N"] == 20)
            raw.loc[affected & (raw["replication"] == 0), "status"] = "failed"
            raw.to_csv(raw_path, index=False)
            remaining = raw[affected & (raw["status"] == "ok")].iloc[0]
            path = source / "summary_instance_matrix.csv"
            summary = pd.read_csv(path)
            row = (summary["policy"] == "FTVA_Strategy") & (summary["N"] == 20)
            summary.loc[row, ["mean_reward", "mean_relative_gap", "num_runs"]] = [
                remaining["simulation_mean_reward"], remaining["relative_gap"], 1,
            ]
            summary.loc[row, ["std_reward", "std_relative_gap"]] = np.nan
            summary.to_csv(path, index=False)
            checked, _, manifest = load_corrected_results(source)
            self.assertEqual(manifest["failed_runs"], 1)
            self.assertEqual(len(manifest["excluded_incomplete_groups"]), 1)
            self.assertTrue(complete_reward_ranks(checked, 20).empty)

    def test_seed_protocol_mismatch_is_rejected(self):
        with TemporaryDirectory() as folder:
            source = self.copy_source(folder)
            path = source / "raw_instance_matrix_results.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "seed"] += 100
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "seeds"):
                load_corrected_results(source)

    def test_gap_heatmap_keeps_negative_values_and_order(self):
        frame = pd.DataFrame({"instance": ["a", "a"], "policy": ["P", "Q"],
                              "mean_relative_gap": [-0.1, 0.2]})
        with patch("generate_paper_figures.plot_heatmap") as draw:
            plot_known_model_gap_heatmap(frame, policy_order=["Q", "P"])
        matrix = draw.call_args.kwargs["matrix"]
        self.assertEqual(list(matrix.columns), ["Q", "P"])
        self.assertEqual(matrix.loc["a", "P"], -0.1)
        self.assertTrue(draw.call_args.kwargs["center_zero"])

    def test_missing_instances_do_not_improve_average_rank(self):
        summary = pd.DataFrame({"instance": ["easy", "easy", "hard"], "policy": ["P", "Q", "P"],
                                "N": [500] * 3, "mean_reward": [1.0, 2.0, 3.0]})
        ranks = complete_reward_ranks(summary, 500)
        self.assertEqual(list(ranks.index), ["easy"])
        beta = pd.DataFrame({"policy": ["P", "Q"], "convergence_beta": [np.nan, 1.0]})
        comparison = aggregate_known_model_recommendation(summary, beta)
        self.assertNotIn("overall_score", comparison)
        self.assertEqual(comparison["shared_instances"].tolist(), [1, 1])
        summary.attrs["policies"] = ["P", "Q", "missing_policy"]
        self.assertTrue(complete_reward_ranks(summary, 500).empty)

    def test_normalization_never_fills_missing_scores(self):
        result = normalized_score(pd.Series([2.0, 2.0, np.nan]), higher_is_better=True)
        np.testing.assert_equal(result.to_numpy(), [1.0, 1.0, np.nan])
        result = normalized_score(pd.Series([1.0, 3.0, np.inf]), higher_is_better=False)
        np.testing.assert_equal(result.to_numpy(), [1.0, 0.0, np.nan])

    def test_reporting_entry_points_use_only_selected_data(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder)
            figures = output / "figures"
            figures.mkdir()
            (figures / "old_cost_plot.png").write_bytes(b"historical file")
            findings = generate_benchmark_findings(output, known_model_dir=self.source)
            index = generate_paper_figures(output, known_model_dir=self.source)
            self.assertEqual(len(index), 4)
            self.assertNotIn("old_cost_plot.png", index["figure"].tolist())
            self.assertTrue((output / "policy_comparison.csv").exists())
            self.assertFalse((output / "aggregate_policy_recommendation.csv").exists())
            self.assertEqual(findings["manifest"]["source_directory"], str(self.source.resolve()))
            self.assertIn("preliminary subset", (output / "benchmark_findings_draft.md").read_text())

    def test_readiness_does_not_certify_a_partial_paper(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            result = check_paper_readiness(folder, known_model_dir=self.source)
            checks = result["checks"].set_index("check")
            self.assertEqual(checks.loc["default_paper_grid", "status"], "pending")
            self.assertEqual(checks.loc["corrected_timing_experiments", "status"], "not_verified")
            blocked = check_paper_readiness(folder, known_model_dir=Path(folder) / "missing")
            self.assertTrue(blocked["checks"]["status"].eq("blocked").any())

    def test_old_draft_requires_explicit_opt_in(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "historical manuscript"):
                generate_paper_draft(folder)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
