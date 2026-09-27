"""Reproduce the corrected deadline comparison without deleting old caches.

python verify_deadline.py          # 9 policies, 5 populations, 20 seeds, T=200
python verify_deadline.py --long   # also run FTVA for 4,000 steps, 5 seeds
The longer check can take tens of minutes. Compatible results are reused.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from report_data import load_corrected_results
from run_instance_matrix_benchmark import run_benchmark


ROOT = Path(__file__).resolve().parent


def main(long_check=False):
    output = ROOT / "verification_outputs" / "release_v2"
    shared = dict(instance_names=["deadline_S10_a20"],
                  cache_dir=ROOT / "verification_outputs" / "corrected_v2" / "cache")
    experiments = [("deadline_matrix", dict(n_values=[20, 50, 100, 200, 500],
                    num_monte_carlo=20, simulation_horizon=200))]
    if long_check:
        experiments.append(("deadline_ftva_long", dict(policy_names=["FTVA_Strategy"],
                            n_values=[20, 100, 500], num_monte_carlo=5,
                            simulation_horizon=4000, include_finite_horizon_bound=False)))
    for name, settings in experiments:
        folder = output / name
        run_benchmark(**shared, **settings, output_dir=folder)
        _, _, manifest = load_corrected_results(folder)
        if manifest["failed_runs"]:
            raise RuntimeError(f"Failed runs are recorded in {folder / 'policy_run_status.csv'}")
        print(f"Verified {manifest['attempted_runs']} successful runs: {folder}", flush=True)
    print("Successful execution does not establish asymptotic optimality.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", action="store_true", help="Also run the longer FTVA check.")
    main(parser.parse_args().long)
