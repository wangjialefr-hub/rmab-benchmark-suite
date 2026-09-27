"""Known-model benchmark with signed differences to stationary and finite LPs.

Import run_benchmark() for small sweeps. Corrected outputs are kept under
instance_matrix_outputs/corrected_v2/, leaving historical results in place.
"""

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import bandit_lp
from known_model_extra_instances import (
    build_extended_known_model_instance_library, extra_instance_metadata_dataframe,
)
from make_policy import (
    POLICY_NAMES, make_policy, canonical_policy_name, QWHITTLE_GAMMA,
    QWHITTLE_NUM_PENALTIES, QWHITTLE_STEPS_PER_PENALTY, QWHITTLE_TRAINING_SEED,
)
from paper_config import KNOWN_MODEL_BENCHMARK_INSTANCES
from simulation_cache import (
    cached_finite_horizon_bound, cached_lp_upper_bound, cached_simulate_policy,
    simulation_code_hash,
)
from simulation_utils import SIMULATION_VERSION, integer_budget, state_counts


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "instance_matrix_outputs" / "corrected_v2"
CACHE_DIR = CURRENT_DIR / "rmab_cache"
INSTANCE_NAMES = list(KNOWN_MODEL_BENCHMARK_INSTANCES)
N_VALUES = [20, 50, 100, 200, 500]
NUM_MONTE_CARLO = 20
MC_SEED = 123
SIMULATION_HORIZON = 200  # Measured steps, after any burn-in.
LP_UPDATE_HORIZON = 20
BURN_IN = 0  # Preserve the original finite-horizon protocol explicitly.


def safe_filename(name):
    return re.sub(r"[ /\\:]", "_", name)


def relative_gap(lp_reference, reward):
    """Signed difference: a finite-run estimate may exceed the LP reference."""
    return np.nan if abs(lp_reference) <= 1e-12 else (lp_reference - reward) / abs(lp_reference)


def calc_beta_one_group(group):
    """Fit Gap(N)=C*N**(-beta) only to resolved positive gaps.

    When replicate variability is available, require the pointwise 95% t-interval
    to lie above zero. This screening is not a confidence interval for beta.
    The fitted finite-range slope is descriptive, not an asymptotic guarantee.
    """
    positive = (np.isfinite(group["N"]) & (group["N"] > 0)
                & np.isfinite(group["mean_relative_gap"]) & (group["mean_relative_gap"] > 1e-12))
    resolved = positive.copy()
    if {"std_relative_gap", "num_runs"} <= set(group.columns):
        n = group["num_runs"].to_numpy(dtype=float)
        margin = student_t.ppf(0.975, np.maximum(n - 1, 1)) * group["std_relative_gap"] / np.sqrt(n)
        resolved &= (n > 1) & (group["mean_relative_gap"] - margin > 1e-12)
    valid = group[resolved].sort_values("N")
    result = {"convergence_beta": np.nan, "r_squared": np.nan,
              "num_fit_points": len(valid), "num_excluded_points": len(group) - len(valid),
              "fit_status": "insufficient_resolved_positive_points"}
    if len(valid) < 3:
        return pd.Series(result)
    log_n = np.log(valid["N"].to_numpy(dtype=float))
    log_gap = np.log(valid["mean_relative_gap"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(log_n, log_gap, 1)
    total = np.sum((log_gap - log_gap.mean()) ** 2)
    result.update(convergence_beta=-slope,
                  r_squared=np.nan if total <= 0 else 1 - np.sum((log_gap - slope * log_n - intercept) ** 2) / total,
                  fit_status="descriptive_fit")
    return pd.Series(result)


def estimate_beta(summary_df):
    rows = []
    for (instance, policy), group in summary_df.groupby(["instance", "policy"]):
        rows.append({"instance": instance, "policy": policy, **calc_beta_one_group(group).to_dict()})
    return pd.DataFrame(rows)


def plot_instance_metric(instance, summary_df, metric, ylabel, log_y, suffix, output_dir=OUTPUT_DIR):
    """Plot raw signed values; use symlog if zero or negative values occur."""
    frame = summary_df[summary_df["instance"] == instance]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    for i, (policy, group) in enumerate(frame.groupby("policy", sort=True)):
        group = group.sort_values("N")
        ax.plot(group["N"], group[metric], marker=markers[i % len(markers)],
                linestyle=["-", "--", "-."][i % 3], label=policy, linewidth=1.6, markersize=4)
    if log_y:
        finite = frame.loc[np.isfinite(frame[metric]), metric]
        if len(finite) and (finite > 0).all():
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.axhline(0, color="black", linewidth=0.7)
    ax.set(title=instance, xlabel="Number of arms (N)", ylabel=ylabel, xscale="log")
    n_values = sorted(frame["N"].unique())
    ax.set_xticks(n_values, labels=[str(n) for n in n_values])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    path = Path(output_dir) / f"{safe_filename(instance)}_{suffix}.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def run_benchmark(*, instance_names=INSTANCE_NAMES, policy_names=POLICY_NAMES,
                  n_values=N_VALUES, num_monte_carlo=NUM_MONTE_CARLO,
                  simulation_horizon=SIMULATION_HORIZON, burn_in=BURN_IN,
                  lp_update_horizon=LP_UPDATE_HORIZON, mc_seed=MC_SEED,
                  output_dir=OUTPUT_DIR, cache_dir=CACHE_DIR,
                  include_finite_horizon_bound=True, force_recompute=False, make_plots=True):
    """Run the matrix; save raw data, summaries, failures and experiment settings.

    Stationary differences use scored steps. Finite-horizon differences use ALL
    steps, including burn-in, and the same rounded initial state. Neither signed
    empirical difference is guaranteed to be nonnegative in a stochastic run.
    """
    for name, value, minimum in (("num_monte_carlo", num_monte_carlo, 1),
                                 ("simulation_horizon", simulation_horizon, 1), ("burn_in", burn_in, 0)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if not instance_names or not policy_names or not n_values:
        raise ValueError("The instance, policy and N lists must be nonempty.")
    if len(set(n_values)) != len(n_values):
        raise ValueError("N values must be distinct.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    library = build_extended_known_model_instance_library(bandit_lp)
    missing = set(instance_names) - set(library)
    if missing:
        raise ValueError(f"Unknown instances: {sorted(missing)}")
    provenance = {"simulation_version": SIMULATION_VERSION, "code_hash": simulation_code_hash(),
                  "instances": list(instance_names), "policies": list(policy_names), "N": list(n_values),
                  "num_monte_carlo": num_monte_carlo, "horizon": simulation_horizon,
                  "burn_in": burn_in, "lp_update_horizon": lp_update_horizon, "mc_seed": mc_seed,
                  "finite_horizon_bound": include_finite_horizon_bound,
                  "stationary_reference": "infinite_horizon_average_reward",
                  "rounding": "largest_remainder_exact_budget"}
    if any(canonical_policy_name(name) == "QWhittleKnownModel" for name in policy_names):
        provenance["qwhittle_training"] = dict(
            gamma=QWHITTLE_GAMMA, num_penalties=QWHITTLE_NUM_PENALTIES,
            steps_per_penalty=QWHITTLE_STEPS_PER_PENALTY, training_seed=QWHITTLE_TRAINING_SEED,
        )
    (output_dir / "experiment_settings.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    metadata = pd.DataFrame(extra_instance_metadata_dataframe(library))
    metadata[metadata["instance"].isin(instance_names)].to_csv(output_dir / "instance_library.csv", index=False)
    rows = []
    for instance_name in instance_names:
        spec = library[instance_name]
        bandit = spec.bandit
        print(f"\nInstance: {instance_name}", flush=True)
        for N in n_values:
            alpha = integer_budget(spec.default_alpha, N) / N
            lp, _ = cached_lp_upper_bound(bandit=bandit, alpha=alpha, cache_dir=cache_dir,
                                          force_recompute=force_recompute)
            finite_lp = np.nan
            if include_finite_horizon_bound:
                finite_lp = cached_finite_horizon_bound(
                    bandit=bandit, alpha=alpha, initial_state=state_counts(spec.initial_state, N) / N,
                    time_horizon=simulation_horizon + burn_in, cache_dir=cache_dir,
                    force_recompute=force_recompute,
                )
            for name in policy_names:
                hits, failures = 0, 0
                for replication in range(num_monte_carlo):
                    seed = mc_seed + replication
                    row = dict(instance=instance_name, source=spec.source, S=bandit.S, A=bandit.A,
                               alpha=alpha, configured_alpha=spec.default_alpha, N=N, policy=name,
                               replication=replication, seed=seed, lp_upper_bound=lp,
                               finite_horizon_lp_bound=finite_lp, horizon=simulation_horizon, burn_in=burn_in,
                               simulation_version=SIMULATION_VERSION, code_hash=provenance["code_hash"])
                    try:
                        mean, x, rewards, y, info = cached_simulate_policy(
                            bandit=bandit, policy_name=name, make_policy=make_policy,
                            initial_state=spec.initial_state, N=N, time_horizon=simulation_horizon,
                            seed=seed, alpha=alpha, lp_update_horizon=lp_update_horizon,
                            cache_dir=cache_dir, burn_in=burn_in, force_recompute=force_recompute,
                            return_info=True,
                        )
                        full_mean = float(np.mean(rewards))
                        hits += int(info["cache_hit"])
                        row.update(simulation_mean_reward=mean, full_horizon_mean_reward=full_mean,
                                   gap=lp - mean, relative_gap=relative_gap(lp, mean),
                                   finite_horizon_relative_gap=relative_gap(finite_lp, full_mean),
                                   runtime_seconds=info["runtime_seconds"], cache_hit=info["cache_hit"],
                                   cache_file=info["cache_file"], status="ok", error="")
                    except Exception as exc:
                        failures += 1
                        row.update(simulation_mean_reward=np.nan, full_horizon_mean_reward=np.nan,
                                   gap=np.nan, relative_gap=np.nan, finite_horizon_relative_gap=np.nan,
                                   runtime_seconds=np.nan, cache_hit=False, cache_file="",
                                   status="failed", error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
                # Save completed rows after each policy so an interruption is recoverable.
                pd.DataFrame(rows).to_csv(output_dir / "raw_instance_matrix_results.csv", index=False)
                label = "hit" if hits == num_monte_carlo else "miss"
                print(f"  N={N}, {name}: {label} ({hits}/{num_monte_carlo} cached, {failures} failed)", flush=True)

    raw = pd.DataFrame(rows)
    status = raw.groupby(["instance", "N", "policy"], as_index=False).agg(
        attempted_runs=("status", "size"), successful_runs=("status", lambda s: (s == "ok").sum()),
        error=("error", lambda s: "; ".join(sorted(set(s) - {""}))))
    status.to_csv(output_dir / "policy_run_status.csv", index=False)
    ok = raw[raw["status"] == "ok"]
    if ok.empty:
        raise RuntimeError(f"All runs failed. Inspect {output_dir / 'policy_run_status.csv'}")
    summary = ok.groupby(["instance", "policy", "N"], as_index=False).agg(
        S=("S", "first"), A=("A", "first"), alpha=("alpha", "first"),
        configured_alpha=("configured_alpha", "first"), lp_upper_bound=("lp_upper_bound", "first"),
        finite_horizon_lp_bound=("finite_horizon_lp_bound", "first"),
        mean_reward=("simulation_mean_reward", "mean"), std_reward=("simulation_mean_reward", "std"),
        mean_full_horizon_reward=("full_horizon_mean_reward", "mean"),
        mean_gap=("gap", "mean"), std_gap=("gap", "std"),
        mean_relative_gap=("relative_gap", "mean"), std_relative_gap=("relative_gap", "std"),
        mean_finite_horizon_relative_gap=("finite_horizon_relative_gap", "mean"),
        mean_runtime=("runtime_seconds", "mean"), num_runs=("status", "size"),
        cache_hits=("cache_hit", "sum"), horizon=("horizon", "first"), burn_in=("burn_in", "first"),
        simulation_version=("simulation_version", "first"), code_hash=("code_hash", "first"),
    ).sort_values(["instance", "N", "policy"])
    beta = estimate_beta(summary)
    summary.to_csv(output_dir / "summary_instance_matrix.csv", index=False)
    beta.to_csv(output_dir / "convergence_beta_by_instance.csv", index=False)
    largest = max(n_values)
    final = summary[summary["N"] == largest]
    final.to_csv(output_dir / f"summary_final_N{largest}.csv", index=False)
    for label, metric in (("reward", "mean_reward"), ("relative_gap", "mean_relative_gap")):
        final.pivot(index="instance", columns="policy", values=metric).to_csv(output_dir / f"pivot_{label}_final_N{largest}.csv")
    if make_plots:
        for instance in summary["instance"].unique():
            plot_instance_metric(instance, summary, "mean_reward", "Average reward per arm", False, "reward_vs_N", output_dir)
            plot_instance_metric(instance, summary, "mean_relative_gap", "Signed relative difference to stationary LP", True, "relative_gap_vs_N", output_dir)
            if include_finite_horizon_bound:
                plot_instance_metric(instance, summary, "mean_finite_horizon_relative_gap", "Signed relative difference to finite-horizon LP (all steps)", False, "finite_horizon_gap_vs_N", output_dir)
    print(f"\nSaved results: {output_dir}", flush=True)
    return {"raw": raw, "summary": summary, "beta": beta, "status": status}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=INSTANCE_NAMES)
    parser.add_argument("--policies", nargs="+", default=POLICY_NAMES)
    parser.add_argument("--n-values", nargs="+", type=int, default=N_VALUES)
    parser.add_argument("--replications", type=int, default=NUM_MONTE_CARLO)
    parser.add_argument("--horizon", type=int, default=SIMULATION_HORIZON)
    parser.add_argument("--burn-in", type=int, default=BURN_IN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    run_benchmark(instance_names=args.instances, policy_names=args.policies,
                  n_values=args.n_values, num_monte_carlo=args.replications,
                  simulation_horizon=args.horizon, burn_in=args.burn_in,
                  output_dir=args.output_dir, force_recompute=args.force_recompute)
