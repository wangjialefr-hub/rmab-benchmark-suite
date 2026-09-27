"""Summarize one corrected known-model experiment, without automatic recommendations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from report_data import (
    CORRECTED_DATA, CORRECTED_REPORT, complete_reward_ranks,
    load_corrected_results, write_manifest,
)


def top_known_reward(summary_final):
    """Include all ties at third place rather than arbitrarily dropping policies."""
    ranks = summary_final.groupby("instance")["mean_reward"].rank(method="min", ascending=False)
    return summary_final.loc[ranks <= 3, [
        "instance", "policy", "N", "mean_reward", "mean_relative_gap", "alpha", "num_runs",
    ]].sort_values(["instance", "mean_reward", "policy"], ascending=[True, False, True])


def aggregate_known_model_recommendation(summary_final, beta_df, cost_known=None):
    """Keep the legacy name, but return descriptive ranks, not a weighted score.

    Reward and signed gap give the same order within an instance. Counting both
    would double-weight performance. Beta can be undefined for excellent policies
    whose gap is indistinguishable from zero; it must not be treated as a penalty.
    Cost is not mixed in without a matched timing protocol.
    """
    policies = summary_final.attrs.get("policies", sorted(summary_final["policy"].unique()))
    ranks = complete_reward_ranks(summary_final, summary_final["N"].max())
    table = pd.DataFrame({"policy": policies})
    table["avg_reward_rank"] = table["policy"].map(ranks.mean(axis=0))
    table["shared_instances"] = len(ranks)
    table["available_instances"] = table["policy"].map(
        summary_final.groupby("policy")["instance"].nunique()
    ).fillna(0).astype(int)
    beta_count = beta_df[np.isfinite(beta_df["convergence_beta"])].groupby("policy").size()
    table["beta_fit_instances"] = table["policy"].map(beta_count).fillna(0).astype(int)
    table["interpretation"] = "descriptive reward rank only; no overall recommendation"
    return table.sort_values(["avg_reward_rank", "policy"], na_position="last").reset_index(drop=True)


def generate_benchmark_findings(output_dir=CORRECTED_REPORT, *, known_model_dir=CORRECTED_DATA):
    summary, beta, manifest = load_corrected_results(known_model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    largest_n = int(summary["N"].max())
    final = summary[summary["N"] == largest_n].copy()
    top = top_known_reward(final)
    comparison = aggregate_known_model_recommendation(final, beta)
    # Save every policy and every unavailable fit, not only a top-beta shortlist.
    top.to_csv(output_dir / "known_model_top_policies.csv", index=False)
    beta.to_csv(output_dir / "convergence_beta_diagnostics.csv", index=False)
    comparison.to_csv(output_dir / "policy_comparison.csv", index=False)
    final.to_csv(output_dir / "known_model_final_results.csv", index=False)
    write_manifest(output_dir, manifest)
    settings = manifest["settings"]
    text = f"""# Corrected Benchmark Summary

Scope: **{manifest['report_scope']}**. This is a data summary, not a final policy recommendation.

Source: `{manifest['source_directory']}`.

## Experimental Coverage

- Instances: {len(settings['instances'])}; selected policies: {len(settings['policies'])}.
- N: {settings['N']}; replications per combination: {settings['num_monte_carlo']}.
- Measured horizon: {settings['horizon']}; burn-in: {settings['burn_in']}.
- Attempted runs: {manifest['attempted_runs']}; failures: {manifest['failed_runs']}.
- Incomplete successful groups excluded from comparison: {len(manifest['excluded_incomplete_groups'])}.

The complete inputs and code fingerprint are recorded in `report_inputs.json`.
Completing the default grid does not by itself establish publication readiness.

## Reward at N={largest_n}

The table includes the top three ranks and all ties. Rankings are descriptive;
small numerical differences are not evidence of statistical significance.

{top.to_markdown(index=False)}

## Cross-Instance Comparison

Average reward ranks below use only instances with complete results for **every
selected policy**. Missing policies are not ranked on a smaller, easier subset.
If there is no shared instance, the average ranks are unavailable.
Exact ties receive their average rank: four policies tied for first receive 2.5.

{comparison.to_markdown(index=False)}

No weighted overall score is reported. Reward and signed gap largely duplicate
one another as ranking criteria. An unavailable beta is not a poor performance
score. A reward-versus-cost recommendation also requires matched timing data,
which this report has not loaded.

## Signed Differences and Beta

`mean_relative_gap` compares the measured reward with the stationary LP value.
It remains signed. Finite-horizon startup effects and sampling variability can
produce negative values; those values are not silently replaced by zero.
The optional finite-horizon LP uses the full trajectory, including burn-in,
and bounds expected reward rather than each sampled realization.

`convergence_beta_diagnostics.csv` contains every fit, its number of retained
points, excluded points and status. Beta is a descriptive log-log slope over
resolved positive gaps, not a proof of convergence. No policy ranking by beta
is used to select an overall winner.

## Not Included

Historical computation-cost, heterogeneous-arm and unknown-model CSVs were not
merged into these corrected results. They require their own checked reruns and
explicitly documented protocols. No conclusion about those experiments is made here.
"""
    (output_dir / "benchmark_findings_draft.md").write_text(text, encoding="utf-8")
    print(f"Saved corrected findings ({manifest['report_scope']}) to:\n{output_dir}")
    return {"known_top": top, "beta": beta, "comparison": comparison, "manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-model-dir", type=Path, default=CORRECTED_DATA)
    parser.add_argument("--output-dir", type=Path, default=CORRECTED_REPORT)
    args = parser.parse_args()
    FINDINGS = generate_benchmark_findings(output_dir=args.output_dir, known_model_dir=args.known_model_dir)
