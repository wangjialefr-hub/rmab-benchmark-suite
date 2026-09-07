"""
Check whether the benchmark folder has enough outputs for a paper draft.

This script does not run experiments. It reports missing files and common
interpretation issues that should be mentioned in the paper.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from paper_config import (
    DUPLICATE_INSTANCE_GROUPS,
    KNOWN_MODEL_BENCHMARK_INSTANCES,
    PAPER_EXCLUDED_INSTANCES,
)


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "paper_summary_outputs"


REQUIRED_FILES = {
    "known_model_summary": CURRENT_DIR
    / "instance_matrix_outputs"
    / "summary_instance_matrix.csv",
    "known_model_final": CURRENT_DIR
    / "instance_matrix_outputs"
    / "summary_final_N500.csv",
    "known_model_beta": CURRENT_DIR
    / "instance_matrix_outputs"
    / "convergence_beta_by_instance.csv",
    "known_model_cost": CURRENT_DIR
    / "computation_cost_outputs"
    / "computation_cost_summary.csv",
    "heterogeneous_summary": CURRENT_DIR
    / "heterogeneous_outputs"
    / "heterogeneous_summary.csv",
    "unknown_model_summary": CURRENT_DIR
    / "unknown_model_outputs"
    / "unknown_model_summary.csv",
    "instance_taxonomy": OUTPUT_DIR / "instance_taxonomy.csv",
    "findings_draft": OUTPUT_DIR / "benchmark_findings_draft.md",
}


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def file_status_rows():
    rows = []
    for name, path in REQUIRED_FILES.items():
        rows.append(
            {
                "item": name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def known_model_checks():
    rows = []
    final_df = read_csv_if_exists(REQUIRED_FILES["known_model_final"])
    beta_df = read_csv_if_exists(REQUIRED_FILES["known_model_beta"])
    taxonomy_df = read_csv_if_exists(REQUIRED_FILES["instance_taxonomy"])
    draft_path = OUTPUT_DIR / "paper_draft_rmab_benchmark.md"
    draft_text = (
        draft_path.read_text(encoding="utf-8").lower()
        if draft_path.exists()
        else ""
    )

    if final_df.empty:
        rows.append(
            {
                "check": "known_model_final_results",
                "status": "missing",
                "message": "Run run_instance_matrix_benchmark.py.",
            }
        )
        return rows

    negative_gap = final_df[final_df["mean_relative_gap"] < 0]
    negative_gap_documented = (
        "negative relative gap" in draft_text
        or "finite-horizon" in draft_text
    )
    rows.append(
        {
            "check": "negative_relative_gap_rows",
            "status": (
                "documented_warning"
                if len(negative_gap) and negative_gap_documented
                else "needs_explanation"
                if len(negative_gap)
                else "ok"
            ),
            "message": (
                f"{len(negative_gap)} final-N rows have negative relative gap. "
                "Explain finite-horizon/transient noise or use clipped gap in figures."
            ),
        }
    )

    failed_beta = beta_df[
        ~np.isfinite(beta_df.get("convergence_beta", pd.Series(dtype=float)))
    ]
    beta_nan_documented = (
        "nan values indicate" in draft_text
        or "too few valid" in draft_text
    )
    rows.append(
        {
            "check": "beta_nan_rows",
            "status": (
                "documented_warning"
                if len(failed_beta) and beta_nan_documented
                else "needs_explanation"
                if len(failed_beta)
                else "ok"
            ),
            "message": (
                f"{len(failed_beta)} beta rows are NaN. Usually caused by "
                "non-positive gaps or too few valid fit points."
            ),
        }
    )

    duplicate_instances_present = []
    for group in DUPLICATE_INSTANCE_GROUPS.values():
        affected = [group["representative"]] + list(group["duplicates"])
        present = [
            name
            for name in affected
            if name in set(final_df["instance"].unique())
        ]
        if len(present) > 1:
            duplicate_instances_present.append(
                {
                    "representative": group["representative"],
                    "present": present,
                    "reason": group["reason"],
                }
            )

    duplicate_message = (
        "Duplicate instance groups are present in final-N outputs; paper-level "
        f"tables exclude {PAPER_EXCLUDED_INSTANCES}. Details: "
        f"{duplicate_instances_present}"
        if duplicate_instances_present
        else (
            "No duplicate instance group is double-counted in final-N outputs. "
            f"Paper-level tables still exclude {PAPER_EXCLUDED_INSTANCES} "
            "when reading older raw files."
        )
    )

    rows.append(
        {
            "check": "duplicate_instance_groups",
            "status": (
                "documented_warning"
                if duplicate_instances_present
                else "ok"
            ),
            "message": duplicate_message,
        }
    )

    required_columns = {"paper_eligible", "has_known_model_results"}
    if not taxonomy_df.empty and required_columns.issubset(taxonomy_df.columns):
        result_instances = set(final_df["instance"].unique())
        missing_result_instances = [
            name
            for name in KNOWN_MODEL_BENCHMARK_INSTANCES
            if name not in result_instances
        ]
        rows.append(
            {
                "check": "default_known_model_instances_without_results",
                "status": (
                    "documented_warning"
                    if missing_result_instances
                    else "ok"
                ),
                "message": (
                    "These default known-model benchmark instances do not "
                    "currently appear in the known-model CSV files: "
                    f"{missing_result_instances}"
                ),
            }
        )

    return rows


def unknown_model_checks():
    rows = []
    unknown_df = read_csv_if_exists(REQUIRED_FILES["unknown_model_summary"])

    if unknown_df.empty:
        rows.append(
            {
                "check": "unknown_model_results",
                "status": "missing",
                "message": (
                    "Run run_unknown_model_benchmark.py before claiming "
                    "unknown-P,R experimental results."
                ),
            }
        )
        return rows

    policies = sorted(unknown_df["policy"].unique())
    rows.append(
        {
            "check": "unknown_model_policy_coverage",
            "status": "ok" if "OnlinePlugInWhittle" in policies else "missing",
            "message": f"Unknown-model policies found: {policies}",
        }
    )
    return rows


def heterogeneous_checks():
    rows = []
    hetero_df = read_csv_if_exists(REQUIRED_FILES["heterogeneous_summary"])

    if hetero_df.empty:
        rows.append(
            {
                "check": "heterogeneous_results",
                "status": "missing",
                "message": "Run run_heterogeneous_benchmark.py.",
            }
        )
        return rows

    policies = sorted(hetero_df["policy"].unique())
    missing = [
        policy
        for policy in ["HeterogeneousLPPriority", "HeterogeneousFTVA"]
        if policy not in policies
    ]
    rows.append(
        {
            "check": "heterogeneous_policy_coverage",
            "status": "missing" if missing else "ok",
            "message": (
                f"Policies found: {policies}. Missing important policies: {missing}"
            ),
        }
    )
    return rows


def check_paper_readiness(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_df = pd.DataFrame(file_status_rows())
    checks_df = pd.DataFrame(
        known_model_checks()
        + unknown_model_checks()
        + heterogeneous_checks()
    )

    files_df.to_csv(output_dir / "paper_readiness_files.csv", index=False)
    checks_df.to_csv(output_dir / "paper_readiness_checks.csv", index=False)

    lines = [
        "# Paper Readiness Check",
        "",
        "## File Status",
        "",
        files_df.to_markdown(index=False),
        "",
        "## Interpretation Checks",
        "",
        checks_df.to_markdown(index=False),
        "",
        "## Recommended next actions",
        "",
    ]

    if (checks_df["status"] == "missing").any():
        lines.append("- Run the missing experiment scripts listed above.")
    if (checks_df["status"] == "needs_explanation").any():
        lines.append(
            "- Add a short metrics caveat explaining finite-horizon negative gaps and NaN beta fits."
        )
    if (checks_df["status"] == "documented_warning").any():
        lines.append(
            "- Caveats are present in the draft; keep them in the final paper."
        )
    if not (checks_df["status"].isin(["missing", "needs_explanation"])).any():
        lines.append("- Outputs are sufficient for a first paper/report draft.")

    (output_dir / "paper_readiness_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved paper readiness report to:\n{output_dir}")
    return {
        "files": files_df,
        "checks": checks_df,
    }


if __name__ == "__main__":
    READINESS = check_paper_readiness()
