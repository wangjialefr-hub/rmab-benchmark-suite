"""
Export a clean taxonomy of all benchmark instances.

This file is for report/paper writing. It separates random synthetic examples,
literature/codebase counterexamples, and application-inspired synthetic models.
"""

import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import bandit_lp
from known_model_extra_instances import (
    build_extended_known_model_instance_library,
    extra_instance_metadata_dataframe,
)
from paper_config import (
    DUPLICATE_INSTANCE_GROUPS,
    KNOWN_MODEL_BENCHMARK_INSTANCES,
    filter_paper_instances,
)


OUTPUT_DIR = CURRENT_DIR / "paper_summary_outputs"
KNOWN_SUMMARY_PATH = CURRENT_DIR / "instance_matrix_outputs" / "summary_instance_matrix.csv"


def infer_source_type(row):
    group = str(row.get("group", "original"))
    instance = str(row["instance"])
    source = str(row.get("source", ""))

    if group == "random" or instance.startswith("random_"):
        return "synthetic_random"
    if group in {"maintenance", "wireless", "deadline"}:
        return "application_inspired_synthetic"
    if "Recovering" in source or instance.startswith("recovering_"):
        return "literature_inspired_benchmark"
    if (
        group == "hong_gast_ftva"
        or "YigeHong" in source
        or "Teacher code" in source
        or "counterexample" in instance
        or "yan_gast" in instance
        or "conveyor" in instance
        or "gast20" in instance
    ):
        return "literature_or_public_codebase_example"
    return "other"


def infer_structure(row):
    group = str(row.get("group", "original"))
    instance = str(row["instance"])

    if group == "random" or instance.startswith("random_"):
        return "dense_random_transition_reward"
    if "conveyor" in instance:
        return "cyclic_conveyor_counterexample"
    if "yan_gast" in instance or "gast20" in instance:
        return "small_dense_three_state_counterexample"
    if "hong" in instance:
        return "eight_state_counterexample"
    if "maintenance" in instance:
        return "degradation_and_repair"
    if "wireless" in instance:
        return "channel_quality_scheduling"
    if "deadline" in instance:
        return "deadline_urgency_scheduling"
    if "recovering" in instance:
        return "recovering_bandit_active_reset"
    return "other"


def infer_suggested_use(row):
    structure = row["structure"]
    if structure == "dense_random_transition_reward":
        return "robustness check across random P,R"
    if "counterexample" in structure or "conveyor" in structure:
        return "stress test for priority/index policies and finite-N behavior"
    if structure in {
        "degradation_and_repair",
        "channel_quality_scheduling",
        "deadline_urgency_scheduling",
        "recovering_bandit_active_reset",
    }:
        return "application-style benchmark with interpretable states"
    return "general benchmark"


def export_instance_taxonomy(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_library = build_extended_known_model_instance_library(
        bandit_lp,
        include_original=True,
    )
    taxonomy = pd.DataFrame(
        extra_instance_metadata_dataframe(instance_library)
    )

    taxonomy["source_type"] = taxonomy.apply(infer_source_type, axis=1)
    taxonomy["structure"] = taxonomy.apply(infer_structure, axis=1)
    taxonomy["homogeneity"] = "homogeneous_shared_P_R"
    taxonomy.loc[
        taxonomy["instance"].str.contains("maintenance|wireless", regex=True),
        "homogeneity",
    ] = "homogeneous_shared_P_R; heterogeneous_variant_available"
    taxonomy["model_knowledge_use"] = (
        "known P,R for planning benchmarks; hidden environment for unknown-model learning"
    )
    taxonomy["suggested_use"] = taxonomy.apply(infer_suggested_use, axis=1)
    eligible_instances = set(filter_paper_instances(taxonomy)["instance"])
    taxonomy["paper_eligible"] = taxonomy["instance"].isin(eligible_instances)
    taxonomy["default_known_model_run"] = taxonomy["instance"].isin(
        KNOWN_MODEL_BENCHMARK_INSTANCES
    )

    if KNOWN_SUMMARY_PATH.exists():
        result_instances = set(pd.read_csv(KNOWN_SUMMARY_PATH)["instance"])
    else:
        result_instances = set()
    taxonomy["has_known_model_results"] = taxonomy["instance"].isin(result_instances)
    taxonomy["paper_included"] = (
        taxonomy["paper_eligible"] & taxonomy["has_known_model_results"]
    )
    taxonomy["duplicate_group"] = ""
    taxonomy["duplicate_representative"] = ""
    taxonomy["duplicate_reason"] = ""

    for group_name, group in DUPLICATE_INSTANCE_GROUPS.items():
        representative = group["representative"]
        duplicate_instances = group["duplicates"]
        affected = [representative] + list(duplicate_instances)
        mask = taxonomy["instance"].isin(affected)
        taxonomy.loc[mask, "duplicate_group"] = group_name
        taxonomy.loc[mask, "duplicate_representative"] = representative
        taxonomy.loc[mask, "duplicate_reason"] = group["reason"]

    column_order = [
        "instance",
        "paper_eligible",
        "default_known_model_run",
        "has_known_model_results",
        "paper_included",
        "source_type",
        "group",
        "structure",
        "S",
        "A",
        "default_alpha",
        "homogeneity",
        "model_knowledge_use",
        "suggested_use",
        "duplicate_group",
        "duplicate_representative",
        "duplicate_reason",
        "source",
        "notes",
        "transition_summary",
        "reward_summary",
    ]
    taxonomy = taxonomy[
        [col for col in column_order if col in taxonomy.columns]
    ].sort_values(["source_type", "group", "instance"])

    taxonomy.to_csv(output_dir / "instance_taxonomy.csv", index=False)

    markdown_lines = [
        "# Instance Taxonomy",
        "",
        "This table is generated from the current instance library.",
        "",
    ]
    for source_type, group_df in taxonomy.groupby("source_type", sort=True):
        markdown_lines.append(f"## {source_type}")
        markdown_lines.append("")
        for _, row in group_df.iterrows():
            if row["paper_included"]:
                inclusion = "included"
            elif not row["paper_eligible"]:
                inclusion = "excluded duplicate"
            elif not row["default_known_model_run"]:
                inclusion = "library only"
            else:
                inclusion = "eligible, not yet run"
            markdown_lines.append(
                f"- **{row['instance']}**: S={row['S']}, A={row['A']}, "
                f"alpha={row['default_alpha']}; structure={row['structure']}; "
                f"use={row['suggested_use']}; paper status={inclusion}."
            )
        markdown_lines.append("")

    (output_dir / "instance_taxonomy.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    print(f"Saved instance taxonomy to:\n{output_dir}")
    return taxonomy


if __name__ == "__main__":
    TAXONOMY = export_instance_taxonomy()
