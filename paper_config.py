"""
Shared paper/report configuration.

This file contains choices that affect presentation but do not change the raw
benchmark data. Raw CSV files can still contain every instance. Paper-level
tables and figures should use these filters to avoid double-counting duplicate
examples.
"""

KNOWN_MODEL_BENCHMARK_INSTANCES = [
    "random_S10_seed123",
    "hong_counterexample",
    "yan_gast_example1",
    "yan_gast_example2",
    "yan_gast_example3",
    "conveyor_eg4unif-tb_S8",
    "conveyor_eg4archive1_S8",
    "conveyor_eg4archive2_S8",
    "maintenance_S10_a20",
    "wireless_channel_S10_a40",
    "deadline_S10_a20",
]


# Only the conveyor action-gap instance is removed: it has the same P,R,alpha
# as hong_counterexample. Non-duplicate conveyor examples remain available for
# the benchmark and paper-level analysis.
DUPLICATE_INSTANCE_GROUPS = {
    "hong_conveyor_duplicate": {
        "representative": "hong_counterexample",
        "duplicates": ["conveyor_eg4action-gap-tb_S8"],
        "reason": (
            "Identical P,R,alpha to hong_counterexample in the current "
            "finite-state construction."
        ),
    },
    "yan_gast_example1_duplicate": {
        "representative": "yan_gast_example1",
        "duplicates": ["gast20_example1"],
        "reason": (
            "Identical P,R,alpha to yan_gast_example1 in the current "
            "finite-state construction."
        ),
    },
    "yan_gast_example3_duplicate": {
        "representative": "yan_gast_example3",
        "duplicates": ["gast20_example3"],
        "reason": (
            "Identical P,R,alpha to yan_gast_example3 in the current "
            "finite-state construction."
        ),
    }
}


PAPER_EXCLUDED_INSTANCES = [
    duplicate
    for group in DUPLICATE_INSTANCE_GROUPS.values()
    for duplicate in group["duplicates"]
]

PAPER_EXCLUDED_INSTANCE_PREFIXES = []


def filter_paper_instances(df, instance_column="instance"):
    """Remove excluded instances from paper-level tables/figures."""
    if df is None or df.empty or instance_column not in df.columns:
        return df
    instance_values = df[instance_column].astype(str)
    keep_mask = ~instance_values.isin(PAPER_EXCLUDED_INSTANCES)
    for prefix in PAPER_EXCLUDED_INSTANCE_PREFIXES:
        keep_mask &= ~instance_values.str.startswith(prefix)
    return df[keep_mask].copy()
