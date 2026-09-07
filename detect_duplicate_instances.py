"""
Detect duplicate RMAB instances in the current library.

Two instances are treated as duplicates if they have the same bandit hash and
the same default alpha. This is stricter than having similar-looking results:
it means the P,R model and activation budget are exactly the same under the
current construction.
"""

import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import bandit_lp
from known_model_extra_instances import build_extended_known_model_instance_library


OUTPUT_DIR = CURRENT_DIR / "paper_summary_outputs"


def detect_duplicate_instances(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    library = build_extended_known_model_instance_library(
        bandit_lp,
        include_original=True,
    )
    rows = []
    for name, spec in library.items():
        rows.append(
            {
                "instance": name,
                "S": spec.bandit.S,
                "A": spec.bandit.A,
                "alpha": spec.default_alpha,
                "bandit_hash": spec.bandit.hashname(),
                "source": spec.source,
            }
        )

    df = pd.DataFrame(rows)
    df["duplicate_key"] = (
        df["bandit_hash"].astype(str)
        + "_alpha_"
        + df["alpha"].round(12).astype(str)
    )
    duplicate_df = df[
        df.duplicated("duplicate_key", keep=False)
    ].sort_values(["duplicate_key", "instance"])

    df.to_csv(output_dir / "all_instance_hashes.csv", index=False)
    duplicate_df.to_csv(output_dir / "duplicate_instances.csv", index=False)

    print("Duplicate instances:")
    if duplicate_df.empty:
        print("  None")
    else:
        print(duplicate_df[["instance", "S", "A", "alpha", "bandit_hash"]])
    print(f"\nSaved duplicate-instance check to:\n{output_dir}")
    return {
        "all_instances": df,
        "duplicates": duplicate_df,
    }


if __name__ == "__main__":
    DUPLICATES = detect_duplicate_instances()
