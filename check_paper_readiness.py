"""Audit corrected result provenance and coverage, not scientific publication readiness."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from report_data import CORRECTED_DATA, CORRECTED_REPORT, load_corrected_results


def check_paper_readiness(output_dir=CORRECTED_REPORT, *, known_model_dir=CORRECTED_DATA):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        summary, beta, manifest = load_corrected_results(known_model_dir)
    except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
        rows.append(dict(check="corrected_result_integrity", status="blocked", message=str(exc)))
    else:
        rows.extend([
            dict(check="corrected_result_integrity", status="ok",
                 message="Version, raw-run keys, protocol and reward summaries agree."),
            dict(check="default_paper_grid", status="ok" if manifest["paper_grid_complete"] else "pending",
                 message=f"Scope: {manifest['report_scope']}. Check report_inputs.json for the exact protocol."),
            dict(check="failed_runs", status="review" if manifest["failed_runs"] else "ok",
                 message=f"{manifest['failed_runs']} failed runs; incomplete groups are excluded from comparisons."),
            dict(check="negative_stationary_differences", status="interpretation_required",
                 message=f"{int(summary['mean_relative_gap'].lt(0).sum())} signed negative differences. "
                         "Examine transients and sampling error; do not clip them or claim an LP violation."),
            dict(check="beta_availability", status="interpretation_required",
                 message=f"{int((~np.isfinite(beta['convergence_beta'])).sum())} unavailable fits. "
                         "A missing beta is not evidence of a poor policy."),
        ])
    rows.extend([
        dict(check="corrected_timing_experiments", status="not_verified",
             message="No timing files are loaded by this check. Rerun and audit them separately."),
        dict(check="heterogeneous_and_learning_experiments", status="not_verified",
             message="No historical extensions are used to certify corrected results."),
        dict(check="manuscript_review", status="not_verified",
             message="This script cannot validate theoretical assumptions, citations or the final manuscript."),
    ])
    checks = pd.DataFrame(rows)
    checks.to_csv(output_dir / "paper_readiness_checks.csv", index=False)
    text = "# Corrected Result Audit\n\n" + checks.to_markdown(index=False)
    text += (
        "\n\nA successful integrity check only establishes internal data consistency. "
        "It does not certify a paper as ready for submission. Historical CSVs are not "
        "a substitute for rerunning an experiment affected by a simulator correction.\n"
    )
    (output_dir / "paper_readiness_report.md").write_text(text, encoding="utf-8")
    print(f"Saved result audit to:\n{output_dir}")
    return {"checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-model-dir", type=Path, default=CORRECTED_DATA)
    parser.add_argument("--output-dir", type=Path, default=CORRECTED_REPORT)
    args = parser.parse_args()
    result = check_paper_readiness(output_dir=args.output_dir, known_model_dir=args.known_model_dir)
    if result["checks"]["status"].eq("blocked").any():
        raise SystemExit("Corrected results failed the integrity check; see paper_readiness_report.md.")
