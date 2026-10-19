from pathlib import Path
import pandas as pd


# ============================================================
# Component 2 - Digital Twin
# Health-State Feature Selection
# ============================================================

INPUT_FILE = Path(
    "results/digital_twin_temporal_state.csv"
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "health_state_feature_summary.csv"
SELECTED_FILE = OUTPUT_DIR / "selected_health_state_features.txt"


# Candidate SMART attributes identified from previous
# data-quality and degradation analyses.
SMART_ATTRIBUTES = [1, 5, 187, 197, 198]

# Temporal representations to evaluate.
FEATURE_SUFFIXES = [
    "",
    "_delta",
    "_change_7d",
    "_change_14d",
    "_rolling_mean_7",
    "_rolling_std_7",
]


def main():

    print("=" * 70)
    print("COMPONENT 2 - HEALTH-STATE FEATURE SELECTION")
    print("=" * 70)

    print(f"\nInput: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # Only read columns required for this analysis.
    candidate_columns = []

    for smart_id in SMART_ATTRIBUTES:
        base = f"smart_{smart_id}_normalized"

        for suffix in FEATURE_SUFFIXES:
            column = f"{base}{suffix}"

            if suffix == "":
                column = base

            candidate_columns.append(column)

    # Find which candidate columns actually exist.
    available_columns = [
        column
        for column in candidate_columns
        if column in pd.read_csv(
            INPUT_FILE,
            nrows=0
        ).columns
    ]

    print(
        f"\nCandidate features: {len(candidate_columns)}"
    )

    print(
        f"Available features: {len(available_columns)}"
    )

    if not available_columns:
        raise ValueError(
            "No candidate health-state features were found."
        )

    print("\nLoading candidate features...")

    data = pd.read_csv(
        INPUT_FILE,
        usecols=available_columns
    )

    print(
        f"Rows loaded: {len(data):,}"
    )

    # --------------------------------------------------------
    # Feature summary
    # --------------------------------------------------------

    summary_rows = []

    for column in available_columns:

        values = data[column]

        missing_percentage = (
            values.isna().mean() * 100
        )

        non_zero_percentage = (
            (values.fillna(0) != 0).mean() * 100
        )

        summary_rows.append(
            {
                "feature": column,
                "missing_percentage":
                    missing_percentage,
                "non_zero_percentage":
                    non_zero_percentage,
                "mean":
                    values.mean(),
                "std":
                    values.std(),
                "absolute_mean":
                    values.abs().mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # --------------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------------

    print("\nCalculating feature correlations...")

    correlation = data.corr(
        numeric_only=True
    ).abs()

    redundant_pairs = []

    threshold = 0.95

    for i in range(len(correlation.columns)):

        for j in range(i + 1, len(correlation.columns)):

            feature_a = correlation.columns[i]
            feature_b = correlation.columns[j]

            value = correlation.iloc[i, j]

            if pd.notna(value) and value >= threshold:

                redundant_pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "absolute_correlation": value,
                    }
                )

    redundant_pairs_df = pd.DataFrame(
        redundant_pairs
    )

    # --------------------------------------------------------
    # Candidate ranking
    # --------------------------------------------------------

    # Higher absolute mean change can indicate stronger
    # temporal variation, but this is only a screening signal.
    #
    # Missingness is penalised because a health-state feature
    # should ideally be available for most HDD observations.

    summary["availability_score"] = (
        1 - summary["missing_percentage"] / 100
    )

    summary["variation_score"] = (
        summary["absolute_mean"]
        / (
            summary["absolute_mean"].max()
            if summary["absolute_mean"].max() > 0
            else 1
        )
    )

    summary["screening_score"] = (
        0.6 * summary["variation_score"]
        + 0.4 * summary["availability_score"]
    )

    summary = summary.sort_values(
        "screening_score",
        ascending=False
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # For now, create a candidate list rather than claiming
    # these are the final model features.
    selected_candidates = summary[
        summary["missing_percentage"] <= 20
    ]["feature"].tolist()

    with open(
        SELECTED_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Component 2 Digital Twin\n"
            "Candidate Health-State Features\n"
            "================================\n\n"
        )

        file.write(
            "These are screening candidates only.\n"
            "They are NOT yet the final model features.\n\n"
        )

        for feature in selected_candidates:

            file.write(
                f"- {feature}\n"
            )

    print("\n" + "=" * 70)
    print("FEATURE SCREENING COMPLETED")
    print("=" * 70)

    print(
        f"\nSummary: {SUMMARY_FILE}"
    )

    print(
        f"Candidate list: {SELECTED_FILE}"
    )

    print(
        f"\nFeatures with <=20% missingness: "
        f"{len(selected_candidates)}"
    )

    print("\nTop screening candidates:")

    print(
        summary[
            [
                "feature",
                "missing_percentage",
                "screening_score",
            ]
        ].head(15).to_string(
            index=False
        )
    )

    if not redundant_pairs_df.empty:

        print(
            f"\nHighly correlated feature pairs "
            f"(>|{threshold}|): "
            f"{len(redundant_pairs_df)}"
        )

    else:

        print(
            "\nNo highly correlated feature pairs "
            "were detected."
        )


if __name__ == "__main__":
    main()