from pathlib import Path
import pandas as pd


# ============================================================
# Component 2 - Digital Twin
# Health-State Candidate Evaluation
# ============================================================

RESULTS_DIR = Path("results")

MISSINGNESS_FILE = RESULTS_DIR / "feature_missingness.csv"
EFFECT_SIZE_FILE = RESULTS_DIR / "failure_feature_effect_size.csv"
REDUNDANCY_FILE = RESULTS_DIR / "high_redundancy_pairs_095.csv"

OUTPUT_FILE = RESULTS_DIR / "health_state_candidate_evaluation.csv"


# Core SMART attributes identified during the previous
# data-quality and degradation analyses.
SMART_ATTRIBUTES = [1, 5, 187, 197, 198]


def load_file(path):
    """Load a CSV file and stop with a clear message if missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required analysis file not found: {path}"
        )

    return pd.read_csv(path)


def smart_id_from_feature(feature):
    """Extract SMART ID from a feature name."""

    text = str(feature)

    if text.startswith("smart_"):
        parts = text.split("_")

        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return None

    return None


def feature_type(feature):
    """Identify the temporal representation of a feature."""

    feature = str(feature)

    if feature.endswith("_rolling_std_7"):
        return "rolling_std_7"

    if feature.endswith("_rolling_mean_7"):
        return "rolling_mean_7"

    if feature.endswith("_change_14d"):
        return "change_14d"

    if feature.endswith("_change_7d"):
        return "change_7d"

    if feature.endswith("_delta"):
        return "delta"

    if feature.endswith("_previous"):
        return "previous"

    return "current"


def main():

    print("=" * 70)
    print("COMPONENT 2 - HEALTH-STATE CANDIDATE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load previous analysis results
    # --------------------------------------------------------

    missingness = load_file(MISSINGNESS_FILE)
    effect_size = load_file(EFFECT_SIZE_FILE)
    redundancy = load_file(REDUNDANCY_FILE)

    print("\nLoaded previous analysis results:")
    print(f"  Missingness: {MISSINGNESS_FILE}")
    print(f"  Effect size: {EFFECT_SIZE_FILE}")
    print(f"  Redundancy: {REDUNDANCY_FILE}")

    # --------------------------------------------------------
    # Standardise column names
    # --------------------------------------------------------

    # The project analysis files use 'feature' as the feature
    # identifier. We check this explicitly rather than silently
    # assuming another structure.

    if "feature" not in missingness.columns:
        raise ValueError(
            "feature_missingness.csv does not contain "
            "a 'feature' column."
        )

    if "feature" not in effect_size.columns:
        raise ValueError(
            "failure_feature_effect_size.csv does not contain "
            "a 'feature' column."
        )

    # Find the effect-size column.
    effect_candidates = [
        "effect_size",
        "cohens_d",
        "absolute_effect_size",
        "effect",
    ]

    effect_column = None

    for column in effect_candidates:

        if column in effect_size.columns:
            effect_column = column
            break

    if effect_column is None:

        print("\nAvailable effect-size columns:")
        print(effect_size.columns.tolist())

        raise ValueError(
            "Could not identify the effect-size column."
        )

    # Find missingness column.
    missing_candidates = [
        "missing_percentage",
        "missing_percent",
        "missingness",
    ]

    missing_column = None

    for column in missing_candidates:

        if column in missingness.columns:
            missing_column = column
            break

    if missing_column is None:

        print("\nAvailable missingness columns:")
        print(missingness.columns.tolist())

        raise ValueError(
            "Could not identify the missingness column."
        )

    # --------------------------------------------------------
    # Restrict analysis to the five core SMART attributes
    # --------------------------------------------------------

    missingness["smart_id"] = (
        missingness["feature"]
        .apply(smart_id_from_feature)
    )

    effect_size["smart_id"] = (
        effect_size["feature"]
        .apply(smart_id_from_feature)
    )

    missingness_core = missingness[
        missingness["smart_id"].isin(SMART_ATTRIBUTES)
    ].copy()

    effect_core = effect_size[
        effect_size["smart_id"].isin(SMART_ATTRIBUTES)
    ].copy()

    # --------------------------------------------------------
    # Build feature-level evaluation
    # --------------------------------------------------------

    evaluation = missingness_core[
        [
            "feature",
            "smart_id",
            missing_column,
        ]
    ].copy()

    evaluation = evaluation.rename(
        columns={
            missing_column: "missing_percentage"
        }
    )

    effect_subset = effect_core[
        [
            "feature",
            effect_column,
        ]
    ].copy()

    effect_subset = effect_subset.rename(
        columns={
            effect_column: "effect_size"
        }
    )

    evaluation = evaluation.merge(
        effect_subset,
        on="feature",
        how="left"
    )

    evaluation["feature_type"] = (
        evaluation["feature"]
        .apply(feature_type)
    )

    # --------------------------------------------------------
    # Availability score
    # --------------------------------------------------------

    evaluation["availability_score"] = (
        1 - evaluation["missing_percentage"] / 100
    )

    # --------------------------------------------------------
    # Effect-size strength
    # --------------------------------------------------------

    max_effect = evaluation["effect_size"].abs().max()

    if pd.notna(max_effect) and max_effect > 0:

        evaluation["relative_effect_strength"] = (
            evaluation["effect_size"].abs()
            / max_effect
        )

    else:

        evaluation["relative_effect_strength"] = 0.0

    # --------------------------------------------------------
    # Redundancy analysis
    # --------------------------------------------------------

    evaluation["redundant_with_other_feature"] = False
    evaluation["redundancy_count"] = 0

    if not redundancy.empty:

        required_columns = {
            "feature_a",
            "feature_b",
        }

        if required_columns.issubset(
            redundancy.columns
        ):

            for index, row in evaluation.iterrows():

                feature = row["feature"]

                count = (
                    (
                        redundancy["feature_a"]
                        == feature
                    )
                    |
                    (
                        redundancy["feature_b"]
                        == feature
                    )
                ).sum()

                evaluation.loc[
                    index,
                    "redundancy_count"
                ] = int(count)

                evaluation.loc[
                    index,
                    "redundant_with_other_feature"
                ] = count > 0

    # --------------------------------------------------------
    # Screening classification
    # --------------------------------------------------------

    def classify(row):

        missing = row["missing_percentage"]
        effect = abs(row["effect_size"])
        redundancy_count = row["redundancy_count"]

        # Strong evidence:
        # high effect size and acceptable availability.
        if (
            effect >= 0.8
            and missing <= 60
        ):
            return "Strong candidate"

        # Good availability but weaker failure association.
        if (
            missing <= 10
            and effect >= 0.2
        ):
            return "Candidate"

        # Strong failure signal but high missingness.
        if (
            effect >= 0.8
            and missing > 60
        ):
            return "Candidate - missingness concern"

        # High redundancy should not automatically eliminate
        # a feature, but should trigger comparison with its
        # redundant alternatives.
        if redundancy_count > 0:
            return "Review redundancy"

        return "Lower priority"

    evaluation["screening_decision"] = (
        evaluation.apply(
            classify,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    evaluation = evaluation.sort_values(
        [
            "screening_decision",
            "effect_size",
            "missing_percentage",
        ],
        ascending=[
            True,
            False,
            True,
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    evaluation.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CANDIDATE EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"\nOutput: {OUTPUT_FILE}"
    )

    print(
        f"\nRows evaluated: {len(evaluation):,}"
    )

    print("\nEvaluation summary:")

    print(
        evaluation[
            [
                "feature",
                "missing_percentage",
                "effect_size",
                "redundancy_count",
                "screening_decision",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("SMART-LEVEL SUMMARY")
    print("=" * 70)

    smart_summary = (
        evaluation
        .groupby("smart_id")
        .agg(
            features=("feature", "count"),
            best_effect_size=(
                "effect_size",
                lambda x: x.abs().max()
            ),
            minimum_missingness=(
                "missing_percentage",
                "min"
            ),
            redundant_features=(
                "redundant_with_other_feature",
                "sum"
            ),
        )
        .reset_index()
        .sort_values(
            "best_effect_size",
            ascending=False
        )
    )

    print(
        smart_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()