import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"D:\Project\J26-DS-342-\02_component_2_digital_twin"
)

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "digital_twin_temporal_state.csv"
)

OUTPUT_DIR = BASE_DIR / "results"

# ============================================================
# START
# ============================================================

print("=" * 70)
print("DIGITAL TWIN FEATURE REDUNDANCY ANALYSIS")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading temporal state dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Unique HDDs: {df['serial_number'].nunique():,}")

# ============================================================
# FEATURE GROUPS
# ============================================================

smart_features = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
]

temporal_features = []

for smart in [1, 5, 187, 197, 198]:

    prefix = f"smart_{smart}_normalized"

    temporal_features.extend([
        f"{prefix}_previous",
        f"{prefix}_delta",
        f"{prefix}_change_7d",
        f"{prefix}_change_14d",
        f"{prefix}_rolling_mean_7",
        f"{prefix}_rolling_std_7",
    ])

missing_features = [
    "smart_1_missing",
    "smart_5_missing",
    "smart_187_missing",
    "smart_197_missing",
    "smart_198_missing",
]

all_features = (
    smart_features
    + temporal_features
    + missing_features
)

# Keep only columns that exist
all_features = [
    column
    for column in all_features
    if column in df.columns
]

print(f"\nAnalysis features: {len(all_features)}")

# ============================================================
# 1. FEATURE-TO-FEATURE CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("1. FEATURE CORRELATION")
print("=" * 70)

print("\nCalculating correlation matrix...")

corr = df[all_features].corr()

corr.to_csv(
    OUTPUT_DIR / "feature_correlation_matrix.csv"
)

# ------------------------------------------------------------
# Find highly correlated feature pairs
# ------------------------------------------------------------

pairs = []

for i in range(len(all_features)):

    for j in range(i + 1, len(all_features)):

        feature_1 = all_features[i]
        feature_2 = all_features[j]

        value = corr.loc[
            feature_1,
            feature_2
        ]

        if pd.notna(value):

            pairs.append({
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": value,
                "absolute_correlation": abs(value)
            })

corr_pairs = pd.DataFrame(pairs)

corr_pairs = corr_pairs.sort_values(
    "absolute_correlation",
    ascending=False
)

corr_pairs.to_csv(
    OUTPUT_DIR / "high_correlation_feature_pairs.csv",
    index=False
)

print("\nTop 20 correlated feature pairs:")

print(
    corr_pairs.head(20).to_string(index=False)
)

# ============================================================
# 2. FAILURE-SPECIFIC CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("2. FAILURE-SPECIFIC FEATURE CORRELATION")
print("=" * 70)

failure_df = df[df["failure"] == 1]

print(
    f"\nFailure rows: {len(failure_df):,}"
)

print(
    f"Failed HDDs: "
    f"{failure_df['serial_number'].nunique():,}"
)

# ------------------------------------------------------------
# Failure correlation matrix
# ------------------------------------------------------------

failure_corr = failure_df[all_features].corr()

failure_corr.to_csv(
    OUTPUT_DIR
    / "failure_feature_correlation_matrix.csv"
)

# ------------------------------------------------------------
# Find highly correlated failure features
# ------------------------------------------------------------

failure_pairs = []

for i in range(len(all_features)):

    for j in range(i + 1, len(all_features)):

        feature_1 = all_features[i]
        feature_2 = all_features[j]

        value = failure_corr.loc[
            feature_1,
            feature_2
        ]

        if pd.notna(value):

            failure_pairs.append({
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": value,
                "absolute_correlation": abs(value)
            })

failure_pairs_df = pd.DataFrame(
    failure_pairs
)

failure_pairs_df = failure_pairs_df.sort_values(
    "absolute_correlation",
    ascending=False
)

failure_pairs_df.to_csv(
    OUTPUT_DIR
    / "failure_high_correlation_pairs.csv",
    index=False
)

print("\nTop 20 failure-state correlated pairs:")

print(
    failure_pairs_df.head(20).to_string(index=False)
)

# ============================================================
# 3. FEATURE VARIANCE
# ============================================================

print("\n" + "=" * 70)
print("3. FEATURE VARIANCE")
print("=" * 70)

print(
    "\nCalculating variance feature-by-feature "
    "to reduce memory usage..."
)

variance_results = []

for feature in all_features:

    print(
        f"  Calculating variance: {feature}"
    )

    series = df[feature]

    variance_value = series.var()

    variance_results.append({
        "feature": feature,
        "variance": variance_value
    })

variance_df = pd.DataFrame(
    variance_results
)

variance_df = variance_df.sort_values(
    "variance",
    ascending=False
)

variance_df.to_csv(
    OUTPUT_DIR / "feature_variance.csv",
    index=False
)

print("\nLowest variance features:")

print(
    variance_df.tail(15).to_string(index=False)
)

# ============================================================
# 4. FAILURE VS NON-FAILURE DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("4. FAILURE VS NON-FAILURE FEATURE DIFFERENCE")
print("=" * 70)

failed = df[df["failure"] == 1][all_features]

normal = df[df["failure"] == 0][all_features]

comparison = []

for feature in all_features:

    failure_mean = failed[feature].mean()
    normal_mean = normal[feature].mean()

    failure_std = failed[feature].std()
    normal_std = normal[feature].std()

    mean_difference = (
        failure_mean - normal_mean
    )

    # Pooled standard deviation
    pooled_std = np.sqrt(
        (
            failure_std ** 2
            + normal_std ** 2
        ) / 2
    )

    if pooled_std != 0:

        effect_size = (
            mean_difference
            / pooled_std
        )

    else:

        effect_size = np.nan

    comparison.append({
        "feature": feature,
        "failure_mean": failure_mean,
        "normal_mean": normal_mean,
        "mean_difference": mean_difference,
        "failure_std": failure_std,
        "normal_std": normal_std,
        "effect_size": effect_size,
        "absolute_effect_size": (
            abs(effect_size)
            if pd.notna(effect_size)
            else np.nan
        )
    })

comparison_df = pd.DataFrame(
    comparison
)

comparison_df = comparison_df.sort_values(
    "absolute_effect_size",
    ascending=False
)

comparison_df.to_csv(
    OUTPUT_DIR
    / "failure_feature_effect_size.csv",
    index=False
)

print(
    "\nTop 20 features by absolute effect size:"
)

print(
    comparison_df.head(20).to_string(
        index=False
    )
)

# ============================================================
# 5. FEATURE MISSINGNESS
# ============================================================

print("\n" + "=" * 70)
print("5. FEATURE MISSINGNESS")
print("=" * 70)

missing_summary = []

for feature in all_features:

    missing_count = df[feature].isna().sum()

    missing_percentage = (
        missing_count
        / len(df)
        * 100
    )

    missing_summary.append({
        "feature": feature,
        "missing_count": missing_count,
        "missing_percentage": (
            missing_percentage
        )
    })

missing_df = pd.DataFrame(
    missing_summary
)

missing_df = missing_df.sort_values(
    "missing_percentage",
    ascending=False
)

missing_df.to_csv(
    OUTPUT_DIR / "feature_missingness.csv",
    index=False
)

print(
    missing_df.head(15).to_string(
        index=False
    )
)

# ============================================================
# 6. SMART-LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("6. SMART-LEVEL SUMMARY")
print("=" * 70)

smart_summary = []

for smart in [1, 5, 187, 197, 198]:

    current = (
        f"smart_{smart}_normalized"
    )

    change_7d = (
        f"smart_{smart}_normalized_change_7d"
    )

    change_14d = (
        f"smart_{smart}_normalized_change_14d"
    )

    rolling_std = (
        f"smart_{smart}_normalized_rolling_std_7"
    )

    missing = (
        f"smart_{smart}_missing"
    )

    if current not in df.columns:
        continue

    smart_summary.append({
        "SMART": smart,

        "current_mean":
            df[current].mean(),

        "current_std":
            df[current].std(),

        "change_7d_mean":
            df[change_7d].mean(),

        "change_14d_mean":
            df[change_14d].mean(),

        "rolling_std_mean":
            df[rolling_std].mean(),

        "missing_percentage":
            (
                df[missing].mean() * 100
                if missing in df.columns
                else np.nan
            )
    })

smart_summary_df = pd.DataFrame(
    smart_summary
)

smart_summary_df.to_csv(
    OUTPUT_DIR
    / "smart_level_feature_summary.csv",
    index=False
)

print(
    smart_summary_df.to_string(
        index=False
    )
)

# ============================================================
# 7. SUMMARY OF HIGHLY REDUNDANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("7. HIGH REDUNDANCY SUMMARY")
print("=" * 70)

high_corr = corr_pairs[
    corr_pairs["absolute_correlation"] >= 0.95
].copy()

print(
    f"\nFeature pairs with "
    f"|correlation| >= 0.95: "
    f"{len(high_corr):,}"
)

print("\nTop highly redundant pairs:")

print(
    high_corr.head(30).to_string(
        index=False
    )
)

high_corr.to_csv(
    OUTPUT_DIR
    / "high_redundancy_pairs_095.csv",
    index=False
)

# ============================================================
# 8. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nCreated files:")

output_files = [
    "feature_correlation_matrix.csv",
    "high_correlation_feature_pairs.csv",
    "failure_feature_correlation_matrix.csv",
    "failure_high_correlation_pairs.csv",
    "feature_variance.csv",
    "failure_feature_effect_size.csv",
    "feature_missingness.csv",
    "smart_level_feature_summary.csv",
    "high_redundancy_pairs_095.csv",
]

for file_name in output_files:

    print(
        f"  results\\{file_name}"
    )

print("\n" + "=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "\nUse the correlation, failure-specific correlation,"
    "\neffect size, variance, and missingness results"
    "\nto define the final Digital Twin state vector."
)

print("\nDo NOT start model training yet.")
print("=" * 70)