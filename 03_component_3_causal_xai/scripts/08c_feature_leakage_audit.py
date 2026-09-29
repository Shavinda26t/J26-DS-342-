from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "feature_leakage_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

HORIZONS = [7, 14, 30]

WINDOWS = [7, 14, 30]

SMART_FEATURES = [
    "smart_1_raw",
    "smart_4_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_12_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_192_raw",
    "smart_193_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_199_raw",
    "smart_240_raw",
    "smart_241_raw",
    "smart_242_raw",
]


# ==========================================================
# CREATE APPROVED MODEL FEATURE LIST
# ==========================================================

MODEL_FEATURES = []


# Current SMART values
MODEL_FEATURES.extend(
    SMART_FEATURES
)


# 1-day changes
for feature in SMART_FEATURES:

    MODEL_FEATURES.append(
        f"{feature}_diff_1d"
    )


# Temporal summaries
for window in WINDOWS:

    for feature in SMART_FEATURES:

        MODEL_FEATURES.append(
            f"{feature}_mean_{window}d"
        )

        MODEL_FEATURES.append(
            f"{feature}_std_{window}d"
        )

        MODEL_FEATURES.append(
            f"{feature}_max_{window}d"
        )


# ==========================================================
# FORBIDDEN / NON-MODEL COLUMNS
# ==========================================================

FORBIDDEN_EXACT = {
    "date",
    "serial_number",
    "model",
    "failure",

    "failure_date",
    "days_to_failure",
    "future_followup_days",
    "eligible_for_prediction",

    "failure_within_7d",
    "failure_within_14d",
    "failure_within_30d",

    "capacity_bytes",

    "datacenter",
    "cluster_id",
    "vault_id",
    "pod_id",
    "pod_slot_num",
    "is_legacy_format",

    "days_since_previous_record",

    "observations_7d",
    "observations_14d",
    "observations_30d",

    "has_sufficient_7d_history",
    "has_sufficient_14d_history",
    "has_sufficient_30d_history",
}


# Known exact duplicate SMART variables
DUPLICATE_SMARTS = {
    "smart_195_raw",
    "smart_190_raw",
    "smart_198_raw",
}


# ==========================================================
# BASIC MANIFEST
# ==========================================================

manifest_df = pd.DataFrame(
    {
        "feature": MODEL_FEATURES,
        "approved_for_reference_model": True
    }
)


manifest_file = (
    OUTPUT_DIR
    / "model_feature_manifest.csv"
)

manifest_df.to_csv(
    manifest_file,
    index=False
)


# ==========================================================
# CHECK EACH SPLIT SCHEMA
# ==========================================================

audit_rows = []

all_good = True


for horizon in HORIZONS:

    for split in [
        "train",
        "validation",
        "test"
    ]:

        file = (
            SPLIT_DIR
            / f"{horizon}d"
            / f"{split}.parquet"
        )

        if not file.exists():

            raise FileNotFoundError(
                f"Missing split file: {file}"
            )


        # Read only Parquet schema.
        # This avoids loading millions of rows into RAM.
        schema = (
            pq.ParquetFile(
                file
            )
            .schema
            .names
        )

        columns = set(
            schema
        )


        # ----------------------------------------------
        # APPROVED FEATURES PRESENT?
        # ----------------------------------------------

        missing_model_features = sorted(
            set(MODEL_FEATURES)
            - columns
        )


        # ----------------------------------------------
        # DUPLICATE SMARTS PRESENT IN DATA?
        #
        # It is okay for them to exist in the stored
        # dataset. They simply must NOT appear in the
        # model feature manifest.
        # ----------------------------------------------

        duplicate_smarts_in_dataset = sorted(
            DUPLICATE_SMARTS
            & columns
        )


        duplicate_smarts_in_model = sorted(
            DUPLICATE_SMARTS
            & set(MODEL_FEATURES)
        )


        # ----------------------------------------------
        # FORBIDDEN COLUMNS ACCIDENTALLY APPROVED?
        # ----------------------------------------------

        forbidden_in_model = sorted(
            FORBIDDEN_EXACT
            & set(MODEL_FEATURES)
        )


        # ----------------------------------------------
        # TARGET CHECK
        # ----------------------------------------------

        target = (
            f"failure_within_{horizon}d"
        )

        target_exists = (
            target in columns
        )

        target_in_model = (
            target in MODEL_FEATURES
        )


        # ----------------------------------------------
        # OTHER POTENTIALLY SUSPICIOUS MODEL FEATURES
        # ----------------------------------------------

        suspicious_model_features = []

        suspicious_terms = [
            "failure",
            "future",
            "target",
            "label",
            "serial",
            "date",
            "eligible",
            "followup",
            "days_to_failure",
        ]


        for feature in MODEL_FEATURES:

            lowered = (
                feature.lower()
            )

            if any(
                term in lowered
                for term in suspicious_terms
            ):

                suspicious_model_features.append(
                    feature
                )


        passed = (
            len(
                missing_model_features
            ) == 0

            and len(
                duplicate_smarts_in_model
            ) == 0

            and len(
                forbidden_in_model
            ) == 0

            and len(
                suspicious_model_features
            ) == 0

            and target_exists

            and not target_in_model
        )


        if not passed:

            all_good = False


        audit_rows.append(
            {
                "horizon_days":
                    horizon,

                "split":
                    split,

                "dataset_column_count":
                    len(columns),

                "approved_model_feature_count":
                    len(MODEL_FEATURES),

                "missing_model_feature_count":
                    len(
                        missing_model_features
                    ),

                "missing_model_features":
                    ";".join(
                        missing_model_features
                    ),

                "duplicate_smarts_in_dataset":
                    ";".join(
                        duplicate_smarts_in_dataset
                    ),

                "duplicate_smarts_in_model_count":
                    len(
                        duplicate_smarts_in_model
                    ),

                "forbidden_columns_in_model_count":
                    len(
                        forbidden_in_model
                    ),

                "suspicious_model_feature_count":
                    len(
                        suspicious_model_features
                    ),

                "target_column_exists":
                    target_exists,

                "target_in_model_features":
                    target_in_model,

                "leakage_check_passed":
                    passed,
            }
        )


# ==========================================================
# SAVE AUDIT
# ==========================================================

audit_df = pd.DataFrame(
    audit_rows
)


audit_file = (
    OUTPUT_DIR
    / "feature_leakage_audit.csv"
)

audit_df.to_csv(
    audit_file,
    index=False
)


# ==========================================================
# PRINT MODEL FEATURE GROUP COUNTS
# ==========================================================

raw_count = len(
    SMART_FEATURES
)

diff_count = len(
    SMART_FEATURES
)

temporal_count = (
    len(SMART_FEATURES)
    * len(WINDOWS)
    * 3
)


print("\n")
print("=" * 70)
print("FEATURE LEAKAGE AUDIT")
print("=" * 70)

print(
    "Raw SMART features:",
    raw_count
)

print(
    "1-day difference features:",
    diff_count
)

print(
    "Rolling temporal features:",
    temporal_count
)

print(
    "Total approved model features:",
    len(MODEL_FEATURES)
)


print(
    "\nApproved duplicate SMART features:"
)

approved_duplicates = (
    DUPLICATE_SMARTS
    & set(MODEL_FEATURES)
)

if approved_duplicates:

    for feature in sorted(
        approved_duplicates
    ):

        print(
            "WARNING:",
            feature
        )

else:

    print(
        "None"
    )


print(
    "\nSplit audit:\n"
)

print(
    audit_df[
        [
            "horizon_days",
            "split",
            "dataset_column_count",
            "approved_model_feature_count",
            "missing_model_feature_count",
            "duplicate_smarts_in_model_count",
            "forbidden_columns_in_model_count",
            "suspicious_model_feature_count",
            "target_column_exists",
            "target_in_model_features",
            "leakage_check_passed",
        ]
    ].to_string(
        index=False
    )
)


print("\n")

if all_good:

    print(
        "RESULT: PASS"
    )

    print(
        "No target/future-information columns are "
        "included in the approved model feature list."
    )

else:

    print(
        "RESULT: FAIL"
    )

    print(
        "Do NOT train a model until the leakage "
        "problems above are corrected."
    )


print(
    "\nFeature manifest saved to:"
)

print(
    manifest_file
)

print(
    "\nLeakage audit saved to:"
)

print(
    audit_file
)