from pathlib import Path

import numpy as np
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

FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "feature_leakage_audit"
    / "model_feature_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "reference_model_readiness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

HORIZONS = [7, 14, 30]

SPLITS = [
    "train",
    "validation",
    "test"
]

BATCH_SIZE = 50000


# ==========================================================
# LOAD APPROVED FEATURES
# ==========================================================

manifest = pd.read_csv(
    FEATURE_MANIFEST
)

MODEL_FEATURES = (
    manifest.loc[
        manifest[
            "approved_for_reference_model"
        ] == True,
        "feature"
    ]
    .tolist()
)


print(
    "Approved model features:",
    len(MODEL_FEATURES)
)


# ==========================================================
# RESULT STORAGE
# ==========================================================

split_summary_rows = []

feature_quality_rows = []


# ==========================================================
# PROCESS HORIZONS
# ==========================================================

for horizon in HORIZONS:

    target = (
        f"failure_within_{horizon}d"
    )

    print("\n")
    print("=" * 70)
    print(
        f"CHECKING {horizon}-DAY DATASET"
    )
    print("=" * 70)


    for split in SPLITS:

        file = (
            SPLIT_DIR
            / f"{horizon}d"
            / f"{split}.parquet"
        )

        if not file.exists():

            raise FileNotFoundError(
                f"Missing split: {file}"
            )


        parquet_file = (
            pq.ParquetFile(file)
        )


        # --------------------------------------------------
        # COUNTERS
        # --------------------------------------------------

        total_rows = 0

        positive_rows = 0
        negative_rows = 0

        missing_counts = {
            feature: 0
            for feature in MODEL_FEATURES
        }

        inf_counts = {
            feature: 0
            for feature in MODEL_FEATURES
        }

        min_values = {
            feature: np.inf
            for feature in MODEL_FEATURES
        }

        max_values = {
            feature: -np.inf
            for feature in MODEL_FEATURES
        }


        # --------------------------------------------------
        # STREAM PARQUET IN BATCHES
        # --------------------------------------------------

        columns_to_read = (
            MODEL_FEATURES
            + [target]
        )


        for batch_number, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=BATCH_SIZE,
                columns=columns_to_read
            ),
            start=1
        ):

            df = batch.to_pandas()

            total_rows += len(df)


            # ----------------------------------------------
            # TARGET COUNTS
            # ----------------------------------------------

            positive_rows += int(
                (
                    df[target] == 1
                ).sum()
            )

            negative_rows += int(
                (
                    df[target] == 0
                ).sum()
            )


            # ----------------------------------------------
            # FEATURE QUALITY
            # ----------------------------------------------

            for feature in MODEL_FEATURES:

                values = pd.to_numeric(
                    df[feature],
                    errors="coerce"
                ).to_numpy(
                    dtype="float64"
                )


                missing_mask = (
                    np.isnan(values)
                )

                inf_mask = (
                    np.isinf(values)
                )


                missing_counts[
                    feature
                ] += int(
                    missing_mask.sum()
                )


                inf_counts[
                    feature
                ] += int(
                    inf_mask.sum()
                )


                finite_values = values[
                    np.isfinite(values)
                ]


                if len(
                    finite_values
                ) > 0:

                    batch_min = (
                        finite_values.min()
                    )

                    batch_max = (
                        finite_values.max()
                    )


                    min_values[
                        feature
                    ] = min(
                        min_values[
                            feature
                        ],
                        batch_min
                    )


                    max_values[
                        feature
                    ] = max(
                        max_values[
                            feature
                        ],
                        batch_max
                    )


            if (
                batch_number % 5
                == 0
            ):

                print(
                    f"{split}: "
                    f"processed "
                    f"{total_rows:,} rows"
                )


        # ==================================================
        # CLASS IMBALANCE
        # ==================================================

        scale_pos_weight = (
            negative_rows
            / positive_rows
            if positive_rows > 0
            else np.nan
        )


        positive_rate = (
            positive_rows
            / total_rows
            * 100
            if total_rows > 0
            else 0
        )


        split_summary_rows.append(
            {
                "horizon_days":
                    horizon,

                "split":
                    split,

                "rows":
                    total_rows,

                "positive_rows":
                    positive_rows,

                "negative_rows":
                    negative_rows,

                "positive_rate_pct":
                    round(
                        positive_rate,
                        6
                    ),

                "scale_pos_weight":
                    round(
                        scale_pos_weight,
                        6
                    )
                    if not np.isnan(
                        scale_pos_weight
                    )
                    else np.nan
            }
        )


        # ==================================================
        # FEATURE QUALITY RESULTS
        # ==================================================

        for feature in MODEL_FEATURES:

            minimum = (
                min_values[
                    feature
                ]
            )

            maximum = (
                max_values[
                    feature
                ]
            )


            all_missing = (
                minimum == np.inf
                and maximum == -np.inf
            )


            constant_feature = (
                not all_missing
                and minimum == maximum
            )


            missing_pct = (
                missing_counts[
                    feature
                ]
                / total_rows
                * 100
            )


            feature_quality_rows.append(
                {
                    "horizon_days":
                        horizon,

                    "split":
                        split,

                    "feature":
                        feature,

                    "missing_rows":
                        missing_counts[
                            feature
                        ],

                    "missing_pct":
                        round(
                            missing_pct,
                            6
                        ),

                    "infinite_rows":
                        inf_counts[
                            feature
                        ],

                    "min_value":
                        (
                            np.nan
                            if all_missing
                            else minimum
                        ),

                    "max_value":
                        (
                            np.nan
                            if all_missing
                            else maximum
                        ),

                    "all_missing":
                        all_missing,

                    "constant_feature":
                        constant_feature,
                }
            )


# ==========================================================
# DATAFRAMES
# ==========================================================

split_summary_df = pd.DataFrame(
    split_summary_rows
)

feature_quality_df = pd.DataFrame(
    feature_quality_rows
)


# ==========================================================
# SAVE FULL RESULTS
# ==========================================================

split_summary_file = (
    OUTPUT_DIR
    / "reference_model_split_summary.csv"
)

feature_quality_file = (
    OUTPUT_DIR
    / "reference_model_feature_quality.csv"
)


split_summary_df.to_csv(
    split_summary_file,
    index=False
)

feature_quality_df.to_csv(
    feature_quality_file,
    index=False
)


# ==========================================================
# TRAIN-ONLY PROBLEM FEATURES
# ==========================================================

train_quality = (
    feature_quality_df[
        feature_quality_df[
            "split"
        ] == "train"
    ]
)


problem_features = (
    train_quality[
        (
            train_quality[
                "all_missing"
            ] == True
        )
        |
        (
            train_quality[
                "constant_feature"
            ] == True
        )
        |
        (
            train_quality[
                "infinite_rows"
            ] > 0
        )
    ]
)


problem_file = (
    OUTPUT_DIR
    / "problem_features.csv"
)

problem_features.to_csv(
    problem_file,
    index=False
)


# ==========================================================
# HIGH-MISSINGNESS FEATURES
# ==========================================================

high_missing_features = (
    feature_quality_df[
        feature_quality_df[
            "missing_pct"
        ] > 5.0
    ]
)


high_missing_file = (
    OUTPUT_DIR
    / "high_missingness_features.csv"
)

high_missing_features.to_csv(
    high_missing_file,
    index=False
)


# ==========================================================
# PRINT SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("REFERENCE MODEL READINESS AUDIT")
print("=" * 70)


print(
    "\nCLASS DISTRIBUTION:\n"
)

print(
    split_summary_df.to_string(
        index=False
    )
)


print(
    "\nProblem features "
    "(all-missing / constant / infinite):",
    len(problem_features)
)


if not problem_features.empty:

    print(
        "\n"
        + problem_features[
            [
                "horizon_days",
                "feature",
                "missing_pct",
                "infinite_rows",
                "all_missing",
                "constant_feature",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\nFeatures with >5% missing values:",
    len(
        high_missing_features
    )
)


if not high_missing_features.empty:

    print(
        "\nHighest missingness:\n"
    )

    print(
        high_missing_features
        .sort_values(
            "missing_pct",
            ascending=False
        )
        [
            [
                "horizon_days",
                "split",
                "feature",
                "missing_pct",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


print(
    "\nSplit summary saved to:"
)

print(
    split_summary_file
)


print(
    "\nFeature quality saved to:"
)

print(
    feature_quality_file
)


print(
    "\nProblem features saved to:"
)

print(
    problem_file
)