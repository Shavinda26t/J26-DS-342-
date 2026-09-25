from pathlib import Path
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "split_readiness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

HORIZONS = [7, 14, 30]

ANALYSIS_MONTHS = [
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
]


# ==========================================================
# LOAD MONTHLY FILES
# ==========================================================

month_data = {}

for month in ANALYSIS_MONTHS:

    file = (
        INPUT_DIR
        / f"{month}.parquet"
    )

    if not file.exists():

        raise FileNotFoundError(
            f"Missing temporal file: {file}"
        )

    print(
        "Loading:",
        file.name
    )

    columns = [
        "date",
        "serial_number",
        "eligible_for_prediction",

        "failure_within_7d",
        "failure_within_14d",
        "failure_within_30d",

        "has_sufficient_7d_history",
        "has_sufficient_14d_history",
        "has_sufficient_30d_history",
    ]

    df = pd.read_parquet(
        file,
        columns=columns
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    month_data[month] = df


# ==========================================================
# MONTH / HORIZON SUMMARY
# ==========================================================

summary_rows = []

positive_drive_sets = {}


for month, df in month_data.items():

    for horizon in HORIZONS:

        target = (
            f"failure_within_{horizon}d"
        )

        history_flag = (
            f"has_sufficient_{horizon}d_history"
        )

        usable = df[
            (df["eligible_for_prediction"] == True)
            & (df[history_flag] == True)
            & (df[target].notna())
        ].copy()


        positive = usable[
            usable[target] == 1
        ]

        negative = usable[
            usable[target] == 0
        ]


        positive_drives = set(
            positive[
                "serial_number"
            ]
            .dropna()
            .astype(str)
        )


        positive_drive_sets[
            (month, horizon)
        ] = positive_drives


        summary_rows.append(
            {
                "month":
                    month,

                "horizon_days":
                    horizon,

                "usable_rows":
                    len(usable),

                "positive_rows":
                    len(positive),

                "negative_rows":
                    len(negative),

                "positive_rate_pct":
                    round(
                        (
                            len(positive)
                            / len(usable)
                            * 100
                        )
                        if len(usable) > 0
                        else 0,
                        6
                    ),

                "unique_hdds":
                    usable[
                        "serial_number"
                    ].nunique(),

                "unique_positive_hdds":
                    positive[
                        "serial_number"
                    ].nunique(),

                "unique_negative_hdds":
                    negative[
                        "serial_number"
                    ].nunique(),
            }
        )


summary_df = pd.DataFrame(
    summary_rows
)


# ==========================================================
# CHECK POSITIVE HDD OVERLAP BETWEEN MONTHS
# ==========================================================

month_pairs = [
    (
        "2025-08",
        "2025-09"
    ),

    (
        "2025-09",
        "2025-10"
    ),

    (
        "2025-10",
        "2025-11"
    ),
]


overlap_rows = []


for horizon in HORIZONS:

    for month_a, month_b in month_pairs:

        set_a = (
            positive_drive_sets[
                (month_a, horizon)
            ]
        )

        set_b = (
            positive_drive_sets[
                (month_b, horizon)
            ]
        )

        overlap = (
            set_a.intersection(
                set_b
            )
        )


        overlap_rows.append(
            {
                "horizon_days":
                    horizon,

                "month_a":
                    month_a,

                "month_b":
                    month_b,

                "positive_hdds_month_a":
                    len(set_a),

                "positive_hdds_month_b":
                    len(set_b),

                "shared_positive_hdds":
                    len(overlap),
            }
        )


overlap_df = pd.DataFrame(
    overlap_rows
)


# ==========================================================
# TRAIN-CANDIDATE VS VALIDATION
# ==========================================================

train_overlap_rows = []


for horizon in HORIZONS:

    august = positive_drive_sets[
        ("2025-08", horizon)
    ]

    september = positive_drive_sets[
        ("2025-09", horizon)
    ]

    october = positive_drive_sets[
        ("2025-10", horizon)
    ]

    november = positive_drive_sets[
        ("2025-11", horizon)
    ]


    train_candidates = (
        august
        | september
    )


    train_val_overlap = (
        train_candidates
        & october
    )


    val_test_overlap = (
        october
        & november
    )


    train_test_overlap = (
        train_candidates
        & november
    )


    train_overlap_rows.append(
        {
            "horizon_days":
                horizon,

            "train_positive_hdds":
                len(
                    train_candidates
                ),

            "validation_positive_hdds":
                len(
                    october
                ),

            "test_positive_hdds":
                len(
                    november
                ),

            "train_validation_shared":
                len(
                    train_val_overlap
                ),

            "validation_test_shared":
                len(
                    val_test_overlap
                ),

            "train_test_shared":
                len(
                    train_test_overlap
                ),
        }
    )


split_overlap_df = pd.DataFrame(
    train_overlap_rows
)


# ==========================================================
# SAVE
# ==========================================================

summary_file = (
    OUTPUT_DIR
    / "month_horizon_readiness.csv"
)

overlap_file = (
    OUTPUT_DIR
    / "adjacent_month_positive_overlap.csv"
)

split_overlap_file = (
    OUTPUT_DIR
    / "candidate_split_overlap.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)

overlap_df.to_csv(
    overlap_file,
    index=False
)

split_overlap_df.to_csv(
    split_overlap_file,
    index=False
)


# ==========================================================
# PRINT
# ==========================================================

print("\n")
print("=" * 70)
print("SPLIT READINESS AUDIT")
print("=" * 70)

print(
    "\nMONTH / HORIZON SUMMARY:\n"
)

print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nADJACENT MONTH POSITIVE-HDD OVERLAP:\n"
)

print(
    overlap_df.to_string(
        index=False
    )
)


print(
    "\nCANDIDATE TRAIN / VALIDATION / TEST OVERLAP:\n"
)

print(
    split_overlap_df.to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR
)