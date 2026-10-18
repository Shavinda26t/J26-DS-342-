from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_final"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_effect"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pretreatment_cohort"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# OUTPUT FILE
# ==========================================================

COHORT_FILE = (
    OUTPUT_DIR
    / "pretreatment_failure_cohort.parquet"
)


# ==========================================================
# SETTINGS
# ==========================================================

TARGET = "failure_within_30d"


# ----------------------------------------------------------
# State variables
# ----------------------------------------------------------

STATE_VARIABLES = [

    "drive_age_hours",

    "temperature",

    "reallocated_sectors",

    "reported_uncorrectable",

    "pending_sectors",

    "crc_errors",
]


# ----------------------------------------------------------
# Daily operational/activity variables
# ----------------------------------------------------------

ACTIVITY_VARIABLES = [

    "start_stop_activity_1d",

    "power_cycle_activity_1d",

    "poweroff_retract_activity_1d",

    "load_cycle_activity_1d",

    "write_workload_1d",

    "read_workload_1d",
]


# ----------------------------------------------------------
# Raw treatment/change columns
# ----------------------------------------------------------

RAW_TREATMENT_COLUMNS = [

    "reallocated_increase_1d",

    "reported_uncorrectable_increase_1d",

    "pending_sector_change_1d",
]


# ----------------------------------------------------------
# Binary treatment names
# ----------------------------------------------------------

TREATMENT_SOURCE_MAP = {

    "treat_reallocated":
        "reallocated_increase_1d",

    "treat_uncorrectable":
        "reported_uncorrectable_increase_1d",

    "treat_pending":
        "pending_sector_change_1d",
}


TREATMENTS = list(
    TREATMENT_SOURCE_MAP.keys()
)


# ==========================================================
# LOAD INPUT FILES
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:

    raise FileNotFoundError(
        f"No files found in {INPUT_DIR}"
    )


columns = (

    [
        "date",
        "serial_number",
        TARGET,
    ]

    + STATE_VARIABLES

    + ACTIVITY_VARIABLES

    + RAW_TREATMENT_COLUMNS
)


# Remove accidental duplicate requested columns
columns = list(
    dict.fromkeys(
        columns
    )
)


parts = []


for i, file in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] "
        f"Loading {file.name}"
    )

    temp = pd.read_parquet(
        file,
        columns=columns
    )

    parts.append(
        temp
    )


df = pd.concat(
    parts,
    ignore_index=True
)

del parts


# ==========================================================
# PREPARE DATE AND SORT
# ==========================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "date",
        "serial_number"
    ]
)


df = (
    df
    .sort_values(
        [
            "serial_number",
            "date"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nRows loaded:",
    f"{len(df):,}"
)

print(
    "Unique HDDs:",
    f"{df['serial_number'].nunique():,}"
)


# ==========================================================
# ENSURE NUMERIC VARIABLES
# ==========================================================

numeric_columns = (

    [TARGET]

    + STATE_VARIABLES

    + ACTIVITY_VARIABLES

    + RAW_TREATMENT_COLUMNS
)


for feature in numeric_columns:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# ==========================================================
# CURRENT-DAY TREATMENT EVENTS
#
# IMPORTANT:
#
# Missing treatment values remain NaN.
#
# We must NOT convert a missing/reset value into treatment=0,
# because that would incorrectly place invalid observations
# into the untreated/control group.
# ==========================================================

for treatment, source in (
    TREATMENT_SOURCE_MAP.items()
):

    source_values = df[
        source
    ]


    df[
        treatment
    ] = np.where(

        source_values.isna(),

        np.nan,

        (
            source_values > 0
        ).astype(
            "int8"
        )
    )


# ==========================================================
# TEMPORAL ORDERING
# ==========================================================

df[
    "previous_date"
] = (
    df
    .groupby(
        "serial_number",
        sort=False
    )["date"]
    .shift(1)
)


df[
    "previous_day_gap"
] = (
    df[
        "date"
    ]
    -
    df[
        "previous_date"
    ]
).dt.days


df[
    "has_exact_previous_day"
] = (
    df[
        "previous_day_gap"
    ]
    == 1
)


# ==========================================================
# BASELINE SOURCE VARIABLES
#
# Treatment occurs on day t.
#
# These variables are shifted to day t-1.
# ==========================================================

BASELINE_SOURCE_VARIABLES = (

    STATE_VARIABLES

    + ACTIVITY_VARIABLES
)


BASELINE_FEATURES = []


for feature in BASELINE_SOURCE_VARIABLES:

    baseline_name = (
        f"baseline_{feature}_lag1"
    )


    df[
        baseline_name
    ] = (
        df
        .groupby(
            "serial_number",
            sort=False
        )[feature]
        .shift(1)
    )


    # Do not allow lagging across a calendar gap.
    df.loc[
        ~df[
            "has_exact_previous_day"
        ],
        baseline_name
    ] = np.nan


    BASELINE_FEATURES.append(
        baseline_name
    )


# ==========================================================
# PRIOR-DAY TREATMENT / DEGRADATION HISTORY
#
# These represent whether the HDD experienced each
# degradation event on day t-1.
# ==========================================================

PRIOR_EVENT_FEATURES = []


for treatment in TREATMENTS:

    baseline_event_name = (
        f"baseline_{treatment}_lag1"
    )


    df[
        baseline_event_name
    ] = (
        df
        .groupby(
            "serial_number",
            sort=False
        )[treatment]
        .shift(1)
    )


    df.loc[
        ~df[
            "has_exact_previous_day"
        ],
        baseline_event_name
    ] = np.nan


    PRIOR_EVENT_FEATURES.append(
        baseline_event_name
    )


# ==========================================================
# CALENDAR CONTEXT
# ==========================================================

df[
    "calendar_month"
] = (
    df[
        "date"
    ]
    .dt
    .month
    .astype(
        "int8"
    )
)


minimum_date = df[
    "date"
].min()


df[
    "calendar_day_index"
] = (
    df[
        "date"
    ]
    -
    minimum_date
).dt.days.astype(
    "int16"
)


# ==========================================================
# ADJUSTMENT FEATURE SET
#
# IMPORTANT:
#
# All HDD-health / activity adjustment variables are measured
# before treatment (t-1).
#
# calendar_day_index is calendar-time context.
# ==========================================================

ADJUSTMENT_FEATURES = (

    BASELINE_FEATURES

    + PRIOR_EVENT_FEATURES

    + [
        "calendar_day_index"
    ]
)


# ==========================================================
# REQUIRE EXACT PREVIOUS-DAY HISTORY
# ==========================================================

before_previous_day_filter = len(
    df
)


df = df[
    df[
        "has_exact_previous_day"
    ]
].copy()


removed_no_previous_day = (
    before_previous_day_filter
    - len(
        df
    )
)


# ==========================================================
# REMOVE INVALID / MISSING CURRENT TREATMENTS
#
# This mainly protects against:
# - counter reset
# - counter wrap
# - invalid delta
# - unexpected missing delta
# ==========================================================

current_treatment_missing = (
    df[
        TREATMENTS
    ]
    .isna()
    .any(
        axis=1
    )
)


removed_missing_treatment = int(
    current_treatment_missing.sum()
)


df = df[
    ~current_treatment_missing
].copy()


# ==========================================================
# CAST CURRENT TREATMENTS TO INTEGER
# ==========================================================

for treatment in TREATMENTS:

    df[
        treatment
    ] = (
        df[
            treatment
        ]
        .astype(
            "int8"
        )
    )


# ==========================================================
# REQUIRE COMPLETE PRE-TREATMENT BASELINE
# ==========================================================

before_complete_case = len(
    df
)


df = df.dropna(
    subset=(
        ADJUSTMENT_FEATURES

        + [
            TARGET
        ]
    )
).copy()


removed_incomplete_baseline = (
    before_complete_case
    - len(
        df
    )
)


# ==========================================================
# CAST PRIOR EVENT FEATURES
# ==========================================================

for feature in PRIOR_EVENT_FEATURES:

    df[
        feature
    ] = (
        df[
            feature
        ]
        .astype(
            "int8"
        )
    )


# ==========================================================
# TARGET TYPE
# ==========================================================

df[
    TARGET
] = (
    df[
        TARGET
    ]
    .astype(
        "int8"
    )
)


# ==========================================================
# SAME-DAY CO-OCCURRENCE FLAGS
#
# IMPORTANT:
#
# These are NOT adjustment/confounder variables.
#
# They are saved only for later sensitivity analysis.
# ==========================================================

df[
    "same_day_event_count"
] = (
    df[
        TREATMENTS
    ]
    .sum(
        axis=1
    )
    .astype(
        "int8"
    )
)


# ==========================================================
# ISOLATED REALLOCATION EVENT
# ==========================================================

df[
    "isolated_reallocated_event"
] = (
    (
        df[
            "treat_reallocated"
        ] == 1
    )
    &
    (
        df[
            "treat_uncorrectable"
        ] == 0
    )
    &
    (
        df[
            "treat_pending"
        ] == 0
    )
).astype(
    "int8"
)


# ==========================================================
# ISOLATED UNCORRECTABLE EVENT
# ==========================================================

df[
    "isolated_uncorrectable_event"
] = (
    (
        df[
            "treat_uncorrectable"
        ] == 1
    )
    &
    (
        df[
            "treat_reallocated"
        ] == 0
    )
    &
    (
        df[
            "treat_pending"
        ] == 0
    )
).astype(
    "int8"
)


# ==========================================================
# ISOLATED PENDING-SECTOR EVENT
# ==========================================================

df[
    "isolated_pending_event"
] = (
    (
        df[
            "treat_pending"
        ] == 1
    )
    &
    (
        df[
            "treat_reallocated"
        ] == 0
    )
    &
    (
        df[
            "treat_uncorrectable"
        ] == 0
    )
).astype(
    "int8"
)


# ==========================================================
# OUTPUT COLUMNS
# ==========================================================

OUTPUT_COLUMNS = (

    [
        "date",
        "serial_number",
        TARGET,

        "treat_reallocated",
        "treat_uncorrectable",
        "treat_pending",

        "isolated_reallocated_event",
        "isolated_uncorrectable_event",
        "isolated_pending_event",

        "same_day_event_count",

        "calendar_month",
    ]

    + ADJUSTMENT_FEATURES
)


# ==========================================================
# DUPLICATE COLUMN SAFETY CHECK
# ==========================================================

if len(
    OUTPUT_COLUMNS
) != len(
    set(
        OUTPUT_COLUMNS
    )
):

    duplicates = [

        column

        for column in OUTPUT_COLUMNS

        if OUTPUT_COLUMNS.count(
            column
        ) > 1
    ]


    raise RuntimeError(
        "Duplicate output columns found: "
        f"{sorted(set(duplicates))}"
    )


# ==========================================================
# FINAL DATA VALIDATION
# ==========================================================

if df.empty:

    raise RuntimeError(
        "Final pre-treatment cohort is empty."
    )


if df[
    TARGET
].isna().any():

    raise RuntimeError(
        "Target contains missing values."
    )


for treatment in TREATMENTS:

    unexpected_values = set(
        df[
            treatment
        ].unique()
    ) - {
        0,
        1
    }


    if unexpected_values:

        raise RuntimeError(
            f"Unexpected values in {treatment}: "
            f"{unexpected_values}"
        )


# ==========================================================
# SAVE FINAL COHORT
# ==========================================================

print(
    "\nSaving pre-treatment cohort..."
)


df[
    OUTPUT_COLUMNS
].to_parquet(
    COHORT_FILE,
    index=False,
    compression="zstd"
)


# ==========================================================
# VERIFY SAVED FILE
# ==========================================================

if not COHORT_FILE.exists():

    raise RuntimeError(
        "Cohort file was not created."
    )


# ==========================================================
# TREATMENT SUMMARY
# ==========================================================

summary_rows = []


for treatment in TREATMENTS:

    treated = (
        df[
            treatment
        ] == 1
    )


    control = (
        df[
            treatment
        ] == 0
    )


    treated_rows = int(
        treated.sum()
    )


    control_rows = int(
        control.sum()
    )


    treated_positive = int(
        (
            treated
            &
            (
                df[
                    TARGET
                ] == 1
            )
        ).sum()
    )


    control_positive = int(
        (
            control
            &
            (
                df[
                    TARGET
                ] == 1
            )
        ).sum()
    )


    treated_failure_rate = (
        treated_positive
        / treated_rows
        if treated_rows > 0
        else np.nan
    )


    control_failure_rate = (
        control_positive
        / control_rows
        if control_rows > 0
        else np.nan
    )


    crude_risk_difference = (
        treated_failure_rate
        - control_failure_rate
    )


    crude_risk_ratio = (

        treated_failure_rate
        / control_failure_rate

        if (
            control_failure_rate > 0
            and not np.isnan(
                treated_failure_rate
            )
        )

        else np.nan
    )


    summary_rows.append(
        {
            "treatment":
                treatment,

            "treated_rows":
                treated_rows,

            "control_rows":
                control_rows,

            "treated_hdds":
                df.loc[
                    treated,
                    "serial_number"
                ].nunique(),

            "control_hdds":
                df.loc[
                    control,
                    "serial_number"
                ].nunique(),

            "treated_positive_rows":
                treated_positive,

            "control_positive_rows":
                control_positive,

            "treated_failure_rate":
                treated_failure_rate,

            "control_failure_rate":
                control_failure_rate,

            "crude_risk_difference":
                crude_risk_difference,

            "crude_risk_ratio":
                crude_risk_ratio,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_file = (
    AUDIT_DIR
    / "pretreatment_treatment_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ==========================================================
# ISOLATED EVENT SUMMARY
# ==========================================================

ISOLATED_EVENTS = [

    "isolated_reallocated_event",

    "isolated_uncorrectable_event",

    "isolated_pending_event",
]


isolated_rows = []


for isolated_feature in ISOLATED_EVENTS:

    mask = (
        df[
            isolated_feature
        ] == 1
    )


    positive_rows = int(
        (
            mask
            &
            (
                df[
                    TARGET
                ] == 1
            )
        ).sum()
    )


    isolated_rows.append(
        {
            "event":
                isolated_feature,

            "rows":
                int(
                    mask.sum()
                ),

            "unique_hdds":
                df.loc[
                    mask,
                    "serial_number"
                ].nunique(),

            "positive_rows":
                positive_rows,

            "failure_rate":
                (
                    positive_rows
                    / mask.sum()
                    if mask.sum() > 0
                    else np.nan
                ),
        }
    )


isolated_df = pd.DataFrame(
    isolated_rows
)


isolated_file = (
    AUDIT_DIR
    / "isolated_event_summary.csv"
)


isolated_df.to_csv(
    isolated_file,
    index=False
)


# ==========================================================
# TREATMENT CO-OCCURRENCE SUMMARY
# ==========================================================

cooccurrence_rows = []


for i, treatment_a in enumerate(
    TREATMENTS
):

    for treatment_b in (
        TREATMENTS[
            i + 1:
        ]
    ):

        both = (
            (
                df[
                    treatment_a
                ] == 1
            )
            &
            (
                df[
                    treatment_b
                ] == 1
            )
        )


        cooccurrence_rows.append(
            {
                "treatment_a":
                    treatment_a,

                "treatment_b":
                    treatment_b,

                "both_event_rows":
                    int(
                        both.sum()
                    ),

                "both_event_hdds":
                    df.loc[
                        both,
                        "serial_number"
                    ].nunique(),

                "positive_rows":
                    int(
                        (
                            both
                            &
                            (
                                df[
                                    TARGET
                                ] == 1
                            )
                        ).sum()
                    ),
            }
        )


cooccurrence_df = pd.DataFrame(
    cooccurrence_rows
)


cooccurrence_file = (
    AUDIT_DIR
    / "pretreatment_event_cooccurrence.csv"
)


cooccurrence_df.to_csv(
    cooccurrence_file,
    index=False
)


# ==========================================================
# ADJUSTMENT MANIFEST
# ==========================================================

manifest_rows = []


for feature in BASELINE_FEATURES:

    manifest_rows.append(
        {
            "adjustment_feature":
                feature,

            "measurement_time":
                "t_minus_1",

            "variable_type":
                "baseline_state_or_activity",
        }
    )


for feature in PRIOR_EVENT_FEATURES:

    manifest_rows.append(
        {
            "adjustment_feature":
                feature,

            "measurement_time":
                "t_minus_1",

            "variable_type":
                "prior_degradation_event",
        }
    )


manifest_rows.append(
    {
        "adjustment_feature":
            "calendar_day_index",

        "measurement_time":
            "calendar_time",

        "variable_type":
            "calendar_context",
    }
)


manifest_df = pd.DataFrame(
    manifest_rows
)


manifest_file = (
    AUDIT_DIR
    / "pretreatment_adjustment_manifest.csv"
)


manifest_df.to_csv(
    manifest_file,
    index=False
)


# ==========================================================
# COHORT BUILD SUMMARY
# ==========================================================

cohort_summary_df = pd.DataFrame(
    [
        {
            "input_rows":
                before_previous_day_filter,

            "removed_without_exact_previous_day":
                removed_no_previous_day,

            "removed_missing_current_treatment":
                removed_missing_treatment,

            "removed_incomplete_baseline":
                removed_incomplete_baseline,

            "final_rows":
                len(
                    df
                ),

            "unique_hdds":
                df[
                    "serial_number"
                ].nunique(),

            "positive_rows":
                int(
                    (
                        df[
                            TARGET
                        ] == 1
                    ).sum()
                ),

            "negative_rows":
                int(
                    (
                        df[
                            TARGET
                        ] == 0
                    ).sum()
                ),

            "adjustment_variables":
                len(
                    ADJUSTMENT_FEATURES
                ),
        }
    ]
)


cohort_summary_file = (
    AUDIT_DIR
    / "pretreatment_cohort_summary.csv"
)


cohort_summary_df.to_csv(
    cohort_summary_file,
    index=False
)


# ==========================================================
# MONTHLY SUMMARY
# ==========================================================

monthly_rows = []


for month, month_df in df.groupby(
    "calendar_month"
):

    monthly_rows.append(
        {
            "calendar_month":
                int(
                    month
                ),

            "rows":
                len(
                    month_df
                ),

            "unique_hdds":
                month_df[
                    "serial_number"
                ].nunique(),

            "positive_rows":
                int(
                    (
                        month_df[
                            TARGET
                        ] == 1
                    ).sum()
                ),

            "reallocated_treated_rows":
                int(
                    month_df[
                        "treat_reallocated"
                    ].sum()
                ),

            "uncorrectable_treated_rows":
                int(
                    month_df[
                        "treat_uncorrectable"
                    ].sum()
                ),

            "pending_treated_rows":
                int(
                    month_df[
                        "treat_pending"
                    ].sum()
                ),
        }
    )


monthly_df = pd.DataFrame(
    monthly_rows
)


monthly_file = (
    AUDIT_DIR
    / "pretreatment_monthly_summary.csv"
)


monthly_df.to_csv(
    monthly_file,
    index=False
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 70)

print(
    "PRE-TREATMENT CAUSAL COHORT COMPLETE"
)

print("=" * 70)


print(
    "Rows removed without exact previous day:",
    f"{removed_no_previous_day:,}"
)


print(
    "Rows removed for missing/invalid current treatment:",
    f"{removed_missing_treatment:,}"
)


print(
    "Rows removed for incomplete baseline:",
    f"{removed_incomplete_baseline:,}"
)


print(
    "Final rows:",
    f"{len(df):,}"
)


print(
    "Unique HDDs:",
    f"{df['serial_number'].nunique():,}"
)


print(
    "30-day positive rows:",
    f"{(df[TARGET] == 1).sum():,}"
)


print(
    "\nTreatment summary:\n"
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nIsolated-event summary:\n"
)


print(
    isolated_df.to_string(
        index=False
    )
)


print(
    "\nTreatment co-occurrence:\n"
)


print(
    cooccurrence_df.to_string(
        index=False
    )
)


print(
    "\nMonthly summary:\n"
)


print(
    monthly_df.to_string(
        index=False
    )
)


print(
    "\nAdjustment variables:"
)


for feature in ADJUSTMENT_FEATURES:

    print(
        " -",
        feature
    )


print(
    "\nCohort saved to:"
)


print(
    COHORT_FILE
)


print(
    "\nTreatment summary saved to:"
)


print(
    summary_file
)


print(
    "\nIsolated-event summary saved to:"
)


print(
    isolated_file
)


print(
    "\nAdjustment manifest saved to:"
)


print(
    manifest_file
)


print(
    "\nAudit directory:"
)


print(
    AUDIT_DIR
)