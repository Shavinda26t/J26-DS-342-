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
    / "interim"
    / "failure_causal_effect_readiness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

TARGET = "failure_within_30d"


TREATMENTS = {

    "reallocated_increase_event":
        "reallocated_increase_1d",

    "reported_uncorrectable_increase_event":
        "reported_uncorrectable_increase_1d",

    "pending_sector_increase_event":
        "pending_sector_change_1d",
}


# Candidate pre-treatment / contextual adjustment variables

CONFOUNDERS = [

    "drive_age_hours",

    "temperature",

    "reallocated_sectors",

    "reported_uncorrectable",

    "pending_sectors",

    "crc_errors",

    "start_stop_activity_1d",

    "power_cycle_activity_1d",

    "poweroff_retract_activity_1d",

    "load_cycle_activity_1d",

    "write_workload_1d",

    "read_workload_1d",
]


# ==========================================================
# LOAD DATA
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:

    raise FileNotFoundError(
        f"No causal files found in {INPUT_DIR}"
    )


columns = (
    [
        "date",
        "serial_number",
        TARGET,
    ]
    + list(
        set(
            list(TREATMENTS.values())
            + CONFOUNDERS
        )
    )
)


parts = []


for i, file in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] Loading {file.name}"
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


df["date"] = pd.to_datetime(
    df["date"]
)


print(
    "\nRows:",
    f"{len(df):,}"
)

print(
    "Unique HDDs:",
    f"{df['serial_number'].nunique():,}"
)


# ==========================================================
# CREATE BINARY TREATMENTS
# ==========================================================

df[
    "reallocated_increase_event"
] = (
    df[
        "reallocated_increase_1d"
    ] > 0
).astype(
    "int8"
)


df[
    "reported_uncorrectable_increase_event"
] = (
    df[
        "reported_uncorrectable_increase_1d"
    ] > 0
).astype(
    "int8"
)


# Pending sector change may legitimately be negative.
# Treatment here specifically means an INCREASE.

df[
    "pending_sector_increase_event"
] = (
    df[
        "pending_sector_change_1d"
    ] > 0
).astype(
    "int8"
)


# ==========================================================
# TREATMENT READINESS SUMMARY
# ==========================================================

summary_rows = []


for treatment in TREATMENTS.keys():

    treated = df[
        df[
            treatment
        ] == 1
    ]

    control = df[
        df[
            treatment
        ] == 0
    ]


    treated_positive = int(
        (
            treated[
                TARGET
            ] == 1
        ).sum()
    )


    control_positive = int(
        (
            control[
                TARGET
            ] == 1
        ).sum()
    )


    treated_failure_rate = (
        treated_positive
        / len(
            treated
        )
        if len(
            treated
        ) > 0
        else np.nan
    )


    control_failure_rate = (
        control_positive
        / len(
            control
        )
        if len(
            control
        ) > 0
        else np.nan
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


    crude_risk_difference = (
        treated_failure_rate
        - control_failure_rate
    )


    summary_rows.append(
        {
            "treatment":
                treatment,

            "treated_rows":
                len(
                    treated
                ),

            "control_rows":
                len(
                    control
                ),

            "treated_hdds":
                treated[
                    "serial_number"
                ].nunique(),

            "control_hdds":
                control[
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


# ==========================================================
# TREATMENT CO-OCCURRENCE
# ==========================================================

treatment_names = list(
    TREATMENTS.keys()
)


cooccurrence_rows = []


for i, treatment_a in enumerate(
    treatment_names
):

    for treatment_b in (
        treatment_names[
            i + 1:
        ]
    ):

        both = int(
            (
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
            ).sum()
        )


        cooccurrence_rows.append(
            {
                "treatment_a":
                    treatment_a,

                "treatment_b":
                    treatment_b,

                "both_event_rows":
                    both,
            }
        )


cooccurrence_df = pd.DataFrame(
    cooccurrence_rows
)


# ==========================================================
# CONFOUNDER MISSINGNESS
# ==========================================================

missing_rows = []


for feature in CONFOUNDERS:

    missing = int(
        df[
            feature
        ]
        .isna()
        .sum()
    )


    missing_rows.append(
        {
            "feature":
                feature,

            "missing_rows":
                missing,

            "missing_pct":
                missing
                / len(
                    df
                )
                * 100,
        }
    )


missing_df = pd.DataFrame(
    missing_rows
)


# ==========================================================
# MONTHLY TREATMENT COUNTS
# ==========================================================

df[
    "month"
] = (
    df[
        "date"
    ]
    .dt
    .to_period(
        "M"
    )
    .astype(str)
)


monthly_rows = []


for month, month_df in df.groupby(
    "month"
):

    for treatment in treatment_names:

        treated = (
            month_df[
                treatment
            ]
            == 1
        )


        monthly_rows.append(
            {
                "month":
                    month,

                "treatment":
                    treatment,

                "treated_rows":
                    int(
                        treated.sum()
                    ),

                "treated_hdds":
                    month_df.loc[
                        treated,
                        "serial_number"
                    ].nunique(),

                "treated_positive_rows":
                    int(
                        (
                            treated
                            &
                            (
                                month_df[
                                    TARGET
                                ] == 1
                            )
                        ).sum()
                    ),
            }
        )


monthly_df = pd.DataFrame(
    monthly_rows
)


# ==========================================================
# SAVE
# ==========================================================

summary_file = (
    OUTPUT_DIR
    / "treatment_readiness_summary.csv"
)

cooccurrence_file = (
    OUTPUT_DIR
    / "treatment_cooccurrence.csv"
)

missing_file = (
    OUTPUT_DIR
    / "confounder_missingness.csv"
)

monthly_file = (
    OUTPUT_DIR
    / "monthly_treatment_counts.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)

cooccurrence_df.to_csv(
    cooccurrence_file,
    index=False
)

missing_df.to_csv(
    missing_file,
    index=False
)

monthly_df.to_csv(
    monthly_file,
    index=False
)


# ==========================================================
# PRINT
# ==========================================================

print("\n")
print("=" * 70)
print("FAILURE CAUSAL-EFFECT READINESS AUDIT")
print("=" * 70)


print(
    "\nTreatment summary:\n"
)

print(
    summary_df.to_string(
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
    "\nConfounder missingness:\n"
)

print(
    missing_df.to_string(
        index=False
    )
)


print(
    "\nMonthly treatment counts:\n"
)

print(
    monthly_df.to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR
)