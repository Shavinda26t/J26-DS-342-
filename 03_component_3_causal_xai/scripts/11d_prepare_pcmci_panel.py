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
    / "pcmci_pilot"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pcmci_pilot"
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
# SETTINGS
# ==========================================================

RANDOM_STATE = 42

PILOT_HDDS = 500

MIN_CONTIGUOUS_DAYS = 60

MAX_CAUSAL_LAG = 7


# ==========================================================
# PRIMARY TEMPORAL DISCOVERY VARIABLES
#
# We deliberately use dynamic/change variables rather than
# all persistent cumulative levels.
# ==========================================================

DISCOVERY_FEATURES = [

    "reallocated_increase_1d",

    "reported_uncorrectable_increase_1d",

    "pending_sector_change_1d",

    "crc_error_increase_1d",

    "temperature",

    "temperature_change_1d",

    "start_stop_activity_1d",

    "power_cycle_activity_1d",

    "poweroff_retract_activity_1d",

    "load_cycle_activity_1d",

    "write_workload_1d",

    "read_workload_1d",
]


TARGET = "failure_within_30d"


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


parts = []

columns = (
    [
        "date",
        "serial_number",
        TARGET,
    ]
    + DISCOVERY_FEATURES
)


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


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


df = df.sort_values(
    [
        "serial_number",
        "date"
    ]
).reset_index(
    drop=True
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
# DROP ROWS WITH MISSING DISCOVERY VARIABLES
# ==========================================================

before = len(df)

df = df.dropna(
    subset=DISCOVERY_FEATURES
).copy()

print(
    "Rows removed for missing causal variables:",
    f"{before - len(df):,}"
)


# ==========================================================
# FIND CONTIGUOUS DAILY SEGMENTS
# ==========================================================

all_segments = []


for serial, drive in df.groupby(
    "serial_number",
    sort=False
):

    drive = drive.sort_values(
        "date"
    ).copy()


    # New segment whenever calendar continuity breaks.
    date_gap = (
        drive["date"]
        .diff()
        .dt.days
    )


    new_segment = (
        date_gap.isna()
        | (date_gap != 1)
    )


    drive[
        "segment_number"
    ] = (
        new_segment
        .cumsum()
    )


    for segment_number, segment in (
        drive.groupby(
            "segment_number"
        )
    ):

        segment_length = len(
            segment
        )


        if (
            segment_length
            >= MIN_CONTIGUOUS_DAYS
        ):

            all_segments.append(
                {
                    "serial_number":
                        serial,

                    "segment_number":
                        int(
                            segment_number
                        ),

                    "start_date":
                        segment[
                            "date"
                        ].min(),

                    "end_date":
                        segment[
                            "date"
                        ].max(),

                    "segment_length":
                        segment_length,

                    "has_positive_30d":
                        bool(
                            (
                                segment[
                                    TARGET
                                ] == 1
                            ).any()
                        ),
                }
            )


segment_summary = pd.DataFrame(
    all_segments
)


if segment_summary.empty:

    raise RuntimeError(
        "No HDDs have sufficiently long contiguous segments."
    )


# ==========================================================
# SELECT LONGEST ELIGIBLE SEGMENT PER HDD
# ==========================================================

segment_summary = (
    segment_summary
    .sort_values(
        [
            "serial_number",
            "segment_length"
        ],
        ascending=[
            True,
            False
        ]
    )
)


longest_segments = (
    segment_summary
    .groupby(
        "serial_number",
        as_index=False
    )
    .first()
)


print(
    "\nEligible HDDs with >= "
    f"{MIN_CONTIGUOUS_DAYS} contiguous days:",
    f"{len(longest_segments):,}"
)


# ==========================================================
# RANDOM TARGET-INDEPENDENT PILOT SAMPLE
#
# IMPORTANT:
# We do NOT select HDDs according to failure status.
# ==========================================================

pilot_n = min(
    PILOT_HDDS,
    len(longest_segments)
)


selected_drives = (
    longest_segments
    .sample(
        n=pilot_n,
        random_state=RANDOM_STATE
    )
    .sort_values(
        "serial_number"
    )
    .reset_index(
        drop=True
    )
)


print(
    "Pilot HDDs selected:",
    len(selected_drives)
)

print(
    "Positive-window HDDs appearing by chance:",
    int(
        selected_drives[
            "has_positive_30d"
        ].sum()
    )
)


# ==========================================================
# EXTRACT ONLY SELECTED SEGMENTS
# ==========================================================

pilot_parts = []


for _, selected in (
    selected_drives.iterrows()
):

    serial = (
        selected[
            "serial_number"
        ]
    )

    start_date = (
        selected[
            "start_date"
        ]
    )

    end_date = (
        selected[
            "end_date"
        ]
    )


    segment = df[
        (
            df[
                "serial_number"
            ] == serial
        )
        &
        (
            df[
                "date"
            ] >= start_date
        )
        &
        (
            df[
                "date"
            ] <= end_date
        )
    ].copy()


    segment[
        "pcmci_dataset_id"
    ] = serial


    pilot_parts.append(
        segment
    )


pilot_df = pd.concat(
    pilot_parts,
    ignore_index=True
)

del pilot_parts


# ==========================================================
# VERIFY CONTINUITY
# ==========================================================

pilot_df = (
    pilot_df
    .sort_values(
        [
            "pcmci_dataset_id",
            "date"
        ]
    )
    .reset_index(
        drop=True
    )
)


pilot_df[
    "gap_days"
] = (
    pilot_df
    .groupby(
        "pcmci_dataset_id"
    )["date"]
    .diff()
    .dt.days
)


bad_gaps = (
    pilot_df[
        "gap_days"
    ].dropna()
    != 1
).sum()


if bad_gaps > 0:

    raise RuntimeError(
        f"Found {bad_gaps} non-daily transitions "
        "inside selected PCMCI segments."
    )


# ==========================================================
# SAVE PANEL
# ==========================================================

OUTPUT_COLUMNS = (
    [
        "pcmci_dataset_id",
        "date",
        TARGET,
    ]
    + DISCOVERY_FEATURES
)


panel_file = (
    OUTPUT_DIR
    / "pcmci_pilot_panel.parquet"
)


pilot_df[
    OUTPUT_COLUMNS
].to_parquet(
    panel_file,
    index=False,
    compression="zstd"
)


# ==========================================================
# SAVE VARIABLE MANIFEST
# ==========================================================

variable_manifest = pd.DataFrame(
    {
        "variable_index":
            range(
                len(
                    DISCOVERY_FEATURES
                )
            ),

        "variable":
            DISCOVERY_FEATURES
    }
)


variable_manifest_file = (
    AUDIT_DIR
    / "pcmci_variable_manifest.csv"
)


variable_manifest.to_csv(
    variable_manifest_file,
    index=False
)


# ==========================================================
# SAVE SELECTED HDD LIST
# ==========================================================

selected_file = (
    AUDIT_DIR
    / "pcmci_selected_hdds.csv"
)


selected_drives.to_csv(
    selected_file,
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

sequence_lengths = (
    pilot_df
    .groupby(
        "pcmci_dataset_id"
    )
    .size()
)


summary = pd.DataFrame(
    [
        {
            "pilot_hdds":
                pilot_df[
                    "pcmci_dataset_id"
                ].nunique(),

            "pilot_rows":
                len(
                    pilot_df
                ),

            "causal_variables":
                len(
                    DISCOVERY_FEATURES
                ),

            "minimum_segment_days":
                MIN_CONTIGUOUS_DAYS,

            "maximum_causal_lag_days":
                MAX_CAUSAL_LAG,

            "minimum_sequence_length":
                int(
                    sequence_lengths.min()
                ),

            "median_sequence_length":
                float(
                    sequence_lengths.median()
                ),

            "maximum_sequence_length":
                int(
                    sequence_lengths.max()
                ),

            "hdds_with_positive_30d":
                int(
                    selected_drives[
                        "has_positive_30d"
                    ].sum()
                ),

            "non_daily_transitions":
                int(
                    bad_gaps
                ),
        }
    ]
)


summary_file = (
    AUDIT_DIR
    / "pcmci_pilot_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


# ==========================================================
# PRINT
# ==========================================================

print("\n")
print("=" * 70)

print(
    "PCMCI PILOT PANEL PREPARATION COMPLETE"
)

print("=" * 70)


print(
    "\nVariables:"
)

for feature in DISCOVERY_FEATURES:

    print(
        " -",
        feature
    )


print(
    "\nSummary:\n"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nPanel saved to:"
)

print(
    panel_file
)


print(
    "\nSelected HDDs saved to:"
)

print(
    selected_file
)