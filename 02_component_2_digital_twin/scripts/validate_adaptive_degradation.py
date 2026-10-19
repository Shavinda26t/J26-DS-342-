import os
import numpy as np
import pandas as pd


# ============================================================
# Adaptive Digital Twin
# Failure-Relative Degradation Validation
#
# Version:
# Nearest available observation within +/- 1 day
#
# IMPORTANT:
# This is a validation experiment.
# Candidate thresholds are NOT yet final/validated.
# ============================================================


INPUT_FILE = r".\results\digital_twin_state_v5.csv"

OUTPUT_DIR = r".\results\adaptive_digital_twin"

TRAJECTORY_FILE = os.path.join(
    OUTPUT_DIR,
    "adaptive_degradation_nearest_trajectory.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "adaptive_degradation_nearest_summary.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

CHUNK_SIZE = 500_000

TOLERANCE_DAYS = 1


# ============================================================
# Target failure-relative days
# ============================================================

TARGET_DAYS = [
    -30,
    -14,
    -7,
    -3,
    -1,
    0
]


# ============================================================
# Candidate severity thresholds
#
# These are experimental thresholds derived from the
# observed negative-change distributions.
#
# They will only be retained if the failure trajectory
# validation supports them.
# ============================================================

SMART1_MODERATE = -3
SMART1_STRONG = -8
SMART1_SEVERE = -15

SMART187_MODERATE = -3
SMART187_STRONG = -9
SMART187_SEVERE = -23


# ============================================================
# Input columns
# ============================================================

FAILURE_COLUMNS = [
    "date",
    "serial_number",
    "failure"
]


STATE_COLUMNS = [
    "date",
    "serial_number",
    "failure",

    "smart_1_normalized_change_7d",
    "smart_187_normalized_change_7d",

    "smart_1_available",
    "smart_187_available"
]


# ============================================================
# Severity functions
# ============================================================

def smart1_severity(value):

    if pd.isna(value):
        return np.nan

    if value >= 0:
        return 0.0

    if value >= SMART1_MODERATE:
        return 1.0

    if value >= SMART1_STRONG:
        return 2.0

    if value >= SMART1_SEVERE:
        return 3.0

    return 4.0


def smart187_severity(value):

    if pd.isna(value):
        return np.nan

    if value >= 0:
        return 0.0

    if value >= SMART187_MODERATE:
        return 1.0

    if value >= SMART187_STRONG:
        return 2.0

    if value >= SMART187_SEVERE:
        return 3.0

    return 4.0


# ============================================================
# PASS 1
# Find failure date for every failed HDD
# ============================================================

print("=" * 70)
print("PASS 1: IDENTIFYING FAILED HDDs")
print("=" * 70)

failure_dates = {}

total_rows = 0


for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=FAILURE_COLUMNS,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    print(
        f"Reading failure information chunk {chunk_number}..."
    )

    total_rows += len(chunk)

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )

    failed = chunk[
        chunk["failure"] == 1
    ]


    for row in failed.itertuples(
        index=False
    ):

        serial = row.serial_number

        failure_date = row.date


        if serial not in failure_dates:

            failure_dates[serial] = failure_date

        elif failure_date < failure_dates[serial]:

            failure_dates[serial] = failure_date


print()

print(
    "Total rows scanned:",
    f"{total_rows:,}"
)

print(
    "Failed HDDs:",
    f"{len(failure_dates):,}"
)


# ============================================================
# PASS 2
# Collect observations around each target
# ============================================================

print("\n" + "=" * 70)
print("PASS 2: FINDING NEAREST OBSERVATIONS")
print("=" * 70)

failed_serials = set(
    failure_dates.keys()
)


candidate_records = []


for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=STATE_COLUMNS,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    print(
        f"Processing data chunk {chunk_number}..."
    )

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )


    # --------------------------------------------------------
    # Keep only failed HDDs
    # --------------------------------------------------------

    chunk = chunk[
        chunk["serial_number"].isin(
            failed_serials
        )
    ].copy()


    if len(chunk) == 0:
        continue


    # --------------------------------------------------------
    # Map failure date
    # --------------------------------------------------------

    chunk["failure_date"] = (
        chunk["serial_number"]
        .map(failure_dates)
    )


    # --------------------------------------------------------
    # Failure-relative day
    # --------------------------------------------------------

    chunk["relative_day"] = (
        chunk["date"]
        -
        chunk["failure_date"]
    ).dt.days


    # --------------------------------------------------------
    # We only need observations from:
    #
    # -31 through 0
    #
    # because the earliest target is -30 with +/-1 day.
    #
    # We intentionally do NOT use +1 after failure.
    # --------------------------------------------------------

    chunk = chunk[
        (
            chunk["relative_day"] >= -31
        )
        &
        (
            chunk["relative_day"] <= 0
        )
    ].copy()


    if len(chunk) == 0:
        continue


    # --------------------------------------------------------
    # Convert temporal values
    # --------------------------------------------------------

    chunk[
        "smart_1_normalized_change_7d"
    ] = pd.to_numeric(
        chunk[
            "smart_1_normalized_change_7d"
        ],
        errors="coerce"
    )


    chunk[
        "smart_187_normalized_change_7d"
    ] = pd.to_numeric(
        chunk[
            "smart_187_normalized_change_7d"
        ],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Respect SMART availability
    # --------------------------------------------------------

    chunk.loc[
        chunk["smart_1_available"] == 0,
        "smart_1_normalized_change_7d"
    ] = np.nan


    chunk.loc[
        chunk["smart_187_available"] == 0,
        "smart_187_normalized_change_7d"
    ] = np.nan


    # --------------------------------------------------------
    # Generate candidate target matches
    # --------------------------------------------------------

    for target_day in TARGET_DAYS:

        # Failure day:
        # use the exact failure-day observation.
        if target_day == 0:

            target_candidates = chunk[
                chunk["relative_day"] == 0
            ].copy()

        else:

            lower = (
                target_day
                -
                TOLERANCE_DAYS
            )

            upper = (
                target_day
                +
                TOLERANCE_DAYS
            )


            target_candidates = chunk[
                (
                    chunk["relative_day"] >= lower
                )
                &
                (
                    chunk["relative_day"] <= upper
                )
                &
                (
                    chunk["relative_day"] <= -1
                )
            ].copy()


        if len(target_candidates) == 0:

            continue


        # ----------------------------------------------------
        # Distance from target
        # ----------------------------------------------------

        target_candidates[
            "target_day"
        ] = target_day


        target_candidates[
            "distance_from_target"
        ] = (
            target_candidates[
                "relative_day"
            ]
            -
            target_day
        ).abs()


        # ----------------------------------------------------
        # We will select the nearest observation for each
        # HDD and target.
        #
        # If there is a tie, prefer the later observation
        # because it is closer to the failure event.
        # ----------------------------------------------------

        target_candidates = (
            target_candidates
            .sort_values(
                [
                    "serial_number",
                    "distance_from_target",
                    "relative_day"
                ],
                ascending=[
                    True,
                    True,
                    False
                ]
            )
        )


        target_candidates = (
            target_candidates
            .drop_duplicates(
                subset=[
                    "serial_number",
                    "target_day"
                ],
                keep="first"
            )
        )


        candidate_records.append(
            target_candidates[
                [
                    "serial_number",
                    "failure_date",
                    "target_day",
                    "relative_day",
                    "distance_from_target",

                    "smart_1_normalized_change_7d",
                    "smart_187_normalized_change_7d",

                    "smart_1_available",
                    "smart_187_available"
                ]
            ]
        )


# ============================================================
# Combine candidate observations
# ============================================================

print("\nCombining candidate observations...")


if len(candidate_records) == 0:

    raise RuntimeError(
        "No failure-relative observations were found."
    )


candidates = pd.concat(
    candidate_records,
    ignore_index=True
)


# ============================================================
# Final safeguard:
# exactly one observation per HDD per target
# ============================================================

candidates = (
    candidates
    .sort_values(
        [
            "serial_number",
            "target_day",
            "distance_from_target",
            "relative_day"
        ],
        ascending=[
            True,
            True,
            True,
            False
        ]
    )
    .drop_duplicates(
        subset=[
            "serial_number",
            "target_day"
        ],
        keep="first"
    )
    .reset_index(drop=True)
)


# ============================================================
# Calculate severity
# ============================================================

candidates[
    "smart_1_severity"
] = candidates[
    "smart_1_normalized_change_7d"
].apply(
    smart1_severity
)


candidates[
    "smart_187_severity"
] = candidates[
    "smart_187_normalized_change_7d"
].apply(
    smart187_severity
)


# ============================================================
# Combine severity
#
# MAX is used.
#
# If SMART187 is missing, it does not reduce the SMART1
# evidence.
#
# If both are missing, combined severity is NaN.
# ============================================================

severity_matrix = np.column_stack(
    [
        candidates[
            "smart_1_severity"
        ].astype(float).values,

        candidates[
            "smart_187_severity"
        ].astype(float).values
    ]
)


combined_severity = np.full(
    len(candidates),
    np.nan,
    dtype=float
)


has_evidence = np.any(
    np.isfinite(
        severity_matrix
    ),
    axis=1
)


if np.any(has_evidence):

    combined_severity[
        has_evidence
    ] = np.nanmax(
        severity_matrix[
            has_evidence
        ],
        axis=1
    )


candidates[
    "combined_degradation_severity"
] = combined_severity


# ============================================================
# Save individual matched observations
# ============================================================

candidates.to_csv(
    TRAJECTORY_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FAILURE TRAJECTORY VALIDATION")
print("=" * 70)


summary_rows = []


for target_day in TARGET_DAYS:

    subset = candidates[
        candidates["target_day"]
        == target_day
    ].copy()


    smart1 = subset[
        "smart_1_severity"
    ].dropna()


    smart187 = subset[
        "smart_187_severity"
    ].dropna()


    combined = subset[
        "combined_degradation_severity"
    ].dropna()


    def pct_at_least(
        values,
        threshold
    ):

        if len(values) == 0:
            return np.nan

        return (
            (values >= threshold).mean()
            * 100
        )


    row = {

        "target_day":
            target_day,

        "matched_hdds":
            subset[
                "serial_number"
            ].nunique(),

        "smart1_n":
            len(smart1),

        "smart1_mean_severity":
            (
                smart1.mean()
                if len(smart1) > 0
                else np.nan
            ),

        "smart1_moderate_or_higher_pct":
            pct_at_least(
                smart1,
                2
            ),

        "smart1_strong_or_higher_pct":
            pct_at_least(
                smart1,
                3
            ),

        "smart1_severe_pct":
            pct_at_least(
                smart1,
                4
            ),

        "smart187_n":
            len(smart187),

        "smart187_mean_severity":
            (
                smart187.mean()
                if len(smart187) > 0
                else np.nan
            ),

        "smart187_moderate_or_higher_pct":
            pct_at_least(
                smart187,
                2
            ),

        "smart187_strong_or_higher_pct":
            pct_at_least(
                smart187,
                3
            ),

        "smart187_severe_pct":
            pct_at_least(
                smart187,
                4
            ),

        "combined_n":
            len(combined),

        "combined_mean_severity":
            (
                combined.mean()
                if len(combined) > 0
                else np.nan
            ),

        "combined_moderate_or_higher_pct":
            pct_at_least(
                combined,
                2
            ),

        "combined_strong_or_higher_pct":
            pct_at_least(
                combined,
                3
            ),

        "combined_severe_pct":
            pct_at_least(
                combined,
                4
            ),

        "mean_observation_offset_days":
            (
                subset[
                    "distance_from_target"
                ].mean()
                if len(subset) > 0
                else np.nan
            )
    }


    summary_rows.append(
        row
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Print summary
# ============================================================

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# Save summary
# ============================================================

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# Coverage analysis
# ============================================================

print("\n" + "=" * 70)
print("COVERAGE CHECK")
print("=" * 70)


for row in summary.itertuples(
    index=False
):

    coverage = (
        row.matched_hdds
        /
        len(failure_dates)
        *
        100
    )


    print(
        f"Target {row.target_day:>3} days: "
        f"{row.matched_hdds:>4} / "
        f"{len(failure_dates)} HDDs "
        f"({coverage:.2f}%)"
    )


# ============================================================
# Trend check
# ============================================================

print("\n" + "=" * 70)
print("TREND CHECK")
print("=" * 70)


print(
    "\nCombined mean severity:"
)


for row in summary.itertuples(
    index=False
):

    print(
        f"Day {row.target_day:>3}: "
        f"{row.combined_mean_severity:.4f}"
    )


print(
    "\nCombined moderate-or-higher:"
)


for row in summary.itertuples(
    index=False
):

    print(
        f"Day {row.target_day:>3}: "
        f"{row.combined_moderate_or_higher_pct:.2f}%"
    )


print(
    "\nCombined strong-or-higher:"
)


for row in summary.itertuples(
    index=False
):

    print(
        f"Day {row.target_day:>3}: "
        f"{row.combined_strong_or_higher_pct:.2f}%"
    )


# ============================================================
# Monotonic trend check
#
# We order from farthest from failure to closest:
#
# -30, -14, -7, -3, -1, 0
#
# ============================================================

ordered = summary.sort_values(
    "target_day"
)


mean_values = ordered[
    "combined_mean_severity"
].dropna().values


moderate_values = ordered[
    "combined_moderate_or_higher_pct"
].dropna().values


strong_values = ordered[
    "combined_strong_or_higher_pct"
].dropna().values


print(
    "\nMonotonic increase checks:"
)


if len(mean_values) >= 2:

    mean_monotonic = np.all(
        np.diff(
            mean_values
        ) >= 0
    )

    print(
        "Mean severity non-decreasing:",
        mean_monotonic
    )


if len(moderate_values) >= 2:

    moderate_monotonic = np.all(
        np.diff(
            moderate_values
        ) >= 0
    )

    print(
        "Moderate+ severity non-decreasing:",
        moderate_monotonic
    )


if len(strong_values) >= 2:

    strong_monotonic = np.all(
        np.diff(
            strong_values
        ) >= 0
    )

    print(
        "Strong+ severity non-decreasing:",
        strong_monotonic
    )


# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)

print(
    "\nTrajectory file:"
)

print(
    TRAJECTORY_FILE
)

print(
    "\nSummary file:"
)

print(
    SUMMARY_FILE
)