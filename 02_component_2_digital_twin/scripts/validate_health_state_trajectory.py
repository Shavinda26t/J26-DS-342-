from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"D:\Project\J26-DS-342-\02_component_2_digital_twin"
)

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "normalized_health_state_features.csv"
)

OUTPUT_DIR = BASE_DIR / "results"

TRAJECTORY_OUTPUT = (
    OUTPUT_DIR
    / "health_state_failure_trajectory.csv"
)

PAIRED_OUTPUT = (
    OUTPUT_DIR
    / "health_state_paired_degradation.csv"
)

CHUNK_SIZE = 500_000

# Relative positions before failure
TARGET_DAYS = [-30, -14, -7, -3, -1, 0]

# Health-state features
STATE_FEATURES = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
    "smart_1_normalized_change_7d",
    "smart_5_normalized_change_7d",
    "smart_187_normalized_change_7d",
    "smart_197_normalized_change_7d",
    "smart_198_normalized_change_7d",
]

AVAILABILITY_FEATURES = [
    "smart_1_available",
    "smart_5_available",
    "smart_187_available",
    "smart_197_available",
    "smart_198_available",
]

BASE_COLUMNS = [
    "date",
    "serial_number",
    "failure",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DIGITAL TWIN HEALTH-STATE TRAJECTORY VALIDATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. FIND FAILURE DATES
    # --------------------------------------------------------

    print("\nSTEP 1: Finding HDD failure dates...")
    print("-" * 70)

    failure_dates = {}

    usecols = [
        "date",
        "serial_number",
        "failure",
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=usecols,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"Reading failure information chunk "
            f"{chunk_number}..."
        )

        chunk["date"] = pd.to_datetime(
            chunk["date"],
            errors="coerce"
        )

        failed = chunk[
            chunk["failure"] == 1
        ]

        for serial, group in failed.groupby(
            "serial_number"
        ):

            failure_date = group["date"].min()

            if serial not in failure_dates:
                failure_dates[serial] = failure_date

            else:
                if failure_date < failure_dates[serial]:
                    failure_dates[serial] = failure_date

    print()
    print(
        f"Unique failed HDDs found: "
        f"{len(failure_dates):,}"
    )

    # --------------------------------------------------------
    # 2. READ STATE DATA AROUND FAILURE
    # --------------------------------------------------------

    print("\nSTEP 2: Extracting failure trajectories...")
    print("-" * 70)

    selected_columns = (
        BASE_COLUMNS
        + STATE_FEATURES
        + AVAILABILITY_FEATURES
    )

    trajectory_chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=selected_columns,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"Processing trajectory chunk "
            f"{chunk_number}..."
        )

        chunk["date"] = pd.to_datetime(
            chunk["date"],
            errors="coerce"
        )

        # Keep only failed HDDs
        chunk = chunk[
            chunk["serial_number"].isin(
                failure_dates.keys()
            )
        ].copy()

        if len(chunk) == 0:
            continue

        # Map failure date
        chunk["failure_date"] = (
            chunk["serial_number"]
            .map(failure_dates)
        )

        # Calculate relative day
        chunk["relative_day"] = (
            chunk["date"] -
            chunk["failure_date"]
        ).dt.days

        # Keep only required windows
        chunk = chunk[
            chunk["relative_day"].isin(
                TARGET_DAYS
            )
        ]

        if len(chunk) > 0:
            trajectory_chunks.append(chunk)

    if not trajectory_chunks:
        print("ERROR: No trajectory data found.")
        return

    trajectory = pd.concat(
        trajectory_chunks,
        ignore_index=True
    )

    trajectory = trajectory.sort_values(
        [
            "serial_number",
            "relative_day"
        ]
    )

    print()
    print(
        f"Trajectory observations: "
        f"{len(trajectory):,}"
    )

    # --------------------------------------------------------
    # 3. POPULATION TRAJECTORY
    # --------------------------------------------------------

    print("\nSTEP 3: Calculating population trajectories...")
    print("-" * 70)

    trajectory_results = []

    for feature in (
        STATE_FEATURES
        + AVAILABILITY_FEATURES
    ):

        for relative_day in TARGET_DAYS:

            values = trajectory.loc[
                trajectory["relative_day"] == relative_day,
                feature
            ]

            values = values.dropna()

            if len(values) == 0:
                continue

            trajectory_results.append({

                "feature": feature,

                "relative_day": relative_day,

                "sample_count": len(values),

                "mean": values.mean(),

                "median": values.median(),

                "std": values.std(),

                "p25": values.quantile(0.25),

                "p75": values.quantile(0.75),

            })

    trajectory_summary = pd.DataFrame(
        trajectory_results
    )

    trajectory_summary.to_csv(
        TRAJECTORY_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # 4. PAIRED -30 TO -1 COMPARISON
    # --------------------------------------------------------

    print("\nSTEP 4: Calculating paired degradation...")
    print("-" * 70)

    early = trajectory[
        trajectory["relative_day"] == -30
    ].set_index("serial_number")

    late = trajectory[
        trajectory["relative_day"] == -1
    ].set_index("serial_number")

    paired_results = []

    for feature in STATE_FEATURES:

        if feature not in early.columns:
            continue

        paired = pd.DataFrame({

            "early": early[feature],

            "late": late[feature],

        }).dropna()

        if len(paired) == 0:
            continue

        change = (
            paired["late"] -
            paired["early"]
        )

        paired_results.append({

            "feature": feature,

            "paired_hdds": len(paired),

            "decreased_count": (
                (change < 0).sum()
            ),

            "decreased_percentage": (
                (change < 0).mean() * 100
            ),

            "unchanged_count": (
                (change == 0).sum()
            ),

            "unchanged_percentage": (
                (change == 0).mean() * 100
            ),

            "increased_count": (
                (change > 0).sum()
            ),

            "increased_percentage": (
                (change > 0).mean() * 100
            ),

            "mean_change": change.mean(),

            "median_change": change.median(),

            "std_change": change.std(),

        })

    paired_summary = pd.DataFrame(
        paired_results
    )

    paired_summary.to_csv(
        PAIRED_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # 5. PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FAILURE TRAJECTORY SUMMARY")
    print("=" * 70)

    for feature in STATE_FEATURES:

        print()
        print(f"FEATURE: {feature}")

        feature_data = trajectory_summary[
            trajectory_summary["feature"] == feature
        ]

        print(
            feature_data[
                [
                    "relative_day",
                    "sample_count",
                    "mean",
                    "median"
                ]
            ].to_string(index=False)
        )

    print()
    print("=" * 70)
    print("PAIRED -30 → -1 DAY DEGRADATION")
    print("=" * 70)

    print(
        paired_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)

    print()
    print("Trajectory output:")
    print(TRAJECTORY_OUTPUT)

    print()
    print("Paired degradation output:")
    print(PAIRED_OUTPUT)


if __name__ == "__main__":
    main()