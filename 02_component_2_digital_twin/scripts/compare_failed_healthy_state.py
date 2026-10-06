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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "failed_vs_healthy_health_state.csv"
)

CHUNK_SIZE = 500_000

# Number of healthy HDD observations sampled per calendar date
HEALTHY_SAMPLE_PER_DATE = 2000

TARGET_DAYS = [-30, -14, -7, -3, -1, 0]

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
# STANDARDIZED MEAN DIFFERENCE
# ============================================================

def standardized_mean_difference(
    group1,
    group2
):
    group1 = group1.dropna()
    group2 = group2.dropna()

    if len(group1) < 2 or len(group2) < 2:
        return np.nan

    mean1 = group1.mean()
    mean2 = group2.mean()

    var1 = group1.var()
    var2 = group2.var()

    pooled_sd = np.sqrt(
        (
            var1 + var2
        ) / 2
    )

    if pooled_sd == 0:
        return np.nan

    return (
        (mean1 - mean2)
        / pooled_sd
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAILED VS HEALTHY DIGITAL TWIN STATE VALIDATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # STEP 1
    # FIND ALL HDDs THAT FAILED IN Q1
    # --------------------------------------------------------

    print()
    print("STEP 1: Identifying failed HDDs...")
    print("-" * 70)

    failure_dates = {}

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=BASE_COLUMNS,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"Reading failure information "
            f"chunk {chunk_number}..."
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

            date = group["date"].min()

            if (
                serial not in failure_dates
                or date < failure_dates[serial]
            ):
                failure_dates[serial] = date

    failed_serials = set(
        failure_dates.keys()
    )

    print()
    print(
        f"Failed HDDs: "
        f"{len(failed_serials):,}"
    )

    # --------------------------------------------------------
    # STEP 2
    # DETERMINE TARGET CALENDAR DATES
    # --------------------------------------------------------

    print()
    print(
        "STEP 2: Determining calendar dates "
        "for failure-relative observations..."
    )
    print("-" * 70)

    target_date_map = {}

    for serial, failure_date in failure_dates.items():

        for relative_day in TARGET_DAYS:

            target_date = (
                failure_date
                + pd.Timedelta(
                    days=relative_day
                )
            )

            if target_date not in target_date_map:
                target_date_map[target_date] = []

            target_date_map[target_date].append(
                (
                    serial,
                    relative_day
                )
            )

    target_dates = set(
        target_date_map.keys()
    )

    print(
        f"Unique target calendar dates: "
        f"{len(target_dates):,}"
    )

    # --------------------------------------------------------
    # STEP 3
    # READ DATA AND COLLECT FAILED + HEALTHY
    # --------------------------------------------------------

    print()
    print(
        "STEP 3: Collecting failed and "
        "healthy comparison observations..."
    )
    print("-" * 70)

    selected_columns = (
        BASE_COLUMNS
        + STATE_FEATURES
        + AVAILABILITY_FEATURES
    )

    failed_rows = []

    healthy_samples = {}

    rng = np.random.default_rng(42)

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=selected_columns,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"Processing data chunk "
            f"{chunk_number}..."
        )

        chunk["date"] = pd.to_datetime(
            chunk["date"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # FAILED OBSERVATIONS
        # ----------------------------------------------------

        failed_chunk = chunk[
            chunk["serial_number"].isin(
                failed_serials
            )
        ].copy()

        if len(failed_chunk) > 0:

            failed_chunk["failure_date"] = (
                failed_chunk["serial_number"]
                .map(failure_dates)
            )

            failed_chunk["relative_day"] = (
                failed_chunk["date"]
                - failed_chunk["failure_date"]
            ).dt.days

            failed_chunk = failed_chunk[
                failed_chunk["relative_day"].isin(
                    TARGET_DAYS
                )
            ]

            if len(failed_chunk) > 0:
                failed_rows.append(
                    failed_chunk
                )

        # ----------------------------------------------------
        # HEALTHY OBSERVATIONS
        # ----------------------------------------------------

        healthy_chunk = chunk[
            ~chunk["serial_number"].isin(
                failed_serials
            )
        ].copy()

        healthy_chunk = healthy_chunk[
            healthy_chunk["date"].isin(
                target_dates
            )
        ]

        if len(healthy_chunk) > 0:

            for date, group in healthy_chunk.groupby(
                "date"
            ):

                if date not in healthy_samples:

                    healthy_samples[date] = []

                healthy_samples[date].append(
                    group
                )

    # --------------------------------------------------------
    # COMBINE FAILED DATA
    # --------------------------------------------------------

    if not failed_rows:
        print("ERROR: No failed observations found.")
        return

    failed_df = pd.concat(
        failed_rows,
        ignore_index=True
    )

    # --------------------------------------------------------
    # SAMPLE HEALTHY DATA BY DATE
    # --------------------------------------------------------

    print()
    print(
        "Sampling healthy observations "
        "by calendar date..."
    )

    healthy_rows = []

    for date, groups in healthy_samples.items():

        group = pd.concat(
            groups,
            ignore_index=True
        )

        if len(group) > HEALTHY_SAMPLE_PER_DATE:

            selected_indices = rng.choice(
                len(group),
                size=HEALTHY_SAMPLE_PER_DATE,
                replace=False
            )

            group = group.iloc[
                selected_indices
            ]

        healthy_rows.append(
            group
        )

    if not healthy_rows:
        print("ERROR: No healthy comparison data found.")
        return

    healthy_df = pd.concat(
        healthy_rows,
        ignore_index=True
    )

    print()
    print(
        f"Failed observations: "
        f"{len(failed_df):,}"
    )

    print(
        f"Healthy observations sampled: "
        f"{len(healthy_df):,}"
    )

    # --------------------------------------------------------
    # STEP 4
    # CALCULATE COMPARISON STATISTICS
    # --------------------------------------------------------

    print()
    print(
        "STEP 4: Calculating failed-vs-healthy "
        "state differences..."
    )
    print("-" * 70)

    results = []

    for relative_day in TARGET_DAYS:

        failed_at_day = failed_df[
            failed_df["relative_day"]
            == relative_day
        ]

        # Match healthy observations to the calendar
        # dates represented by failed observations.
        relevant_dates = set(
            failed_at_day["date"]
        )

        healthy_at_day = healthy_df[
            healthy_df["date"].isin(
                relevant_dates
            )
        ]

        for feature in STATE_FEATURES:

            failed_values = (
                failed_at_day[feature]
                .dropna()
            )

            healthy_values = (
                healthy_at_day[feature]
                .dropna()
            )

            if (
                len(failed_values) == 0
                or len(healthy_values) == 0
            ):
                continue

            results.append({

                "relative_day": relative_day,

                "feature": feature,

                "failed_n": len(
                    failed_values
                ),

                "healthy_n": len(
                    healthy_values
                ),

                "failed_mean": (
                    failed_values.mean()
                ),

                "healthy_mean": (
                    healthy_values.mean()
                ),

                "mean_difference": (
                    failed_values.mean()
                    - healthy_values.mean()
                ),

                "failed_median": (
                    failed_values.median()
                ),

                "healthy_median": (
                    healthy_values.median()
                ),

                "failed_p25": (
                    failed_values.quantile(0.25)
                ),

                "healthy_p25": (
                    healthy_values.quantile(0.25)
                ),

                "failed_p75": (
                    failed_values.quantile(0.75)
                ),

                "healthy_p75": (
                    healthy_values.quantile(0.75)
                ),

                "standardized_mean_difference": (
                    standardized_mean_difference(
                        failed_values,
                        healthy_values
                    )
                ),

            })

    # --------------------------------------------------------
    # AVAILABILITY COMPARISON
    # --------------------------------------------------------

    for relative_day in TARGET_DAYS:

        failed_at_day = failed_df[
            failed_df["relative_day"]
            == relative_day
        ]

        relevant_dates = set(
            failed_at_day["date"]
        )

        healthy_at_day = healthy_df[
            healthy_df["date"].isin(
                relevant_dates
            )
        ]

        for feature in AVAILABILITY_FEATURES:

            failed_values = (
                failed_at_day[feature]
            )

            healthy_values = (
                healthy_at_day[feature]
            )

            if (
                len(failed_values) == 0
                or len(healthy_values) == 0
            ):
                continue

            results.append({

                "relative_day": relative_day,

                "feature": feature,

                "failed_n": len(
                    failed_values
                ),

                "healthy_n": len(
                    healthy_values
                ),

                "failed_mean": (
                    failed_values.mean()
                ),

                "healthy_mean": (
                    healthy_values.mean()
                ),

                "mean_difference": (
                    failed_values.mean()
                    - healthy_values.mean()
                ),

                "failed_median": (
                    failed_values.median()
                ),

                "healthy_median": (
                    healthy_values.median()
                ),

                "failed_p25": (
                    failed_values.quantile(0.25)
                ),

                "healthy_p25": (
                    healthy_values.quantile(0.25)
                ),

                "failed_p75": (
                    failed_values.quantile(0.75)
                ),

                "healthy_p75": (
                    healthy_values.quantile(0.75)
                ),

                "standardized_mean_difference": (
                    standardized_mean_difference(
                        failed_values,
                        healthy_values
                    )
                ),

            })

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # STEP 5
    # PRINT IMPORTANT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FAILED VS HEALTHY COMPARISON")
    print("=" * 70)

    for relative_day in TARGET_DAYS:

        print()
        print(
            f"RELATIVE DAY: {relative_day}"
        )

        subset = results_df[
            results_df["relative_day"]
            == relative_day
        ]

        print(
            subset[
                [
                    "feature",
                    "failed_mean",
                    "healthy_mean",
                    "mean_difference",
                    "standardized_mean_difference"
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # SUMMARY AT -7 DAYS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY: -7 DAYS BEFORE FAILURE")
    print("=" * 70)

    summary = results_df[
        results_df["relative_day"] == -7
    ].copy()

    summary = summary.sort_values(
        "standardized_mean_difference",
        key=lambda x: abs(x),
        ascending=False
    )

    print(
        summary[
            [
                "feature",
                "failed_mean",
                "healthy_mean",
                "mean_difference",
                "standardized_mean_difference"
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)

    print()
    print("Output:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()