from pathlib import Path
import gc
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAUSAL_FINAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_final"
    / "ST12000NM0008_Q3_Q4"
)

COHORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_effect"
    / "pretreatment_failure_cohort.parquet"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pretreatment_cohort"
    / "pretreatment_adjustment_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_effect"
    / "matched_risksets_refined"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "riskset_matching_refined"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "causal_effect"
    / "riskset_matching_refined"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

TARGET = "failure_within_30d"

RANDOM_STATE = 42


TREATMENT_SOURCE_MAP = {

    "treat_reallocated":
        "reallocated_increase_1d",

    "treat_uncorrectable":
        "reported_uncorrectable_increase_1d",

    "treat_pending":
        "pending_sector_change_1d",
}


ISOLATED_FLAG_MAP = {

    "treat_reallocated":
        "isolated_reallocated_event",

    "treat_uncorrectable":
        "isolated_uncorrectable_event",

    "treat_pending":
        "isolated_pending_event",
}


TREATMENTS = list(
    TREATMENT_SOURCE_MAP.keys()
)


# ==========================================================
# PRIOR-DAY EVENT VARIABLES
# ==========================================================

PRIOR_DAY_EVENT_FEATURES = [

    "baseline_treat_reallocated_lag1",

    "baseline_treat_uncorrectable_lag1",

    "baseline_treat_pending_lag1",
]


# ==========================================================
# MATCHING VARIABLES
#
# All are measured at t-1.
# ==========================================================

MATCH_FEATURES = [

    "baseline_drive_age_hours_lag1",

    "baseline_temperature_lag1",

    "baseline_reallocated_sectors_lag1",

    "baseline_reported_uncorrectable_lag1",

    "baseline_pending_sectors_lag1",

    "baseline_crc_errors_lag1",

    "baseline_start_stop_activity_1d_lag1",

    "baseline_power_cycle_activity_1d_lag1",

    "baseline_poweroff_retract_activity_1d_lag1",

    "baseline_load_cycle_activity_1d_lag1",

    "baseline_write_workload_1d_lag1",

    "baseline_read_workload_1d_lag1",
]


# ==========================================================
# EXACT STATE VARIABLES
#
# Match whether each degradation state was already present
# before the current event.
# ==========================================================

STATE_FEATURES = [

    "baseline_reallocated_sectors_lag1",

    "baseline_reported_uncorrectable_lag1",

    "baseline_pending_sectors_lag1",
]


# ==========================================================
# LOAD BALANCE MANIFEST
# ==========================================================

manifest = pd.read_csv(
    MANIFEST_FILE
)


BALANCE_FEATURES = (
    manifest[
        "adjustment_feature"
    ]
    .tolist()
)


print(
    "Balance variables:",
    len(
        BALANCE_FEATURES
    )
)


# ==========================================================
# LOAD FULL CAUSAL DATA FOR TRUE OBSERVED EVENT HISTORY
# ==========================================================

history_files = sorted(
    CAUSAL_FINAL_DIR.glob("*.parquet")
)


if not history_files:

    raise FileNotFoundError(
        f"No causal-final files found in {CAUSAL_FINAL_DIR}"
    )


history_columns = [

    "date",

    "serial_number",

    "reallocated_increase_1d",

    "reported_uncorrectable_increase_1d",

    "pending_sector_change_1d",
]


history_parts = []


print(
    "\nLoading full causal-final history..."
)


for i, file in enumerate(
    history_files,
    start=1
):

    print(
        f"[{i}/{len(history_files)}] "
        f"Loading {file.name}"
    )


    temp = pd.read_parquet(
        file,
        columns=history_columns
    )


    history_parts.append(
        temp
    )


history = pd.concat(
    history_parts,
    ignore_index=True
)


del history_parts


history[
    "date"
] = pd.to_datetime(
    history[
        "date"
    ],
    errors="coerce"
)


history = history.dropna(
    subset=[
        "date",
        "serial_number",
    ]
)


history = (
    history
    .sort_values(
        [
            "serial_number",
            "date",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "History rows:",
    f"{len(history):,}"
)


print(
    "History HDDs:",
    f"{history['serial_number'].nunique():,}"
)


# ==========================================================
# DUPLICATE DATE CHECK
# ==========================================================

duplicate_history_rows = int(
    history.duplicated(
        subset=[
            "serial_number",
            "date",
        ]
    ).sum()
)


if duplicate_history_rows > 0:

    raise RuntimeError(
        "Duplicate serial/date rows found in causal history: "
        f"{duplicate_history_rows}"
    )


# ==========================================================
# RECONSTRUCT TREATMENT EVENTS FROM FULL HISTORY
# ==========================================================

for treatment, source in (
    TREATMENT_SOURCE_MAP.items()
):

    history[
        source
    ] = pd.to_numeric(
        history[
            source
        ],
        errors="coerce"
    )


    # Positive change = degradation event.
    #
    # NaN can include invalid/reset delta values.
    # It is NOT interpreted as a positive event.

    history[
        f"history_{treatment}"
    ] = (
        history[
            source
        ] > 0
    ).astype(
        "int8"
    )


# ==========================================================
# ANY DEGRADATION EVENT
# ==========================================================

HISTORY_EVENT_COLUMNS = [

    f"history_{treatment}"

    for treatment in TREATMENTS
]


history[
    "history_any_event"
] = (
    history[
        HISTORY_EVENT_COLUMNS
    ]
    .max(
        axis=1
    )
    .astype(
        "int8"
    )
)


# ==========================================================
# TRUE PRIOR OBSERVED EVENT COUNTS
#
# These are calculated BEFORE the pre-treatment cohort
# filtering.
# ==========================================================

for treatment in TREATMENTS:

    event_col = (
        f"history_{treatment}"
    )


    prior_col = (
        f"history_prior_count_{treatment}"
    )


    cumulative = (
        history
        .groupby(
            "serial_number",
            sort=False
        )[event_col]
        .cumsum()
    )


    history[
        prior_col
    ] = (
        cumulative
        -
        history[
            event_col
        ]
    )


# ==========================================================
# PRIOR ANY-EVENT COUNT
# ==========================================================

cumulative_any = (
    history
    .groupby(
        "serial_number",
        sort=False
    )[
        "history_any_event"
    ]
    .cumsum()
)


history[
    "history_prior_any_event_count"
] = (
    cumulative_any
    -
    history[
        "history_any_event"
    ]
)


# ==========================================================
# KEEP HISTORY FIELDS NEEDED FOR MERGE
# ==========================================================

history_keep_columns = [

    "date",

    "serial_number",

    "history_any_event",

    "history_prior_any_event_count",
]


for treatment in TREATMENTS:

    history_keep_columns.extend(
        [
            f"history_{treatment}",

            f"history_prior_count_{treatment}",
        ]
    )


history = history[
    history_keep_columns
].copy()


# ==========================================================
# LOAD PRE-TREATMENT COHORT
# ==========================================================

cohort_columns = (

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
    ]

    + BALANCE_FEATURES
)


cohort_columns = list(
    dict.fromkeys(
        cohort_columns
    )
)


print(
    "\nLoading pre-treatment cohort..."
)


cohort = pd.read_parquet(
    COHORT_FILE,
    columns=cohort_columns
)


cohort[
    "date"
] = pd.to_datetime(
    cohort[
        "date"
    ],
    errors="coerce"
)


cohort = (
    cohort
    .sort_values(
        [
            "serial_number",
            "date",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "Cohort rows:",
    f"{len(cohort):,}"
)


print(
    "Cohort HDDs:",
    f"{cohort['serial_number'].nunique():,}"
)


# ==========================================================
# COHORT DUPLICATE CHECK
# ==========================================================

duplicate_cohort_rows = int(
    cohort.duplicated(
        subset=[
            "serial_number",
            "date",
        ]
    ).sum()
)


if duplicate_cohort_rows > 0:

    raise RuntimeError(
        "Duplicate serial/date rows found in cohort: "
        f"{duplicate_cohort_rows}"
    )


# ==========================================================
# MERGE TRUE HISTORY INTO COHORT
# ==========================================================

cohort = cohort.merge(

    history,

    on=[
        "serial_number",
        "date",
    ],

    how="left",

    validate="one_to_one",
)


missing_history = int(
    cohort[
        "history_any_event"
    ].isna().sum()
)


if missing_history > 0:

    raise RuntimeError(
        "Missing full-history records after merge: "
        f"{missing_history}"
    )


# ==========================================================
# VALIDATE CURRENT TREATMENTS AGAINST HISTORY
# ==========================================================

for treatment in TREATMENTS:

    history_treatment = (
        f"history_{treatment}"
    )


    mismatch = int(
        (
            cohort[
                treatment
            ].astype(
                "int8"
            )
            !=
            cohort[
                history_treatment
            ].astype(
                "int8"
            )
        ).sum()
    )


    print(
        f"Current-event mismatch for {treatment}:",
        f"{mismatch:,}"
    )


    if mismatch > 0:

        raise RuntimeError(
            f"Current treatment mismatch found for "
            f"{treatment}: {mismatch}"
        )


# ==========================================================
# REQUIRE PRIOR-DAY FLAGS TO BE CLEAR
#
# In a clean first-event risk set, these should be zero.
# ==========================================================

prior_day_clear = (

    (
        cohort[
            "baseline_treat_reallocated_lag1"
        ]
        == 0
    )

    &

    (
        cohort[
            "baseline_treat_uncorrectable_lag1"
        ]
        == 0
    )

    &

    (
        cohort[
            "baseline_treat_pending_lag1"
        ]
        == 0
    )
)


print(
    "\nRows with all prior-day event flags clear:",
    f"{prior_day_clear.sum():,}"
)


# ==========================================================
# NUMERIC SAFETY
# ==========================================================

for feature in BALANCE_FEATURES:

    cohort[
        feature
    ] = pd.to_numeric(
        cohort[
            feature
        ],
        errors="coerce"
    )


if cohort[
    BALANCE_FEATURES
].isna().any().any():

    raise RuntimeError(
        "Missing balance variables found."
    )


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def mean_value(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )


    if len(
        values
    ) == 0:

        return np.nan


    return float(
        np.mean(
            values
        )
    )


def variance_value(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )


    if len(
        values
    ) == 0:

        return np.nan


    return float(
        np.var(
            values
        )
    )


def calculate_smd(
    treated_values,
    control_values
):

    treated_values = np.asarray(
        treated_values,
        dtype=np.float64
    )


    control_values = np.asarray(
        control_values,
        dtype=np.float64
    )


    treated_mean = mean_value(
        treated_values
    )


    control_mean = mean_value(
        control_values
    )


    treated_var = variance_value(
        treated_values
    )


    control_var = variance_value(
        control_values
    )


    pooled_sd = np.sqrt(
        (
            treated_var
            +
            control_var
        )
        / 2
    )


    if (
        not np.isfinite(
            pooled_sd
        )
        or
        pooled_sd == 0
    ):

        if np.isclose(
            treated_mean,
            control_mean
        ):

            return 0.0

        return np.nan


    return (
        treated_mean
        -
        control_mean
    ) / pooled_sd


def robust_center_scale(
    control_df,
    features
):

    centers = {}

    scales = {}


    for feature in features:

        values = (
            control_df[
                feature
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        median = float(
            np.median(
                values
            )
        )


        q25 = float(
            np.quantile(
                values,
                0.25
            )
        )


        q75 = float(
            np.quantile(
                values,
                0.75
            )
        )


        iqr = (
            q75
            -
            q25
        )


        # Convert IQR approximately to SD under normality.

        scale = (
            iqr
            /
            1.349
        )


        if (
            not np.isfinite(
                scale
            )
            or
            scale <= 0
        ):

            scale = float(
                np.std(
                    values
                )
            )


        if (
            not np.isfinite(
                scale
            )
            or
            scale <= 0
        ):

            scale = 1.0


        centers[
            feature
        ] = median


        scales[
            feature
        ] = scale


    return (
        pd.Series(
            centers
        ),

        pd.Series(
            scales
        ),
    )


# ==========================================================
# STORAGE
# ==========================================================

summary_rows = []

all_balance_frames = []

all_quality_rows = []

history_audit_rows = []


# ==========================================================
# ANALYSE EACH TREATMENT
# ==========================================================

for treatment in TREATMENTS:

    print("\n")
    print("=" * 76)

    print(
        f"HISTORY-CORRECTED MATCHING: {treatment}"
    )

    print("=" * 76)


    isolated_flag = (
        ISOLATED_FLAG_MAP[
            treatment
        ]
    )


    history_prior_specific = (
        f"history_prior_count_{treatment}"
    )


    # ======================================================
    # CLEAN FIRST-OBSERVED INCIDENT TREATMENT
    # ======================================================

    clean_treated_mask = (

        (
            cohort[
                treatment
            ] == 1
        )

        &

        (
            cohort[
                isolated_flag
            ] == 1
        )

        &

        (
            cohort[
                history_prior_specific
            ] == 0
        )

        &

        (
            cohort[
                "history_prior_any_event_count"
            ] == 0
        )

        &

        prior_day_clear
    )


    clean_treated = cohort[
        clean_treated_mask
    ].copy()


    print(
        "Clean incident treated rows:",
        f"{len(clean_treated):,}"
    )


    print(
        "Clean incident HDDs:",
        f"{clean_treated['serial_number'].nunique():,}"
    )


    if len(
        clean_treated
    ) == 0:

        print(
            "No clean treated events. Skipping."
        )

        continue


    # ======================================================
    # TREATMENT DATES
    # ======================================================

    treatment_dates = (
        clean_treated[
            "date"
        ]
        .drop_duplicates()
        .sort_values()
    )


    # ======================================================
    # TRUE AT-RISK CONTROLS
    #
    # Control must:
    #
    # 1. Be on same calendar dates.
    # 2. Have no event today.
    # 3. Have no previous observed event.
    # 4. Have no prior-day event flag.
    # ======================================================

    eligible_control_mask = (

        cohort[
            "date"
        ].isin(
            treatment_dates
        )

        &

        (
            cohort[
                "history_any_event"
            ] == 0
        )

        &

        (
            cohort[
                "history_prior_any_event_count"
            ] == 0
        )

        &

        prior_day_clear
    )


    eligible_controls = cohort[
        eligible_control_mask
    ].copy()


    print(
        "Eligible at-risk control rows:",
        f"{len(eligible_controls):,}"
    )


    print(
        "Eligible control HDDs:",
        f"{eligible_controls['serial_number'].nunique():,}"
    )


    # ======================================================
    # HISTORY AUDIT
    # ======================================================

    history_audit_rows.append(
        {
            "treatment":
                treatment,

            "clean_treated_rows":
                len(
                    clean_treated
                ),

            "clean_treated_hdds":
                clean_treated[
                    "serial_number"
                ].nunique(),

            "event_dates":
                len(
                    treatment_dates
                ),

            "eligible_control_rows":
                len(
                    eligible_controls
                ),

            "eligible_control_hdds":
                eligible_controls[
                    "serial_number"
                ].nunique(),
        }
    )


    # ======================================================
    # CREATE EXACT STATE KEY
    # ======================================================

    for data in [

        clean_treated,

        eligible_controls,
    ]:

        data[
            "state_reallocated_present"
        ] = (
            data[
                "baseline_reallocated_sectors_lag1"
            ] > 0
        ).astype(
            "int8"
        )


        data[
            "state_uncorrectable_present"
        ] = (
            data[
                "baseline_reported_uncorrectable_lag1"
            ] > 0
        ).astype(
            "int8"
        )


        data[
            "state_pending_present"
        ] = (
            data[
                "baseline_pending_sectors_lag1"
            ] > 0
        ).astype(
            "int8"
        )


        data[
            "exact_state_key"
        ] = (
            data[
                [
                    "state_reallocated_present",

                    "state_uncorrectable_present",

                    "state_pending_present",
                ]
            ]
            .astype(
                str
            )
            .agg(
                "_".join,
                axis=1
            )
        )


    # ======================================================
    # RISKSET FOR BEFORE-MATCHING BALANCE
    # ======================================================

    riskset = pd.concat(
        [
            clean_treated.assign(
                analysis_treatment=1
            ),

            eligible_controls.assign(
                analysis_treatment=0
            ),
        ],
        ignore_index=True,
    )


    treated_before = (
        riskset[
            "analysis_treatment"
        ]
        == 1
    )


    control_before = (
        riskset[
            "analysis_treatment"
        ]
        == 0
    )


    # ======================================================
    # BEFORE-MATCHING BALANCE
    # ======================================================

    before_smd = {}


    for feature in BALANCE_FEATURES:

        before_smd[
            feature
        ] = calculate_smd(

            riskset.loc[
                treated_before,
                feature
            ],

            riskset.loc[
                control_before,
                feature
            ],
        )


    # ======================================================
    # ROBUST SCALING
    #
    # Fit scaling using eligible controls only.
    # ======================================================

    centers, scales = robust_center_scale(

        eligible_controls,

        MATCH_FEATURES,
    )


    # ======================================================
    # 1:1 NEAREST-NEIGHBOUR MATCHING WITH REPLACEMENT
    #
    # Exact:
    # - date
    # - baseline degradation-state pattern
    #
    # Replacement improves balance for rare treatments.
    # ======================================================

    matched_rows = []

    quality_rows = []

    unmatched_rows = []

    match_set_id = 0


    for treated_index, treated_row in (
        clean_treated.iterrows()
    ):

        candidate_mask = (

            (
                eligible_controls[
                    "date"
                ]
                ==
                treated_row[
                    "date"
                ]
            )

            &

            (
                eligible_controls[
                    "exact_state_key"
                ]
                ==
                treated_row[
                    "exact_state_key"
                ]
            )

            &

            (
                eligible_controls[
                    "serial_number"
                ]
                !=
                treated_row[
                    "serial_number"
                ]
            )
        )


        candidates = (
            eligible_controls[
                candidate_mask
            ]
        )


        if len(
            candidates
        ) == 0:

            unmatched_rows.append(
                {
                    "treatment":
                        treatment,

                    "serial_number":
                        treated_row[
                            "serial_number"
                        ],

                    "date":
                        treated_row[
                            "date"
                        ],

                    "exact_state_key":
                        treated_row[
                            "exact_state_key"
                        ],

                    "reason":
                        "no_same_date_exact_state_control",
                }
            )

            continue


        treated_vector = (

            (
                treated_row[
                    MATCH_FEATURES
                ]
                .astype(
                    "float64"
                )
                -
                centers
            )

            /
            scales
        ).to_numpy(
            dtype=np.float64
        )


        candidate_matrix = (

            (
                candidates[
                    MATCH_FEATURES
                ]
                -
                centers
            )

            /
            scales
        ).to_numpy(
            dtype=np.float64
        )


        squared_distance = np.sum(

            (
                candidate_matrix
                -
                treated_vector
            ) ** 2,

            axis=1
        )


        best_position = int(
            np.argmin(
                squared_distance
            )
        )


        best_control = (
            candidates.iloc[
                best_position
            ]
        )


        distance = float(
            np.sqrt(
                squared_distance[
                    best_position
                ]
            )
        )


        match_set_id += 1


        # --------------------------------------------------
        # Treated row
        # --------------------------------------------------

        treated_output = (
            treated_row.copy()
        )


        treated_output[
            "match_set_id"
        ] = match_set_id


        treated_output[
            "match_role"
        ] = "treated"


        treated_output[
            "match_distance"
        ] = 0.0


        matched_rows.append(
            treated_output
        )


        # --------------------------------------------------
        # Control row
        # --------------------------------------------------

        control_output = (
            best_control.copy()
        )


        control_output[
            "match_set_id"
        ] = match_set_id


        control_output[
            "match_role"
        ] = "control"


        control_output[
            "match_distance"
        ] = distance


        matched_rows.append(
            control_output
        )


        quality_rows.append(
            {
                "treatment":
                    treatment,

                "match_set_id":
                    match_set_id,

                "treated_serial_number":
                    treated_row[
                        "serial_number"
                    ],

                "control_serial_number":
                    best_control[
                        "serial_number"
                    ],

                "date":
                    treated_row[
                        "date"
                    ],

                "exact_state_key":
                    treated_row[
                        "exact_state_key"
                    ],

                "distance":
                    distance,
            }
        )


    # ======================================================
    # CREATE MATCHED DATASET
    # ======================================================

    if not matched_rows:

        raise RuntimeError(
            f"No matches found for {treatment}"
        )


    matched_df = pd.DataFrame(
        matched_rows
    )


    matched_df[
        "date"
    ] = pd.to_datetime(
        matched_df[
            "date"
        ]
    )


    matched_treated = (
        matched_df[
            "match_role"
        ]
        == "treated"
    )


    matched_control = (
        matched_df[
            "match_role"
        ]
        == "control"
    )


    original_treated = len(
        clean_treated
    )


    matched_treated_n = int(
        matched_treated.sum()
    )


    matched_control_n = int(
        matched_control.sum()
    )


    unmatched_treated_n = (
        original_treated
        -
        matched_treated_n
    )


    retention_pct = (

        matched_treated_n

        /
        original_treated

        *
        100
    )


    # ======================================================
    # UNIQUE CONTROL USE / REPLACEMENT
    # ======================================================

    unique_matched_controls = (
        matched_df.loc[
            matched_control,
            "serial_number"
        ]
        .nunique()
    )


    # ======================================================
    # POST-MATCHING BALANCE
    # ======================================================

    balance_rows = []


    for feature in BALANCE_FEATURES:

        after = calculate_smd(

            matched_df.loc[
                matched_treated,
                feature
            ],

            matched_df.loc[
                matched_control,
                feature
            ],
        )


        before = (
            before_smd[
                feature
            ]
        )


        balance_rows.append(
            {
                "treatment":
                    treatment,

                "feature":
                    feature,

                "smd_before":
                    before,

                "abs_smd_before":
                    (
                        abs(
                            before
                        )
                        if np.isfinite(
                            before
                        )
                        else np.nan
                    ),

                "smd_after_matching":
                    after,

                "abs_smd_after_matching":
                    (
                        abs(
                            after
                        )
                        if np.isfinite(
                            after
                        )
                        else np.nan
                    ),
            }
        )


    balance_df = pd.DataFrame(
        balance_rows
    )


    balance_df = (
        balance_df
        .sort_values(
            "abs_smd_after_matching",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    max_smd_before = (
        balance_df[
            "abs_smd_before"
        ]
        .max()
    )


    max_smd_after = (
        balance_df[
            "abs_smd_after_matching"
        ]
        .max()
    )


    features_ge_010 = int(
        (
            balance_df[
                "abs_smd_after_matching"
            ]
            >= 0.10
        ).sum()
    )


    features_ge_005 = int(
        (
            balance_df[
                "abs_smd_after_matching"
            ]
            >= 0.05
        ).sum()
    )


    # ======================================================
    # MATCH DISTANCE
    # ======================================================

    quality_df = pd.DataFrame(
        quality_rows
    )


    median_distance = float(
        quality_df[
            "distance"
        ].median()
    )


    p95_distance = float(
        quality_df[
            "distance"
        ].quantile(
            0.95
        )
    )


    maximum_distance = float(
        quality_df[
            "distance"
        ].max()
    )


    # ======================================================
    # OUTCOME COUNTS
    #
    # Diagnostic only.
    # Formal estimation is the next step.
    # ======================================================

    treated_positive = int(
        (
            matched_df.loc[
                matched_treated,
                TARGET
            ]
            == 1
        ).sum()
    )


    control_positive = int(
        (
            matched_df.loc[
                matched_control,
                TARGET
            ]
            == 1
        ).sum()
    )


    treated_rate = float(
        matched_df.loc[
            matched_treated,
            TARGET
        ].mean()
    )


    control_rate = float(
        matched_df.loc[
            matched_control,
            TARGET
        ].mean()
    )


    diagnostic_risk_difference = (
        treated_rate
        -
        control_rate
    )


    # ======================================================
    # READINESS
    # ======================================================

    balance_ok = (
        max_smd_after
        < 0.10
    )


    retention_ok = (
        retention_pct
        >= 80.0
    )


    if (
        balance_ok
        and
        retention_ok
    ):

        status = (
            "READY_FOR_MATCHED_ATT_ESTIMATION"
        )

    else:

        status = (
            "REVIEW_BEFORE_EFFECT_ESTIMATION"
        )


    # ======================================================
    # SAVE MATCHED DATASET
    # ======================================================

    output_file = (
        OUTPUT_DIR
        / f"{treatment}_history_corrected_matched.parquet"
    )


    matched_df.to_parquet(
        output_file,
        index=False,
        compression="zstd"
    )


    # ======================================================
    # SAVE BALANCE
    # ======================================================

    balance_file = (
        AUDIT_DIR
        / f"{treatment}_balance.csv"
    )


    balance_df.to_csv(
        balance_file,
        index=False
    )


    # ======================================================
    # SAVE MATCH QUALITY
    # ======================================================

    quality_file = (
        AUDIT_DIR
        / f"{treatment}_match_quality.csv"
    )


    quality_df.to_csv(
        quality_file,
        index=False
    )


    # ======================================================
    # SAVE UNMATCHED
    # ======================================================

    unmatched_df = pd.DataFrame(
        unmatched_rows
    )


    unmatched_file = (
        AUDIT_DIR
        / f"{treatment}_unmatched.csv"
    )


    unmatched_df.to_csv(
        unmatched_file,
        index=False
    )


    # ======================================================
    # FIGURE
    # ======================================================

    plot_df = (
        balance_df
        .sort_values(
            "abs_smd_before",
            ascending=True
        )
    )


    y_positions = np.arange(
        len(
            plot_df
        )
    )


    plt.figure(
        figsize=(
            11,
            8
        )
    )


    plt.scatter(

        plot_df[
            "smd_before"
        ],

        y_positions,

        label="Before matching"
    )


    plt.scatter(

        plot_df[
            "smd_after_matching"
        ],

        y_positions,

        label="After matching"
    )


    plt.axvline(
        0,
        linewidth=1
    )


    plt.axvline(
        -0.10,
        linestyle="--",
        linewidth=1
    )


    plt.axvline(
        0.10,
        linestyle="--",
        linewidth=1
    )


    plt.yticks(

        y_positions,

        plot_df[
            "feature"
        ]
    )


    plt.xlabel(
        "Standardized Mean Difference (SMD)"
    )


    plt.title(
        f"History-corrected 1:1 matching: {treatment}"
    )


    plt.legend()


    plt.tight_layout()


    figure_file = (
        FIGURE_DIR
        / f"{treatment}_balance.png"
    )


    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ======================================================
    # SUMMARY
    # ======================================================

    summary_rows.append(
        {
            "treatment":
                treatment,

            "clean_incident_treated":
                original_treated,

            "matched_treated":
                matched_treated_n,

            "unmatched_treated":
                unmatched_treated_n,

            "retention_pct":
                retention_pct,

            "matched_control_rows":
                matched_control_n,

            "unique_matched_control_hdds":
                unique_matched_controls,

            "treated_positive_rows":
                treated_positive,

            "control_positive_rows":
                control_positive,

            "treated_failure_rate":
                treated_rate,

            "control_failure_rate":
                control_rate,

            "diagnostic_risk_difference":
                diagnostic_risk_difference,

            "max_abs_smd_before":
                max_smd_before,

            "max_abs_smd_after":
                max_smd_after,

            "features_abs_smd_ge_010_after":
                features_ge_010,

            "features_abs_smd_ge_005_after":
                features_ge_005,

            "median_match_distance":
                median_distance,

            "p95_match_distance":
                p95_distance,

            "maximum_match_distance":
                maximum_distance,

            "status":
                status,
        }
    )


    all_balance_frames.append(
        balance_df
    )


    quality_copy = (
        quality_df.copy()
    )


    all_quality_rows.append(
        quality_copy
    )


    # ======================================================
    # PRINT
    # ======================================================

    print(
        "\nMatching results:"
    )


    print(
        "Original clean treated:",
        original_treated
    )


    print(
        "Matched treated:",
        matched_treated_n
    )


    print(
        "Unmatched treated:",
        unmatched_treated_n
    )


    print(
        "Treated retention:",
        f"{retention_pct:.2f}%"
    )


    print(
        "Matched controls:",
        matched_control_n
    )


    print(
        "Unique matched control HDDs:",
        unique_matched_controls
    )


    print(
        "Treated positive outcomes:",
        treated_positive
    )


    print(
        "Control positive outcomes:",
        control_positive
    )


    print(
        "Maximum |SMD| before:",
        round(
            max_smd_before,
            4
        )
    )


    print(
        "Maximum |SMD| after:",
        round(
            max_smd_after,
            4
        )
    )


    print(
        "Features |SMD| >= 0.10 after:",
        features_ge_010
    )


    print(
        "Features |SMD| >= 0.05 after:",
        features_ge_005
    )


    print(
        "Median match distance:",
        round(
            median_distance,
            4
        )
    )


    print(
        "\nWorst post-matching balance:\n"
    )


    print(
        balance_df[
            [
                "feature",

                "smd_before",

                "smd_after_matching",
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )


    print(
        "\nStatus:",
        status
    )


    print(
        "Saved:",
        output_file
    )


    # ======================================================
    # CLEANUP
    # ======================================================

    del riskset
    del clean_treated
    del eligible_controls
    del matched_df
    del balance_df
    del quality_df

    gc.collect()


# ==========================================================
# FINAL SUMMARY
# ==========================================================

summary_df = pd.DataFrame(
    summary_rows
)


history_audit_df = pd.DataFrame(
    history_audit_rows
)


all_balance_df = pd.concat(
    all_balance_frames,
    ignore_index=True
)


all_quality_df = pd.concat(
    all_quality_rows,
    ignore_index=True
)


# ==========================================================
# SAVE FINAL AUDITS
# ==========================================================

summary_file = (
    AUDIT_DIR
    / "history_corrected_matching_summary.csv"
)


history_audit_file = (
    AUDIT_DIR
    / "history_corrected_riskset_summary.csv"
)


all_balance_file = (
    AUDIT_DIR
    / "history_corrected_balance_all.csv"
)


all_quality_file = (
    AUDIT_DIR
    / "history_corrected_match_quality_all.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


history_audit_df.to_csv(
    history_audit_file,
    index=False
)


all_balance_df.to_csv(
    all_balance_file,
    index=False
)


all_quality_df.to_csv(
    all_quality_file,
    index=False
)


# ==========================================================
# METADATA
# ==========================================================

metadata = {

    "design":
        "history-corrected exact-date incident risk-set matching",

    "estimand":
        "ATT among first-observed isolated degradation events",

    "matching_ratio":
        "1:1",

    "replacement":
        True,

    "exact_date":
        True,

    "exact_state_presence":
        STATE_FEATURES,

    "prior_day_event_flags_required_zero":
        PRIOR_DAY_EVENT_FEATURES,

    "distance_features":
        MATCH_FEATURES,

    "balance_features":
        BALANCE_FEATURES,

    "balance_threshold":
        0.10,

    "random_state":
        RANDOM_STATE,
}


metadata_file = (
    AUDIT_DIR
    / "history_corrected_matching_metadata.json"
)


with open(
    metadata_file,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================================
# FINAL PRINT
# ==========================================================

print("\n")
print("=" * 76)

print(
    "HISTORY-CORRECTED INCIDENT MATCHING COMPLETE"
)

print("=" * 76)


display_columns = [

    "treatment",

    "clean_incident_treated",

    "matched_treated",

    "unmatched_treated",

    "retention_pct",

    "treated_positive_rows",

    "control_positive_rows",

    "max_abs_smd_before",

    "max_abs_smd_after",

    "features_abs_smd_ge_010_after",

    "status",
]


print(
    "\nSummary:\n"
)


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\nMatched datasets:"
)


print(
    OUTPUT_DIR
)


print(
    "\nDiagnostics:"
)


print(
    AUDIT_DIR
)


print(
    "\nFigures:"
)


print(
    FIGURE_DIR
)