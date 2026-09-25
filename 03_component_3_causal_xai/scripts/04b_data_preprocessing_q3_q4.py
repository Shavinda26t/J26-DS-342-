from pathlib import Path
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIRS = {
    "Q3_2025": (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "Q4_2025"
        / "data_Q3_2025"
    ),

    "Q4_2025": (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "Q4_2025"
        / "data_Q4_2025"
    ),
}

Q4_AUDIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "smart_attribute_audit"
    / "ST12000NM0008_smart_attribute_audit.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
    / "ST12000NM0008_Q3_Q4"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "combined_preprocessing"
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

SELECTED_MODEL = "ST12000NM0008"


# ==========================================================
# LOAD Q4 SMART AUDIT
# ==========================================================

audit = pd.read_csv(
    Q4_AUDIT_FILE
)

usable_smart_columns = (
    audit.loc[
        (audit["missing_pct"] < 1.0)
        & (audit["unique_values"] > 1),
        "smart_attribute"
    ]
    .tolist()
)

print(
    "SMART attributes retained:",
    len(usable_smart_columns)
)


# ==========================================================
# METADATA COLUMNS
# ==========================================================

metadata_candidates = [
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
    "datacenter",
    "cluster_id",
    "vault_id",
    "pod_id",
    "pod_slot_num",
    "is_legacy_format",
]


# ==========================================================
# COUNTERS
# ==========================================================

total_raw_rows = 0
total_selected_rows = 0
total_output_rows = 0
total_invalid_removed = 0
total_duplicates_removed = 0

quarter_summary = []


# ==========================================================
# PROCESS BOTH QUARTERS
# ==========================================================

for quarter, raw_dir in RAW_DIRS.items():

    print("\n" + "=" * 70)
    print(f"PROCESSING {quarter}")
    print("=" * 70)

    csv_files = sorted(
        raw_dir.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}"
        )

    print(
        "CSV files:",
        len(csv_files)
    )

    quarter_raw = 0
    quarter_selected = 0
    quarter_output = 0
    quarter_invalid = 0
    quarter_duplicates = 0

    for i, file in enumerate(
        csv_files,
        start=1
    ):

        print(
            f"[{quarter} {i}/{len(csv_files)}] "
            f"{file.name}"
        )

        # --------------------------------------------------
        # READ AVAILABLE COLUMNS
        # --------------------------------------------------

        available_columns = (
            pd.read_csv(
                file,
                nrows=0
            )
            .columns
            .tolist()
        )

        metadata_columns = [
            col
            for col in metadata_candidates
            if col in available_columns
        ]

        smart_columns = [
            col
            for col in usable_smart_columns
            if col in available_columns
        ]

        usecols = (
            metadata_columns
            + smart_columns
        )

        # --------------------------------------------------
        # LOAD DAILY FILE
        # --------------------------------------------------

        df = pd.read_csv(
            file,
            usecols=usecols,
            low_memory=False
        )

        quarter_raw += len(df)
        total_raw_rows += len(df)

        # --------------------------------------------------
        # SELECT HDD MODEL
        # --------------------------------------------------

        df = df[
            df["model"] == SELECTED_MODEL
        ].copy()

        if df.empty:
            continue

        quarter_selected += len(df)
        total_selected_rows += len(df)

        # --------------------------------------------------
        # CLEAN DATE
        # --------------------------------------------------

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        # --------------------------------------------------
        # CLEAN SERIAL NUMBER
        # --------------------------------------------------

        df["serial_number"] = (
            df["serial_number"]
            .astype("string")
            .str.strip()
        )

        # --------------------------------------------------
        # CLEAN FAILURE LABEL
        # --------------------------------------------------

        df["failure"] = (
            pd.to_numeric(
                df["failure"],
                errors="coerce"
            )
            .fillna(0)
            .astype("int8")
        )

        # --------------------------------------------------
        # SMART VALUES -> NUMERIC
        # --------------------------------------------------

        for col in smart_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # --------------------------------------------------
        # REMOVE INVALID KEY ROWS
        # --------------------------------------------------

        before = len(df)

        df = df.dropna(
            subset=[
                "date",
                "serial_number"
            ]
        )

        invalid_removed = (
            before - len(df)
        )

        quarter_invalid += (
            invalid_removed
        )

        total_invalid_removed += (
            invalid_removed
        )

        # --------------------------------------------------
        # REMOVE DUPLICATE DRIVE-DATE ROWS
        # --------------------------------------------------

        before = len(df)

        df = df.drop_duplicates(
            subset=[
                "serial_number",
                "date"
            ],
            keep="last"
        )

        duplicates_removed = (
            before - len(df)
        )

        quarter_duplicates += (
            duplicates_removed
        )

        total_duplicates_removed += (
            duplicates_removed
        )

        # --------------------------------------------------
        # SORT
        # --------------------------------------------------

        df = df.sort_values(
            [
                "serial_number",
                "date"
            ]
        )

        # --------------------------------------------------
        # SAVE DAILY PARQUET
        # --------------------------------------------------

        output_file = (
            OUTPUT_DIR
            / f"{file.stem}.parquet"
        )

        df.to_parquet(
            output_file,
            index=False
        )

        quarter_output += len(df)
        total_output_rows += len(df)

    # ======================================================
    # QUARTER SUMMARY
    # ======================================================

    quarter_summary.append(
        {
            "quarter": quarter,
            "raw_rows": quarter_raw,
            "selected_model_rows":
                quarter_selected,
            "invalid_rows_removed":
                quarter_invalid,
            "duplicate_rows_removed":
                quarter_duplicates,
            "output_rows":
                quarter_output,
        }
    )


# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary_df = pd.DataFrame(
    quarter_summary
)

summary_file = (
    AUDIT_DIR
    / "Q3_Q4_preprocessing_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ==========================================================
# VERIFY DATE RANGE
# ==========================================================

output_files = sorted(
    OUTPUT_DIR.glob("*.parquet")
)

dates = []

for file in output_files:

    temp = pd.read_parquet(
        file,
        columns=["date"]
    )

    dates.append(
        temp["date"].min()
    )

    dates.append(
        temp["date"].max()
    )


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n")
print("=" * 70)
print("Q3 + Q4 PREPROCESSING COMPLETE")
print("=" * 70)

print(
    "\nQuarter summary:\n"
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\nTotal raw rows:",
    f"{total_raw_rows:,}"
)

print(
    "Total selected ST12000NM0008 rows:",
    f"{total_selected_rows:,}"
)

print(
    "Invalid rows removed:",
    f"{total_invalid_removed:,}"
)

print(
    "Duplicate rows removed:",
    f"{total_duplicates_removed:,}"
)

print(
    "Final combined rows:",
    f"{total_output_rows:,}"
)

print(
    "SMART attributes retained:",
    len(usable_smart_columns)
)

if dates:

    print(
        "Combined date range:",
        min(dates),
        "to",
        max(dates)
    )

print(
    "Daily Parquet files:",
    len(output_files)
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSummary saved to:"
)

print(
    summary_file
)