from pathlib import Path
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Q4_2025"
    / "data_Q4_2025"
)

AUDIT_FILE = (
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
    / "ST12000NM0008"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

SELECTED_MODEL = "ST12000NM0008"


# ==========================================================
# LOAD SMART ATTRIBUTE AUDIT
# ==========================================================

audit = pd.read_csv(AUDIT_FILE)

# Keep:
# 1. SMART attributes with < 1% missing values
# 2. Attributes with more than one unique value
#
# This automatically removes:
# - unsupported SMART fields
# - constant SMART fields

usable_smart_columns = (
    audit.loc[
        (audit["missing_pct"] < 1.0)
        & (audit["unique_values"] > 1),
        "smart_attribute"
    ]
    .tolist()
)

print(
    "Usable SMART columns:",
    len(usable_smart_columns)
)

print("\nSMART columns retained:")

for col in usable_smart_columns:
    print(" -", col)


# ==========================================================
# BASIC METADATA
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
    "is_legacy_format"
]


# ==========================================================
# FIND INPUT FILES
# ==========================================================

csv_files = sorted(
    RAW_DATA_DIR.glob("*.csv")
)

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {RAW_DATA_DIR}"
    )

print(
    "\nCSV files found:",
    len(csv_files)
)


# ==========================================================
# COUNTERS
# ==========================================================

total_input_rows = 0
total_selected_rows = 0
total_output_rows = 0
total_duplicates_removed = 0
total_invalid_rows_removed = 0


# ==========================================================
# PROCESS FILE BY FILE
# ==========================================================

for i, file in enumerate(
    csv_files,
    start=1
):

    print(
        f"[{i}/{len(csv_files)}] "
        f"Processing {file.name}"
    )

    # Read header first
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

    smart_columns_for_file = [
        col
        for col in usable_smart_columns
        if col in available_columns
    ]

    usecols = (
        metadata_columns
        + smart_columns_for_file
    )

    df = pd.read_csv(
        file,
        usecols=usecols,
        low_memory=False
    )

    total_input_rows += len(df)


    # ======================================================
    # FILTER SELECTED HDD MODEL
    # ======================================================

    df = df[
        df["model"] == SELECTED_MODEL
    ].copy()

    if df.empty:
        continue

    total_selected_rows += len(df)


    # ======================================================
    # CLEAN DATE
    # ======================================================

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    # ======================================================
    # CLEAN SERIAL NUMBER
    # ======================================================

    df["serial_number"] = (
        df["serial_number"]
        .astype("string")
        .str.strip()
    )


    # ======================================================
    # CLEAN FAILURE LABEL
    # ======================================================

    df["failure"] = (
        pd.to_numeric(
            df["failure"],
            errors="coerce"
        )
        .fillna(0)
        .astype("int8")
    )


    # ======================================================
    # SMART VALUES -> NUMERIC
    # ======================================================

    for col in smart_columns_for_file:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


    # ======================================================
    # REMOVE INVALID KEY ROWS
    # ======================================================

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

    total_invalid_rows_removed += (
        invalid_removed
    )


    # ======================================================
    # REMOVE DUPLICATE HDD-DATE ROWS
    # ======================================================

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

    total_duplicates_removed += (
        duplicates_removed
    )


    # ======================================================
    # SORT DATA
    # ======================================================

    df = df.sort_values(
        by=[
            "serial_number",
            "date"
        ]
    )


    # ======================================================
    # SAVE AS PARQUET
    # ======================================================

    output_file = (
        OUTPUT_DIR
        / f"{file.stem}.parquet"
    )

    df.to_parquet(
        output_file,
        index=False
    )

    total_output_rows += len(df)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)

print(
    f"Raw input rows: "
    f"{total_input_rows:,}"
)

print(
    f"Rows for {SELECTED_MODEL}: "
    f"{total_selected_rows:,}"
)

print(
    f"Invalid key rows removed: "
    f"{total_invalid_rows_removed:,}"
)

print(
    f"Duplicate HDD-date rows removed: "
    f"{total_duplicates_removed:,}"
)

print(
    f"Final output rows: "
    f"{total_output_rows:,}"
)

print(
    f"SMART attributes retained: "
    f"{len(usable_smart_columns)}"
)

print("\nOutput directory:")

print(
    OUTPUT_DIR
)

print(
    "\nPreprocessing stage completed."
)