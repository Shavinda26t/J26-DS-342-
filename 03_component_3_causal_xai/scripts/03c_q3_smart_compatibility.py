from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

Q3_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Q4_2025"
    / "data_Q3_2025"
)

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
    / "interim"
    / "q3_smart_compatibility"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SELECTED_MODEL = "ST12000NM0008"


# ==========================================================
# LOAD SMART ATTRIBUTES RETAINED FROM Q4
# ==========================================================

q4_audit = pd.read_csv(Q4_AUDIT_FILE)

q4_usable = (
    q4_audit.loc[
        (q4_audit["missing_pct"] < 1.0)
        & (q4_audit["unique_values"] > 1),
        "smart_attribute"
    ]
    .tolist()
)

print(
    "Q4 usable SMART attributes:",
    len(q4_usable)
)


# ==========================================================
# FIND Q3 FILES
# ==========================================================

files = sorted(
    Q3_DIR.glob("*.csv")
)

if not files:
    raise FileNotFoundError(
        f"No Q3 CSV files found in {Q3_DIR}"
    )

print(
    "Q3 CSV files:",
    len(files)
)


# ==========================================================
# CHECK COLUMN EXISTENCE
# ==========================================================

first_columns = (
    pd.read_csv(
        files[0],
        nrows=0
    )
    .columns
    .tolist()
)

missing_columns = [
    col
    for col in q4_usable
    if col not in first_columns
]

if missing_columns:

    print(
        "\nWARNING - Q4 SMART columns missing in Q3:"
    )

    for col in missing_columns:
        print(col)

else:
    print(
        "\nAll Q4 retained SMART columns exist in Q3."
    )


usable_in_q3 = [
    col
    for col in q4_usable
    if col in first_columns
]


# ==========================================================
# STATISTICS
# ==========================================================

stats = {
    col: {
        "total": 0,
        "missing": 0,
        "non_zero": 0,
        "unique_values": set()
    }
    for col in usable_in_q3
}

selected_rows = 0


# ==========================================================
# PROCESS Q3 FILES
# ==========================================================

for i, file in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] "
        f"{file.name}"
    )

    usecols = (
        ["model"]
        + usable_in_q3
    )

    df = pd.read_csv(
        file,
        usecols=usecols,
        low_memory=False
    )

    df = df[
        df["model"] == SELECTED_MODEL
    ]

    if df.empty:
        continue

    selected_rows += len(df)

    for col in usable_in_q3:

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        stats[col]["total"] += len(values)

        stats[col]["missing"] += (
            values.isna().sum()
        )

        valid = (
            values.dropna()
        )

        stats[col]["non_zero"] += (
            valid != 0
        ).sum()

        if len(
            stats[col]["unique_values"]
        ) < 10000:

            remaining = (
                10000
                - len(
                    stats[col][
                        "unique_values"
                    ]
                )
            )

            stats[col][
                "unique_values"
            ].update(
                valid.unique()[
                    :remaining
                ]
            )


# ==========================================================
# CREATE SUMMARY
# ==========================================================

rows = []

for col, s in stats.items():

    total = s["total"]

    if total == 0:
        continue

    rows.append(
        {
            "smart_attribute": col,

            "total_rows":
                total,

            "missing_rows":
                s["missing"],

            "missing_pct":
                round(
                    s["missing"]
                    / total
                    * 100,
                    6
                ),

            "non_zero_rows":
                s["non_zero"],

            "non_zero_pct":
                round(
                    s["non_zero"]
                    / total
                    * 100,
                    6
                ),

            "unique_values":
                len(
                    s["unique_values"]
                )
        }
    )


summary = pd.DataFrame(
    rows
)

summary = summary.sort_values(
    [
        "missing_pct",
        "smart_attribute"
    ]
)


output_file = (
    OUTPUT_DIR
    / "Q3_ST12000NM0008_smart_compatibility.csv"
)

summary.to_csv(
    output_file,
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("Q3 SMART COMPATIBILITY CHECK")
print("=" * 70)

print(
    "Selected model:",
    SELECTED_MODEL
)

print(
    "Selected Q3 rows:",
    f"{selected_rows:,}"
)

print(
    "Q4 usable SMART attributes:",
    len(q4_usable)
)

print(
    "SMART attributes present in Q3:",
    len(usable_in_q3)
)

print(
    "Missing Q3 SMART columns:",
    len(missing_columns)
)

print(
    "\nAttributes with > 1% missing:"
)

problematic = summary[
    summary["missing_pct"] > 1.0
]

if problematic.empty:
    print("None")
else:
    print(
        problematic.to_string(
            index=False
        )
    )

print(
    "\nSaved to:"
)

print(
    output_file
)