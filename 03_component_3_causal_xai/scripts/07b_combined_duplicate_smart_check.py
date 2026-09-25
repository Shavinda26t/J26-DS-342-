from pathlib import Path
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_clean"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "combined_smart_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SMART PAIRS TO VERIFY
# ==========================================================

PAIRS = [
    ("smart_1_raw", "smart_195_raw"),
    ("smart_190_raw", "smart_194_raw"),
    ("smart_197_raw", "smart_198_raw"),
]


# ==========================================================
# FIND FILES
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:
    raise FileNotFoundError(
        f"No Parquet files found in {INPUT_DIR}"
    )


stats = {
    pair: {
        "both_available": 0,
        "equal": 0,
        "different": 0,
        "only_one_missing": 0
    }
    for pair in PAIRS
}


# ==========================================================
# PROCESS FILES
# ==========================================================

columns_needed = sorted(
    {
        col
        for pair in PAIRS
        for col in pair
    }
)


for i, file in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] "
        f"{file.name}"
    )

    df = pd.read_parquet(
        file,
        columns=columns_needed
    )

    for a, b in PAIRS:

        x = pd.to_numeric(
            df[a],
            errors="coerce"
        )

        y = pd.to_numeric(
            df[b],
            errors="coerce"
        )

        both = (
            x.notna()
            & y.notna()
        )

        one_missing = (
            x.isna()
            ^ y.isna()
        )

        equal = (
            x[both]
            == y[both]
        )

        stats[(a, b)][
            "both_available"
        ] += int(
            both.sum()
        )

        stats[(a, b)][
            "equal"
        ] += int(
            equal.sum()
        )

        stats[(a, b)][
            "different"
        ] += int(
            (~equal).sum()
        )

        stats[(a, b)][
            "only_one_missing"
        ] += int(
            one_missing.sum()
        )


# ==========================================================
# BUILD RESULT
# ==========================================================

rows = []

for (a, b), values in stats.items():

    both = values[
        "both_available"
    ]

    equality_pct = (
        values["equal"]
        / both
        * 100
        if both > 0
        else 0
    )

    rows.append(
        {
            "feature_a":
                a,

            "feature_b":
                b,

            "both_available":
                both,

            "equal_rows":
                values["equal"],

            "different_rows":
                values["different"],

            "equality_pct":
                round(
                    equality_pct,
                    6
                ),

            "only_one_missing":
                values[
                    "only_one_missing"
                ]
        }
    )


result = pd.DataFrame(
    rows
)


output_file = (
    OUTPUT_DIR
    / "combined_duplicate_smart_check.csv"
)

result.to_csv(
    output_file,
    index=False
)


# ==========================================================
# PRINT
# ==========================================================

print("\n")
print("=" * 70)
print("COMBINED SMART DUPLICATE CHECK")
print("=" * 70)

print(
    result.to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    output_file
)