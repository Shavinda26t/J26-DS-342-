from pathlib import Path
import pandas as pd


# ============================================================
# Component 2 - Digital Twin
# Build Candidate Health-State Features
# ============================================================

INPUT_FILE = Path(
    "results/digital_twin_temporal_state.csv"
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "candidate_health_state_features.csv"
)


# ------------------------------------------------------------
# Candidate health-state variables
# ------------------------------------------------------------

HEALTH_STATE_FEATURES = [

    # SMART 1
    "smart_1_normalized",
    "smart_1_normalized_change_7d",
    "smart_1_normalized_rolling_std_7",

    # SMART 5
    "smart_5_normalized",
    "smart_5_normalized_change_7d",

    # SMART 187
    "smart_187_normalized",
    "smart_187_normalized_change_7d",
    "smart_187_normalized_rolling_std_7",

    # SMART 197
    "smart_197_normalized",
    "smart_197_normalized_change_7d",

    # SMART 198
    "smart_198_normalized",
    "smart_198_normalized_change_7d",
]


BASE_COLUMNS = [
    "date",
    "serial_number",
    "failure",
]


def main():

    print("=" * 70)
    print("COMPONENT 2 - BUILD CANDIDATE HEALTH-STATE FEATURES")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(f"\nInput: {INPUT_FILE}")

    # --------------------------------------------------------
    # Check available columns
    # --------------------------------------------------------

    header = pd.read_csv(
        INPUT_FILE,
        nrows=0
    )

    available_columns = set(
        header.columns
    )

    selected_columns = []

    for column in BASE_COLUMNS + HEALTH_STATE_FEATURES:

        if column in available_columns:

            selected_columns.append(column)

        else:

            print(
                f"WARNING: Missing column: {column}"
            )

    print(
        f"\nSelected columns: "
        f"{len(selected_columns)}"
    )

    # --------------------------------------------------------
    # Process in chunks
    # --------------------------------------------------------

    chunksize = 500_000

    first_chunk = True

    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=selected_columns,
            chunksize=chunksize
        ),
        start=1
    ):

        total_rows += len(chunk)

        print(
            f"Processing chunk {chunk_number}: "
            f"{len(chunk):,} rows"
        )

        # ----------------------------------------------------
        # Add SMART 187 observation availability
        # ----------------------------------------------------

        if "smart_187_normalized" in chunk.columns:

            chunk[
                "smart_187_available"
            ] = (
                chunk[
                    "smart_187_normalized"
                ]
                .notna()
                .astype("int8")
            )

        # ----------------------------------------------------
        # Add availability indicators for all core SMART
        # attributes
        # ----------------------------------------------------

        for smart_id in [1, 5, 187, 197, 198]:

            column = (
                f"smart_{smart_id}_normalized"
            )

            availability_column = (
                f"smart_{smart_id}_available"
            )

            if column in chunk.columns:

                chunk[
                    availability_column
                ] = (
                    chunk[column]
                    .notna()
                    .astype("int8")
                )

        # ----------------------------------------------------
        # Write output
        # ----------------------------------------------------

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    print("\n" + "=" * 70)
    print("HEALTH-STATE FEATURE DATASET CREATED")
    print("=" * 70)

    print(
        f"\nOutput: {OUTPUT_FILE}"
    )

    print(
        f"Total rows: {total_rows:,}"
    )

    print(
        "\nThis is a candidate state representation."
    )

    print(
        "It is NOT yet the final HDD health score."
    )


if __name__ == "__main__":
    main()