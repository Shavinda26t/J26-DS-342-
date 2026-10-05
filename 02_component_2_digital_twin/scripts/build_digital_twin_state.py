from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    r"D:\Project\J26-DS-342-\02_component_2_digital_twin"
    r"\data\interim\temporal_features.csv"
)

OUTPUT_FILE = Path(
    r"D:\Project\J26-DS-342-\02_component_2_digital_twin"
    r"\results\digital_twin_temporal_state.csv"
)

SMART_ATTRIBUTES = [1, 5, 187, 197, 198]

BASE_COLUMNS = [
    "date",
    "serial_number",
    "failure",
]

TEMPORAL_SUFFIXES = [
    "",
    "_previous",
    "_delta",
    "_change_7d",
    "_change_14d",
    "_rolling_mean_7",
    "_rolling_std_7",
]

CHUNK_SIZE = 500_000


def main():

    print("=" * 60)
    print("BUILDING DIGITAL TWIN TEMPORAL STATE")
    print("=" * 60)

    print(f"Input file:")
    print(INPUT_FILE)
    print()

    # ---------------------------------------------------------
    # 1. Check which required columns exist
    # ---------------------------------------------------------
    header = pd.read_csv(INPUT_FILE, nrows=0)
    available_columns = set(header.columns)

    selected_columns = BASE_COLUMNS.copy()

    for smart_id in SMART_ATTRIBUTES:

        prefix = f"smart_{smart_id}_normalized"

        for suffix in TEMPORAL_SUFFIXES:

            column = prefix + suffix

            if column in available_columns:
                selected_columns.append(column)

    selected_columns = list(dict.fromkeys(selected_columns))

    print(f"Selected columns: {len(selected_columns)}")
    print()

    # ---------------------------------------------------------
    # 2. Process CSV in chunks
    # ---------------------------------------------------------
    chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=selected_columns,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(f"Processing chunk {chunk_number}...")

        chunks.append(chunk)

    # ---------------------------------------------------------
    # 3. Combine chunks
    # ---------------------------------------------------------
    print()
    print("Combining temporal state data...")

    state = pd.concat(
        chunks,
        ignore_index=True
    )

    # ---------------------------------------------------------
    # 4. Convert date
    # ---------------------------------------------------------
    state["date"] = pd.to_datetime(
        state["date"]
    )

    # ---------------------------------------------------------
    # 5. Add missingness indicators
    # ---------------------------------------------------------
    print("Creating SMART missingness indicators...")

    for smart_id in SMART_ATTRIBUTES:

        value_column = f"smart_{smart_id}_normalized"

        if value_column in state.columns:

            state[f"smart_{smart_id}_missing"] = (
                state[value_column]
                .isna()
                .astype("int8")
            )

    # ---------------------------------------------------------
    # 6. Sort by HDD and date
    # ---------------------------------------------------------
    print("Sorting temporal state...")

    state = state.sort_values(
        ["serial_number", "date"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 7. Save Digital Twin state dataset
    # ---------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Saving Digital Twin temporal state...")

    state.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("DIGITAL TWIN TEMPORAL STATE CREATED")
    print("=" * 60)

    print(f"Rows: {len(state):,}")
    print(f"Columns: {len(state.columns):,}")
    print(
        f"Unique HDDs: "
        f"{state['serial_number'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{state['date'].min()} → "
        f"{state['date'].max()}"
    )

    print()
    print("SMART attributes:")
    for smart_id in SMART_ATTRIBUTES:
        print(f"  SMART {smart_id}")

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()