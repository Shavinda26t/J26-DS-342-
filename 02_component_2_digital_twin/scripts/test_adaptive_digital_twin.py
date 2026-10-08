import pandas as pd

from adaptive_digital_twin import AdaptiveDigitalTwin


# ============================================================
# TEST HDD
# ============================================================

TEST_SERIAL = "Z302SZP7"

V5_STATE_FILE = (
    r".\results\digital_twin_state_v5.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ADAPTIVE DIGITAL TWIN - SINGLE HDD TEST")
    print("=" * 80)

    print(
        f"\nTesting HDD: {TEST_SERIAL}"
    )

    # --------------------------------------------------------
    # Load selected HDD from V5 state
    # --------------------------------------------------------

    print(
        "\nLoading V5 Digital Twin state..."
    )

    chunks = []

    for chunk in pd.read_csv(
        V5_STATE_FILE,
        parse_dates=["date"],
        chunksize=100000
    ):

        matched = chunk[
            chunk["serial_number"].astype(str)
            == TEST_SERIAL
        ]

        if not matched.empty:

            chunks.append(matched)

    if not chunks:

        raise ValueError(
            f"HDD {TEST_SERIAL} was not found."
        )

    hdd = pd.concat(
        chunks,
        ignore_index=True
    )

    hdd = (
        hdd
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Observations found: {len(hdd)}"
    )

    print(
        f"First date: {hdd['date'].min().date()}"
    )

    print(
        f"Last date: {hdd['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Remove failure-day observation
    # --------------------------------------------------------

    pre_failure_hdd = hdd[
        hdd["failure"] == 0
    ].copy()

    pre_failure_hdd = (
        pre_failure_hdd
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Pre-failure observations: "
        f"{len(pre_failure_hdd)}"
    )

    if len(pre_failure_hdd) < 30:

        raise ValueError(
            "Not enough pre-failure observations "
            "for the 30-step BiLSTM."
        )

    # --------------------------------------------------------
    # Create Adaptive Digital Twin
    # --------------------------------------------------------

    print(
        "\nLoading Adaptive Digital Twin..."
    )

    twin = AdaptiveDigitalTwin()

    # --------------------------------------------------------
    # Calculate RUL
    # --------------------------------------------------------

    print(
        "\nCalculating RUL..."
    )

    rul = twin.predict_rul(
        pre_failure_hdd
    )

    # --------------------------------------------------------
    # Calculate adaptive degradation
    # --------------------------------------------------------

    print(
        "Calculating adaptive degradation state..."
    )

    degradation = (
        twin.calculate_degradation_state(
            pre_failure_hdd
        )
    )

    # --------------------------------------------------------
    # Latest pre-failure observation
    # --------------------------------------------------------

    latest = pre_failure_hdd.iloc[-1]

    # --------------------------------------------------------
    # Health feature availability
    # --------------------------------------------------------

    availability_columns = [
        "smart_1_available",
        "smart_5_available",
        "smart_187_available",
        "smart_197_available",
        "smart_198_available"
    ]

    available_count = int(
        latest[
            availability_columns
        ].sum()
    )

    # --------------------------------------------------------
    # Actual failure date
    # --------------------------------------------------------

    failure_dates = hdd[
        hdd["failure"] == 1
    ]["date"]

    if not failure_dates.empty:

        failure_date = failure_dates.iloc[0]

        actual_rul = (
            failure_date - latest["date"]
        ).days

    else:

        failure_date = None
        actual_rul = None

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "ADAPTIVE DIGITAL TWIN RESULT"
    )

    print(
        "=" * 80
    )

    print(
        f"HDD ID:                         "
        f"{TEST_SERIAL}"
    )

    print(
        f"Latest pre-failure observation: "
        f"{latest['date'].date()}"
    )

    print(
        f"Failure date:                   "
        f"{failure_date.date() if failure_date is not None else 'Unknown'}"
    )

    print(
        f"Pre-failure observations:       "
        f"{len(pre_failure_hdd)}"
    )

    print(
        f"Health features available:     "
        f"{available_count}/5"
    )

    print(
        f"Current degradation persistence: "
        f"{degradation['current_persistence']}"
    )

    print(
        f"Longest degradation persistence: "
        f"{degradation['longest_persistence']}"
    )

    print(
        f"Adaptive state:                 "
        f"{degradation['adaptive_state']}"
    )

    print(
        f"Predicted RUL:                  "
        f"{rul:.2f} days"
    )

    if actual_rul is not None:

        print(
            f"Actual RUL:                     "
            f"{actual_rul} day(s)"
        )

        print(
            f"Absolute error:                 "
            f"{abs(rul - actual_rul):.2f} days"
        )

    print(
        "=" * 80
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()