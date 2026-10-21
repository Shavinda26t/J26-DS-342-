import numpy as np
import pandas as pd

from adaptive_digital_twin import AdaptiveDigitalTwin

RESULTS_FILE = (
    r".\results\digital_twin_rul"
    r"\heldout_test_adaptive_digital_twin_results.csv"
)

STATE_FILE = r".\results\digital_twin_state_v5.csv"

SERIALS = ["1060A01QF97G", "10C0A0CCF97G"]

results = pd.read_csv(RESULTS_FILE)
states = pd.read_csv(STATE_FILE, parse_dates=["date"])

twin = AdaptiveDigitalTwin()

for serial in SERIALS:
    result = results.loc[
        results["serial_number"].astype(str).str.strip() == serial
    ]

    sequence = states.loc[
        states["serial_number"].astype(str).str.strip() == serial
    ].copy()

    sequence = sequence.loc[sequence["failure"] == 0]
    sequence = sequence.sort_values("date").tail(30)

    print("\n" + "=" * 65)
    print("HDD:", serial)
    print("Validation prediction:",
          result["predicted_rul_days"].iloc[0])
    print("Input observations:", len(sequence))

    if len(sequence) < 30:
        print("ERROR: Fewer than 30 observations")
        continue

    processed = twin.preprocess(sequence)

    print("Processed input shape:", processed.shape)
    print("All values finite:", np.isfinite(processed).all())
    print("Input mean:", round(float(processed.mean()), 6))
    print("Input standard deviation:",
          round(float(processed.std()), 6))
    print("First row, first 10 features:",
          np.round(processed[0, :10], 4).tolist())

    direct_prediction = twin.predict_rul(sequence)
    print("Direct prediction:", direct_prediction)

print("\nDiagnostic complete. No model or source files were modified.")
