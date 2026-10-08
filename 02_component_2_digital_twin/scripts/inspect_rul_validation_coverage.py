import os
import pandas as pd


OUTPUT_DIR = r".\results\digital_twin_rul"

ELIGIBILITY_FILE = os.path.join(
    OUTPUT_DIR,
    "all_eligible_hdd_eligibility.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "all_eligible_hdd_summary.csv"
)


print("=" * 80)
print("RUL VALIDATION COVERAGE INSPECTION")
print("=" * 80)


# ============================================================
# Load eligibility
# ============================================================

eligibility = pd.read_csv(
    ELIGIBILITY_FILE
)

eligibility["failure_date"] = pd.to_datetime(
    eligibility["failure_date"]
)


eligible = eligibility[
    eligibility["eligible_for_30_step"] == True
].copy()


print(
    "\nEligible HDDs:",
    len(eligible)
)


# ============================================================
# Load evaluated HDDs
# ============================================================

summary = pd.read_csv(
    SUMMARY_FILE
)

evaluated_serials = set(
    summary["serial_number"]
    .astype(str)
)


eligible["serial_number"] = (
    eligible["serial_number"]
    .astype(str)
)


eligible["evaluated"] = (
    eligible["serial_number"]
    .isin(evaluated_serials)
)


not_evaluated = eligible[
    ~eligible["evaluated"]
].copy()


print(
    "Evaluated HDDs:",
    eligible["evaluated"].sum()
)

print(
    "Not evaluated:",
    len(not_evaluated)
)


# ============================================================
# Display missing HDDs
# ============================================================

if not not_evaluated.empty:

    print(
        "\n" + "-" * 80
    )

    print(
        "ELIGIBLE BUT NOT EVALUATED HDDs"
    )

    print(
        "-" * 80
    )

    print(
        not_evaluated[
            [
                "serial_number",
                "failure_date",
                "pre_failure_observations",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# Distribution of pre-failure history
# ============================================================

print(
    "\n" + "-" * 80
)

print(
    "PRE-FAILURE HISTORY DISTRIBUTION"
)

print(
    "-" * 80
)

print(
    eligible[
        "pre_failure_observations"
    ].describe().to_string()
)


# ============================================================
# Save inspection
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "rul_validation_coverage_inspection.csv"
)

not_evaluated.to_csv(
    output_file,
    index=False
)


print(
    "\nSaved:"
)

print(
    output_file
)

print("=" * 80)