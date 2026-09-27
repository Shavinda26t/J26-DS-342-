from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path(
    r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"
)


# --------------------------------------------------
# Find daily CSV files
# --------------------------------------------------

files = sorted(DATA_DIR.glob("*.csv"))

print(f"Files found: {len(files)}")


# --------------------------------------------------
# Store first-seen and failure dates
# --------------------------------------------------

first_seen = {}
failure_dates = {}


# --------------------------------------------------
# Process each daily file
# --------------------------------------------------

for i, file in enumerate(files, start=1):

    print(f"Processing {i}/{len(files)}: {file.name}")

    df = pd.read_csv(
        file,
        usecols=["date", "serial_number", "failure"]
    )

    # Make sure date is a proper datetime value
    df["date"] = pd.to_datetime(df["date"])

    # First date each HDD is observed
    first_dates = (
        df.groupby("serial_number")["date"]
        .min()
    )

    for serial, date in first_dates.items():
        if serial not in first_seen:
            first_seen[serial] = date

    # HDDs that failed on this date
    failed = df[df["failure"] == 1]

    for serial, date in zip(
        failed["serial_number"],
        failed["date"]
    ):
        failure_dates[serial] = date


# --------------------------------------------------
# Calculate pre-failure history
# --------------------------------------------------

history_days = []

for serial, failure_date in failure_dates.items():

    if serial in first_seen:

        first_date = first_seen[serial]

        days = (
            failure_date - first_date
        ).days + 1

        history_days.append(days)


# --------------------------------------------------
# Convert to Series
# --------------------------------------------------

history = pd.Series(history_days)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print()
print("========================================")
print("Q1 2023 FAILURE HISTORY ANALYSIS")
print("========================================")

print(f"Daily CSV files: {len(files)}")
print(f"Unique failed HDDs: {len(failure_dates)}")
print(f"Failure records: {len(failure_dates)}")

if len(history) > 0:

    print()
    print("Pre-failure observation history")
    print("----------------------------------------")

    print(f"Minimum history: {history.min()} days")
    print(f"Maximum history: {history.max()} days")
    print(f"Median history: {history.median():.1f} days")
    print(f"Mean history: {history.mean():.1f} days")

    print()
    print("History thresholds")
    print("----------------------------------------")

    for threshold in [7, 14, 30, 60, 90]:

        count = int(
            (history >= threshold).sum()
        )

        percentage = (
            count / len(history) * 100
        )

        print(
            f">= {threshold:2d} days: "
            f"{count:4d} drives "
            f"({percentage:.1f}%)"
        )

else:

    print("No failed HDD history found.")