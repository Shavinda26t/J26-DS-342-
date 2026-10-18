
"""
Step 157 - Paired Bootstrap Confidence Intervals

Analyze Step 156's held-out evaluation.

Compare filtered DQN with:
1. NO_OP
2. SAFE_MIGRATION_FIRST
3. RAW_DQN

Uses paired episode returns and bootstrap resampling.

Synthetic simulation only.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results"
    / "156_dqn_v2_safety_filter_holdout.json"
)

OUTPUT_FILE = (
    ROOT / "results"
    / "157_v2_holdout_confidence_intervals.json"
)

BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 15742

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

if OUTPUT_FILE.exists():
    raise FileExistsError(
        "Step 157 report already exists. "
        "Do not overwrite existing research evidence."
    )

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
) as file:
    source = json.load(file)

returns = {
    item["policy"]: np.asarray(
        item["episode_returns"],
        dtype=float,
    )
    for item in source["results"]
}

required_policies = (
    "FILTERED_DQN",
    "NO_OP",
    "SAFE_MIGRATION_FIRST",
    "RAW_DQN",
)

for policy in required_policies:
    if policy not in returns:
        raise ValueError(
            f"Missing policy: {policy}"
        )

n = len(returns["FILTERED_DQN"])

if n != 200:
    raise ValueError(
        f"Expected 200 episodes, received {n}"
    )

for policy in required_policies:
    if len(returns[policy]) != n:
        raise ValueError(
            "Policies have different episode counts"
        )

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

comparisons = {}

print("=" * 62)
print("STEP 157 - HELD-OUT PAIRED CONFIDENCE INTERVALS")
print("=" * 62)

print("Paired episodes:", n)
print("Bootstrap repetitions:", BOOTSTRAP_REPETITIONS)

for baseline in (
    "NO_OP",
    "SAFE_MIGRATION_FIRST",
    "RAW_DQN",
):

    # Positive difference means filtered DQN
    # achieved a higher episode reward.
    differences = (
        returns["FILTERED_DQN"]
        - returns[baseline]
    )

    observed_mean = float(
        np.mean(differences)
    )

    bootstrap_indices = rng.integers(
        low=0,
        high=n,
        size=(BOOTSTRAP_REPETITIONS, n),
    )

    bootstrap_means = differences[
        bootstrap_indices
    ].mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    comparisons[baseline] = {
        "mean_paired_difference": round(
            observed_mean, 5
        ),
        "bootstrap_95pct_interval": [
            round(float(lower), 5),
            round(float(upper), 5),
        ],
        "positive_difference_episodes": int(
            np.sum(differences > 0)
        ),
        "negative_difference_episodes": int(
            np.sum(differences < 0)
        ),
        "equal_difference_episodes": int(
            np.sum(differences == 0)
        ),
    }

    print("\nFiltered DQN minus:", baseline)

    print(
        "Mean paired difference:",
        round(observed_mean, 5)
    )

    print(
        "Bootstrap 95% interval:",
        [
            round(float(lower), 5),
            round(float(upper), 5),
        ]
    )

    print(
        "Episodes favouring filtered DQN:",
        comparisons[baseline][
            "positive_difference_episodes"
        ],
    )

report = {
    "experiment": "C4_V2_HELDOUT_PAIRED_BOOTSTRAP_157",
    "source_report": INPUT_FILE.name,
    "simulation_only": True,
    "paired_episodes": n,
    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "comparisons": comparisons,
    "limitations": [
        "Synthetic, uncalibrated simulator",
        "Single trained DQN checkpoint",
        "Bootstrap intervals quantify seed variation only",
        "No uncertainty over training seeds",
        "Not a real-world HDD effectiveness estimate",
    ],
}

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(report, file, indent=4)

print("\nReport saved:", OUTPUT_FILE)
print("STEP 157 COMPLETE")
