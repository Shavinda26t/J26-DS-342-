import numpy as np

from digital_twin_state_manager import DigitalTwinStateManager


twin = DigitalTwinStateManager(
    serial_number="TEST-HDD-001",
    history_size=30
)


print("Adaptive Digital Twin Test")
print("=" * 60)


states = {}

for day in range(1, 31):

    observation = {
        "date": f"2023-01-{day:02d}",
        "serial_number": "TEST-HDD-001",

        # Deliberately degrading SMART values.
        "smart_1_normalized": 100 - day,
        "smart_5_normalized": 100,
        "smart_187_normalized": 100 - (day * 2),
        "smart_197_normalized": 100,
        "smart_198_normalized": 100,
    }

    twin.add_observation(observation)

    state = twin.get_current_state()

    states[day] = state

    print(
        f"Day {day:02d} | "
        f"Severity: {state['combined_severity']} | "
        f"Streak: {state['moderate_plus_streak']} | "
        f"State: {state['adaptive_state']} | "
        f"{state['adaptive_state_name']}"
    )


# ============================================================
# Assertions
#
# The state manager must use the same tuned production
# configuration as scripts/adaptive_digital_twin.py and
# scripts/validate_heldout_adaptive_lead_time.py:
# MIN_SEVERITY=1, PERSISTENCE=7 and the original
# SMART1 / SMART187 threshold tables.
# ============================================================

print()
print("Assertions")
print("=" * 60)


# ------------------------------------------------------------
# 1. Detector configuration matches the tuned production config
# ------------------------------------------------------------

assert DigitalTwinStateManager.MIN_SEVERITY == 1
assert DigitalTwinStateManager.PERSISTENCE == 7

assert DigitalTwinStateManager.SMART1_THRESHOLDS == [
    (-15, 4),
    (-8, 3),
    (-3, 2),
    (-1, 1),
]

assert DigitalTwinStateManager.SMART187_THRESHOLDS == [
    (-23, 4),
    (-9, 3),
    (-3, 2),
    (-1, 1),
]

print("Configuration matches the production detector.")


# ------------------------------------------------------------
# 2. Severity tables
# ------------------------------------------------------------

smart1 = DigitalTwinStateManager.calculate_smart1_severity
smart187 = DigitalTwinStateManager.calculate_smart187_severity

# No degradation
assert smart1(0) == 0
assert smart1(5) == 0
assert smart187(0) == 0

# Mild (severity 1): counts as evidence under MIN_SEVERITY=1
assert smart1(-1) == 1
assert smart1(-2) == 1
assert smart187(-1) == 1
assert smart187(-2) == 1

# Moderate (severity 2)
assert smart1(-3) == 2
assert smart1(-7) == 2
assert smart187(-3) == 2
assert smart187(-8) == 2

# Strong (severity 3)
assert smart1(-8) == 3
assert smart1(-14) == 3
assert smart187(-9) == 3
assert smart187(-22) == 3

# Severe (severity 4)
assert smart1(-15) == 4
assert smart1(-100) == 4
assert smart187(-23) == 4
assert smart187(-100) == 4

# Missing evidence stays NaN (never mistaken for "healthy")
assert np.isnan(smart1(np.nan))
assert np.isnan(smart187(np.nan))

print("SMART 1 / SMART 187 severity tables are correct.")


# ------------------------------------------------------------
# 3. Persistence escalation (PERSISTENCE=7)
#
# The synthetic scenario samples one observation per calendar
# day and degrades SMART 1 by -7 and SMART 187 by -14 every 7
# observations, so combined severity is 3 (Strong) from day 8
# onward. Days 1-7 have no diff(7) change evidence yet.
# ------------------------------------------------------------

# No evidence yet: the streak stays empty (rows without evidence
# reset the run and are flagged evidence_available=False), so
# nothing escalates.
assert states[1]["degradation_evidence_available"] is False
assert states[7]["moderate_plus_streak"] == 0
assert states[7]["adaptive_state"] == 0

# Streak builds but stays below the Early Warning threshold
# (max(1, PERSISTENCE - 2) = 5).
assert states[8]["moderate_plus_streak"] == 1
assert states[8]["adaptive_state"] == 0
assert states[11]["moderate_plus_streak"] == 4
assert states[11]["adaptive_state"] == 0

# Early Warning at streak >= 5
assert states[12]["moderate_plus_streak"] == 5
assert states[12]["adaptive_state"] == 1
assert states[12]["adaptive_state_name"] == "Early Warning"

# Elevated Degradation at streak >= PERSISTENCE (7)
assert states[14]["moderate_plus_streak"] == 7
assert states[14]["adaptive_state"] == 2
assert states[14]["adaptive_state_name"] == "Elevated Degradation"

# Strong Persistent Degradation at streak >= PERSISTENCE + 2 (9)
assert states[16]["moderate_plus_streak"] == 9
assert states[16]["adaptive_state"] == 3
assert (
    states[16]["adaptive_state_name"]
    == "Strong Persistent Degradation"
)

assert states[30]["adaptive_state"] == 3

print("Persistence escalation matches the production detector.")


# ------------------------------------------------------------
# 4. Reading the state must never mutate it
#
# get_current_state() used to increment the streak on every
# call, so polling a twin escalated it from 0 to 3 without a
# single new observation.
# ------------------------------------------------------------

polled_streak = states[30]["moderate_plus_streak"]

first_read = twin.get_current_state()
second_read = twin.get_current_state()
third_read = twin.get_current_state()

for read in (first_read, second_read, third_read):
    assert read["moderate_plus_streak"] == polled_streak
    assert read["adaptive_state"] == 3
    assert read["adaptive_state_name"] == (
        "Strong Persistent Degradation"
    )

print("Repeated reads do not mutate the persistence state.")


# ------------------------------------------------------------
# 5. Out-of-order ingestion
#
# Temporal features assume chronological order, so observations
# arriving shuffled must produce the same state as observations
# arriving in date order.
# ------------------------------------------------------------

def build_observation(day):

    return {
        "date": f"2023-01-{day:02d}",
        "serial_number": "TEST-HDD-002",
        "smart_1_normalized": 100 - day,
        "smart_5_normalized": 100,
        "smart_187_normalized": 100 - (day * 2),
        "smart_197_normalized": 100,
        "smart_198_normalized": 100,
    }


in_order = DigitalTwinStateManager(
    serial_number="TEST-HDD-002",
    history_size=30
)

shuffled = DigitalTwinStateManager(
    serial_number="TEST-HDD-002",
    history_size=30
)

for day in range(1, 16):
    in_order.add_observation(build_observation(day))

for day in range(15, 0, -1):
    shuffled.add_observation(build_observation(day))

in_order_state = in_order.get_current_state()
shuffled_state = shuffled.get_current_state()

assert shuffled_state["moderate_plus_streak"] == (
    in_order_state["moderate_plus_streak"]
)
assert shuffled_state["adaptive_state"] == (
    in_order_state["adaptive_state"]
)
assert shuffled_state["last_update"] == "2023-01-15"
assert in_order_state["moderate_plus_streak"] == 8

print("Out-of-order ingestion matches in-order ingestion.")

print()
print("All assertions passed.")
