from digital_twin_state_manager import DigitalTwinStateManager


twin = DigitalTwinStateManager(
    serial_number="TEST-HDD-001",
    history_size=30
)


print("Adaptive Digital Twin Test")
print("=" * 60)


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

    print(
        f"Day {day:02d} | "
        f"Severity: {state['combined_severity']} | "
        f"Streak: {state['moderate_plus_streak']} | "
        f"State: {state['adaptive_state']} | "
        f"{state['adaptive_state_name']}"
    )