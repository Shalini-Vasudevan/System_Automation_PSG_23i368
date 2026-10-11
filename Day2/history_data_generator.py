import csv
import random


def generate_sensor_data():
    temperature = round(random.uniform(30, 120), 2)
    pressure = round(random.uniform(10, 80), 2)
    vibration = round(random.uniform(0, 100), 2)
    rpm = round(random.uniform(500, 3000), 2)
    current = round(random.uniform(5, 50), 2)

    if (temperature >= 100 or pressure >= 70 or
            vibration >= 85 or current >= 45):
        state = "CRITICAL"

    elif (temperature >= 85 or pressure >= 60 or
          vibration >= 70 or rpm >= 2700 or current >= 35):
        state = "HIGH"

    elif (temperature < 40 or pressure < 20 or
          vibration < 15 or rpm < 800 or current < 10):
        state = "LOW"

    else:
        state = "NORMAL"

    return {
        "temperature": temperature,
        "pressure": pressure,
        "vibration": vibration,
        "rpm": rpm,
        "current": current,
        "state": state
    }


with open("sensor_history.csv", mode="w", newline="") as file:
    headers = [
        "temperature",
        "pressure",
        "vibration",
        "rpm",
        "current",
        "state"
    ]

    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()

    for _ in range(250):
        writer.writerow(generate_sensor_data())

