import csv
import random


def generate_sensor_data():
    return {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(20, 60), 2),
        "vibration": round(random.uniform(50, 95), 2),
        "rpm": round(random.uniform(50, 95), 2),
        "current": round(random.uniform(50, 95), 2),
        "state": round(random.uniform(50, 95), 2),
    }


with open("sensor_history.csv", mode="w", newline="") as file:
    headers = ["temperature", "pressure", "vibration", "rpm", "current", "state"]
    writer = csv.DictWriter(file, fieldnames=headers)

    writer.writeheader() 

    for _ in range(250):
        writer.writerow(generate_sensor_data())


