"""Creates the synthetic historical dataset (requirement 2).

A hidden 'wear' variable (0 = brand new, 1 = about to fail) drives all
sensors, mimicking physics:
  * worn joints have more friction  -> more torque for the same load
  * more friction                   -> more heat and more vibration
  * worn joints are usually long past their last service
A joint is labelled failed_within_30d = 1 when wear (plus a little label
noise, because real failures are never perfectly predictable) exceeds 0.6.
"""
import numpy as np
import pandas as pd
from config import JOINTS, RATED_TORQUE_NM


def sample_reading(rng, joint, load_pct, wear):
    """One physical sensor reading for a joint at a given load and wear level."""
    torque_pct = load_pct + 25.0 * wear + rng.normal(0, 3.0)
    temp_c = 35.0 + 0.30 * torque_pct + 20.0 * wear + rng.normal(0, 2.5)
    vib = 0.8 + 0.02 * torque_pct + 5.5 * wear + rng.normal(0, 0.35)
    return {
        "torque_nm": torque_pct / 100.0 * RATED_TORQUE_NM[joint],
        "temp_c": temp_c,
        "vibration_mm_s": max(vib, 0.05),
    }


def hours_since_service(rng, wear):
    return max(0.0, rng.uniform(50, 500) + 1500.0 * wear + rng.normal(0, 150))


def generate(n=150, seed=42, path="historical_joint_data.csv"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        joint = JOINTS[i % 3]
        wear = rng.uniform(0, 1)
        load = rng.uniform(35, 75)
        r = sample_reading(rng, joint, load, wear)
        failed = int(wear + rng.normal(0, 0.07) > 0.60)
        rows.append({
            "reading_id": i + 1,
            "arm_id": f"ARM-{(i % 6) + 1:02d}",
            "joint": joint,
            "torque_nm": round(r["torque_nm"], 2),
            "temp_c": round(r["temp_c"], 1),
            "vibration_mm_s": round(r["vibration_mm_s"], 2),
            "hours_since_service": round(hours_since_service(rng, wear)),
            "failed_within_30d": failed,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = generate()
    print(f"Wrote historical_joint_data.csv  ({len(d)} rows, "
          f"{d.failed_within_30d.sum()} failures / "
          f"{(1 - d.failed_within_30d).sum()} healthy)")
