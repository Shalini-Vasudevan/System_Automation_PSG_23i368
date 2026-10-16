"""Central configuration: every threshold and weight lives here so it can be
justified (and changed) in one place."""

JOINTS = ("shoulder", "elbow", "wrist")

# Rated (continuous) torque of each joint's actuator, in N·m.
RATED_TORQUE_NM = {"shoulder": 120.0, "elbow": 80.0, "wrist": 30.0}

# ---- Requirement 1: HARD safety limits (deterministic, no ML) --------------
SAFETY_LIMITS = {
    "torque_pct": 130.0,      # % of rated torque  (drive peak rating ~ 130 %)
    "temp_c": 95.0,           # joint housing temperature, deg C
    "vibration_mm_s": 11.0,   # RMS velocity; ISO 10816 "damage occurs" zone
}

# ---- Requirement 2: model risk levels --------------------------------------
WEAR_MED = 0.30    # wear probability >= this  -> MED
WEAR_HIGH = 0.60   # wear probability >= this  -> HIGH

# ---- Requirement 3: maintenance notes --------------------------------------
NOTE_MAX_AGE_DAYS = 14   # older notes are assumed to have been dealt with
NOTE_SEVERE_POINTS = 3   # points needed for level SEVERE
NOTE_MINOR_POINTS = 1    # points needed for level MINOR

# ---- Requirement 4: spoof-defence tuning -----------------------------------
SPOOF_FLATLINE_STD_PCT = 0.3    # real torque noise is ~3 %; < 0.3 % = frozen
SPOOF_MAX_STEP_PCT = 40.0       # max believable sample-to-sample jump
SPOOF_RESIDUAL_Z = 3.0          # cross-sensor disagreement, in std-devs

DECISIONS = ("SAFE", "MONITOR", "SCHEDULE_MAINTENANCE", "EMERGENCY_STOP")
