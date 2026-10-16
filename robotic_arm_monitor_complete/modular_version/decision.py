"""Requirement 5 - one final decision per arm.

Pipeline order (this order matters):
  1. Hard safety check      -> EMERGENCY_STOP immediately, model never touched
  2. Spoof check per joint  -> torque untrusted => model runs in degraded mode
  3. ML wear probability    -> LOW / MED / HIGH
  4. Maintenance notes      -> NONE / MINOR / SEVERE
  5. Combine (table below)

  model \\ notes |  NONE            MINOR            SEVERE
  --------------+------------------------------------------------
  LOW           |  SAFE            MONITOR          SCHEDULE_MAINT.
  MED           |  MONITOR         SCHEDULE_MAINT.  SCHEDULE_MAINT.
  HIGH          |  SCHEDULE_MAINT. SCHEDULE_MAINT.  EMERGENCY_STOP

Overrides:
  * spoof suspected + any other signal agrees something is wrong -> EMERGENCY_STOP
    (someone is hiding something AND evidence of a problem exists)
  * spoof suspected, nothing else wrong -> at least SCHEDULE_MAINTENANCE
    (inspect the sensor / possible intrusion)
  * model unavailable -> a SAFE result is downgraded to MONITOR
    (can't claim 'safe' without evidence)
"""
from dataclasses import dataclass, field
from config import (WEAR_MED, WEAR_HIGH, RATED_TORQUE_NM, DECISIONS)
from safety import check_safety
import numpy as np

RANK = {d: i for i, d in enumerate(DECISIONS)}
SAFE, MONITOR, SCHEDULE, STOP = DECISIONS

TABLE = {
    ("LOW", "NONE"): SAFE,      ("LOW", "MINOR"): MONITOR,      ("LOW", "SEVERE"): SCHEDULE,
    ("MED", "NONE"): MONITOR,   ("MED", "MINOR"): SCHEDULE,     ("MED", "SEVERE"): SCHEDULE,
    ("HIGH", "NONE"): SCHEDULE, ("HIGH", "MINOR"): SCHEDULE,    ("HIGH", "SEVERE"): STOP,
}


@dataclass
class JointWindow:
    torque_nm: list
    temp_c: list
    vibration_mm_s: list
    hours_since_service: float = 0.0


@dataclass
class ArmReport:
    arm_id: str
    decision: str = SAFE
    reasons: list = field(default_factory=list)
    safety_trips: list = field(default_factory=list)
    spoof_flags: dict = field(default_factory=dict)
    wear_prob: float = None
    model_level: str = "UNAVAILABLE"
    note_level: str = "NONE"


def model_level(p):
    return "LOW" if p < WEAR_MED else "MED" if p < WEAR_HIGH else "HIGH"


def assess_arm(arm_id, windows, model, spoof_detector, notes):
    rep = ArmReport(arm_id)

    # 1. HARD SAFETY - first, and returns before the model is ever called.
    trips = check_safety(windows)
    if trips:
        rep.decision, rep.safety_trips = STOP, trips
        rep.reasons.append("HARD SAFETY LIMIT crossed: " + trips[0] + (f" (+{len(trips)-1} more)" if len(trips) > 1 else ""))
        return rep

    # 2. spoof check per joint
    for joint, w in windows.items():
        flags = spoof_detector.check(joint, w) if spoof_detector else []
        if flags:
            rep.spoof_flags[joint] = flags

    # 3. ML wear probability (max over joints); torque ignored on spoofed joints
    probs = []
    try:
        for joint, w in windows.items():
            torque_pct = float(np.mean(w.torque_nm)) / RATED_TORQUE_NM[joint] * 100.0
            p = model.predict(torque_pct, float(np.mean(w.temp_c)), float(np.mean(w.vibration_mm_s)),
                              w.hours_since_service, torque_trusted=joint not in rep.spoof_flags)
            probs.append(p)
        rep.wear_prob = max(probs)
        rep.model_level = model_level(rep.wear_prob)
    except Exception as exc:                      # model missing / broken
        rep.reasons.append(f"model unavailable ({type(exc).__name__}) - running on notes only")

    # 4. maintenance notes
    nr = notes.get(arm_id)
    rep.note_level = nr.level if nr else "NONE"
    if nr:
        rep.reasons.append(f"notes {nr.level}: " + ", ".join(nr.evidence))

    # 5. combine
    lvl = rep.model_level if rep.model_level != "UNAVAILABLE" else "LOW"
    d = TABLE[(lvl, rep.note_level)]
    if rep.model_level == "UNAVAILABLE" and d == SAFE:
        d = MONITOR
    if rep.wear_prob is not None:
        rep.reasons.append(f"model {rep.model_level} (p={rep.wear_prob:.2f})"
                           + (" [degraded: torque ignored]" if rep.spoof_flags else ""))

    if rep.spoof_flags:
        rep.reasons.append("SPOOF suspected on " + ", ".join(rep.spoof_flags))
        corroborated = rep.model_level in ("MED", "HIGH") or rep.note_level != "NONE"
        d = STOP if corroborated else max(d, SCHEDULE, key=RANK.get)
        rep.reasons.append("other signals corroborate a real problem -> STOP" if corroborated
                           else "no corroboration -> inspect sensor")
    rep.decision = d
    return rep
