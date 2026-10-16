"""Requirement 1 - immediate safety stop.

DELIBERATELY standalone: imports nothing but `math` and the config. No model,
no sklearn, no pandas, no log parsing. If everything else in the system is
broken, this still works. Rules are plain `if` comparisons.
"""
import math
from config import SAFETY_LIMITS, RATED_TORQUE_NM


def _bad(x):
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def check_safety(arm_windows):
    """Return a list of human-readable trip reasons ([] means no trip).

    Every single sample of every joint is checked (not just an average):
    one dangerous reading is enough.  Missing / NaN readings are treated as a
    trip too (fail-safe: a blind arm is not a safe arm).
    """
    reasons = []
    for joint, w in arm_windows.items():
        rated = RATED_TORQUE_NM[joint]
        torque_limit_nm = rated * SAFETY_LIMITS["torque_pct"] / 100.0
        for i, (tq, tp, vb) in enumerate(zip(w.torque_nm, w.temp_c, w.vibration_mm_s)):
            if _bad(tq) or _bad(tp) or _bad(vb):
                reasons.append(f"{joint}[{i}]: missing/invalid reading (fail-safe)")
                continue
            if tq > torque_limit_nm:
                reasons.append(f"{joint}[{i}]: torque {tq:.1f} N·m > limit {torque_limit_nm:.1f} N·m")
            if tp > SAFETY_LIMITS["temp_c"]:
                reasons.append(f"{joint}[{i}]: temperature {tp:.1f} °C > limit {SAFETY_LIMITS['temp_c']} °C")
            if vb > SAFETY_LIMITS["vibration_mm_s"]:
                reasons.append(f"{joint}[{i}]: vibration {vb:.1f} mm/s > limit {SAFETY_LIMITS['vibration_mm_s']} mm/s")
    return reasons
