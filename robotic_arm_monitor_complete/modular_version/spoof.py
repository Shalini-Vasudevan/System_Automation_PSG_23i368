"""Requirement 4 - spoofed joint-torque defence.

Threat: an attacker feeds a fake (lower) torque reading to hide a real problem.
Four independent checks, all on the torque channel of a window of samples:

 1. FLATLINE      - real torque is noisy; a frozen value has ~0 variance.
 2. REPLAY        - second half of the window is an exact copy of the first.
 3. IMPLAUSIBLE STEP - a sample-to-sample jump no physical arm could make
                    (e.g. torque suddenly 'drops' to hide a fault).
 4. CROSS-SENSOR  - physics ties torque to heat and vibration. Fit on history:
                    expected_torque = f(temperature, vibration).  If the reported
                    torque is > Z std-devs from what the *other* sensors imply,
                    the torque channel is lying (the attacker can't easily fake
                    the heat of the motor).
"""
import numpy as np
from config import (SPOOF_FLATLINE_STD_PCT, SPOOF_MAX_STEP_PCT, SPOOF_RESIDUAL_Z,
                    RATED_TORQUE_NM)


class SpoofDetector:
    def __init__(self, history_df):
        X = np.column_stack([np.ones(len(history_df)),
                             history_df["temp_c"], history_df["vibration_mm_s"]])
        y = history_df["torque_pct"].to_numpy()
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.sigma = float(np.std(y - X @ self.coef))
        z = np.abs((y - X @ self.coef) / self.sigma)
        self.history_false_alarm_rate = float((z > SPOOF_RESIDUAL_Z).mean())

    def expected_torque_pct(self, temp_c, vib):
        return float(self.coef @ np.array([1.0, temp_c, vib]))

    def check(self, joint, w):
        """Return a list of spoof flags for this joint's window ([] = looks real)."""
        tq = np.asarray(w.torque_nm, dtype=float) / RATED_TORQUE_NM[joint] * 100.0
        flags = []
        if tq.std() < SPOOF_FLATLINE_STD_PCT:
            flags.append("flatline: torque has no sensor noise (frozen/injected value)")
        n = len(tq) // 2
        if n >= 4 and tq.std() >= SPOOF_FLATLINE_STD_PCT and np.allclose(tq[:n], tq[n:2 * n]):
            flags.append("replay: second half of window is a copy of the first")
        if len(tq) > 1 and np.abs(np.diff(tq)).max() > SPOOF_MAX_STEP_PCT:
            flags.append(f"implausible step: torque jumped {np.abs(np.diff(tq)).max():.0f} % of rated in one sample")
        exp = self.expected_torque_pct(np.mean(w.temp_c), np.mean(w.vibration_mm_s))
        z = (tq.mean() - exp) / self.sigma
        if abs(z) > SPOOF_RESIDUAL_Z:
            flags.append(f"cross-sensor mismatch: torque {tq.mean():.0f} % but temp/vibration "
                         f"imply ~{exp:.0f} % (z={z:+.1f})")
        return flags
