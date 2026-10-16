"""Runs the whole pipeline end-to-end.   Usage:  python demo.py"""
import os
from datetime import datetime
import numpy as np

from config import RATED_TORQUE_NM, DECISIONS
from generate_data import generate, sample_reading
from model import WearModel, load_history
from spoof import SpoofDetector
from notes import score_log
from decision import assess_arm, JointWindow

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "historical_joint_data.csv")
LOG = os.path.join(HERE, "maintenance_log.txt")
NOW = datetime(2026, 10, 14, 12, 0)
N = 20  # samples per window
rng = np.random.default_rng(7)


def window(joint, load, wear, hours=None, torque_override=None):
    """Simulate N samples from a joint; optionally replace the torque channel."""
    rows = [sample_reading(rng, joint, load, wear) for _ in range(N)]
    tq = [r["torque_nm"] for r in rows] if torque_override is None else list(torque_override)
    return JointWindow(tq, [r["temp_c"] for r in rows], [r["vibration_mm_s"] for r in rows],
                       hours if hours is not None else 150 + 1500 * wear)


def healthy_arm():
    return {"shoulder": window("shoulder", 55, 0.05), "elbow": window("elbow", 50, 0.05),
            "wrist": window("wrist", 45, 0.05)}


def pct_to_nm(joint, pct_list):
    return [p / 100 * RATED_TORQUE_NM[joint] for p in pct_list]


def fake_torque(joint, pct, noise=3.0):
    """Attacker-supplied torque that *looks* noisy and realistic."""
    return pct_to_nm(joint, rng.normal(pct, noise, N))


class BrokenModel:
    def predict(self, *a, **k):
        raise RuntimeError("model server down")


def scenarios():
    s = {}
    s["ARM-01"] = ("Healthy arm, clean notes", healthy_arm())
    s["ARM-02"] = ("Sensors fine, technician hears a squeak", healthy_arm())
    a = healthy_arm(); a["elbow"] = window("elbow", 65, 0.85)
    s["ARM-03"] = ("Elbow clearly worn, no notes", a)
    a = healthy_arm(); a["shoulder"].torque_nm[11] = 175.0   # 146 % of rated, single sample
    s["ARM-04"] = ("One shoulder torque spike over hard limit", a)
    a = healthy_arm(); a["elbow"] = window("elbow", 65, 0.85)
    s["ARM-05"] = ("Elbow worn + 'burning smell / grinding' notes", a)
    a = healthy_arm(); a["elbow"] = window("elbow", 65, 0.85)
    a["elbow"].torque_nm = fake_torque("elbow", 35)           # attacker hides ~90 % torque
    s["ARM-06"] = ("SPOOFED elbow torque (hides worn joint)", a)
    a = healthy_arm(); a["wrist"].torque_nm = [14.0] * N      # frozen value, everything else healthy
    s["ARM-07"] = ("Frozen wrist torque, nothing else wrong", a)
    s["ARM-08"] = ("Healthy arm but model service is down", healthy_arm())
    return s


def main():
    if not os.path.exists(CSV):
        generate(path=CSV)
    hist = load_history(CSV)
    model = WearModel(hist)
    spoof = SpoofDetector(hist)
    notes = score_log(LOG, NOW)

    print(f"Dataset: {len(hist)} labelled readings | model 5-fold CV accuracy "
          f"{model.cv_full:.0%} (no-torque degraded model {model.cv_no_torque:.0%}) | "
          f"spoof-check false-alarm rate on history {spoof.history_false_alarm_rate:.1%}\n")

    reports = {}
    for arm, (title, w) in scenarios().items():
        m = BrokenModel() if arm == "ARM-08" else model
        reports[arm] = assess_arm(arm, w, m, spoof, notes)
        r = reports[arm]
        print(f"{arm}  {r.decision:<21} <- {title}")
        for line in r.reasons:
            print(f"          - {line}")
        for j, fl in r.spoof_flags.items():
            for f in fl:
                print(f"          - [{j}] {f}")
        print()

    # ---- ARM-04: safety stop must work with NO model at all ------------------
    r = assess_arm("ARM-04", scenarios()["ARM-04"][1], None, None, {})
    assert r.decision == "EMERGENCY_STOP"
    print("Safety independence check: ARM-04 still -> EMERGENCY_STOP with model=None "
          "and spoof detector=None.\n")

    # ---- all four outcomes reachable? ----------------------------------------
    got = {r.decision for r in reports.values()}
    print("Outcomes demonstrated:", ", ".join(sorted(got, key=DECISIONS.index)))
    assert got == set(DECISIONS), "not every decision was reached!"

    spoof_attack_table(spoof)


def spoof_attack_table(spoof):
    """Which attacks does the spoof defence catch, and which does it miss?"""
    print("\nSpoofed-torque attack test (elbow joint, true state = worn, ~90 % torque):")
    true = window("elbow", 65, 0.85)
    cases = {}
    cases["Frozen value (flatline at 35 %)"] = JointWindow(pct_to_nm("elbow", [35.0] * N), true.temp_c, true.vibration_mm_s)
    rec = pct_to_nm("elbow", rng.normal(35, 3, N // 2))
    cases["Replay of a recorded 10-sample window"] = JointWindow(rec + rec, true.temp_c, true.vibration_mm_s)
    stepped = list(true.torque_nm[:10]) + pct_to_nm("elbow", rng.normal(35, 3, 10))
    cases["Sudden drop mid-window (hide a fault)"] = JointWindow(stepped, true.temp_c, true.vibration_mm_s)
    cases["Noisy fake value, far from truth (35 %)"] = JointWindow(fake_torque("elbow", 35), true.temp_c, true.vibration_mm_s)
    cases["Noisy fake, only modestly low (true -10 %)"] = JointWindow(
        pct_to_nm("elbow", np.array(true.torque_nm) / 80 * 100 - 10), true.temp_c, true.vibration_mm_s)
    healthy = window("elbow", 50, 0.05)
    cases["Coherent fake: torque+temp+vibration all forged"] = JointWindow(
        fake_torque("elbow", 50), healthy.temp_c, healthy.vibration_mm_s)
    print(f"  {'attack':<52}{'caught?'}")
    for name, w in cases.items():
        caught = bool(spoof.check("elbow", w))
        print(f"  {name:<52}{'YES' if caught else 'NO  (known limitation)'}")


if __name__ == "__main__":
    main()
