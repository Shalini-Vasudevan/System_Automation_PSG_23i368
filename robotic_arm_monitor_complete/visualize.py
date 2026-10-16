"""
Visualisation suite for the Robotic Arm Joint Health Monitoring System.

RUN (same folder as robotic_arm_monitor.py):
    pip install matplotlib          (plus numpy pandas scikit-learn)
    python visualize.py             -> saves PNGs into ./visuals/
    python visualize.py --show      -> also opens the windows

Creates 11 charts + one contact sheet (00_dashboard.png).
"""
import os
import sys

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict

import robotic_arm_monitor as ram

SHOW = "--show" in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visuals")
os.makedirs(OUT, exist_ok=True)

OK, BAD, GREY = "#2a9d8f", "#e76f51", "#8d99ae"
DEC_COL = {"SAFE": "#2a9d8f", "MONITOR": "#4c8bd6",
           "SCHEDULE_MAINTENANCE": "#f4a261", "EMERGENCY_STOP": "#d62828"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 10})

# ---------------------------------------------------------------- run pipeline
if not os.path.exists(ram.CSV):
    ram.generate(path=ram.CSV)
hist = ram.load_history(ram.CSV)
model = ram.WearModel(hist)
spoof = ram.SpoofDetector(hist)
notes = ram.score_log(ram.LOG, ram.NOW)
scen = ram.scenarios()
reports = {}
for arm, (title, w) in scen.items():
    m = ram.BrokenModel() if arm == "ARM-08" else model
    reports[arm] = ram.assess_arm(arm, w, m, spoof, notes)
RATED = ram.RATED_TORQUE_NM
LIM = ram.SAFETY_LIMITS
saved = []


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130)
    saved.append(path)
    print("saved", path)
    if not SHOW:
        plt.close(fig)


def pct(joint, nm):
    return np.asarray(nm, dtype=float) / RATED[joint] * 100


# 1 ---------------------------------------------------- dataset scatter
def fig_dataset():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    marker = {"shoulder": "o", "elbow": "s", "wrist": "^"}
    for j, mk in marker.items():
        for lab, col, name in ((0, OK, "healthy"), (1, BAD, "fails within 30 d")):
            d = hist[(hist.joint == j) & (hist.failed_within_30d == lab)]
            ax.scatter(d.temp_c, d.vibration_mm_s, c=col, marker=mk, s=48, alpha=.8,
                       edgecolor="white", linewidth=.5, label=f"{j} - {name}")
    ax.axvline(LIM["temp_c"], color="#d62828", ls="--")
    ax.axhline(LIM["vibration_mm_s"], color="#d62828", ls="--")
    ax.axvspan(LIM["temp_c"], 100, color="#d62828", alpha=.08)
    ax.axhspan(LIM["vibration_mm_s"], 12.5, color="#d62828", alpha=.08)
    ax.text(96, 0.4, "hard-stop\nzone", color="#d62828", fontsize=9)
    ax.set_xlim(30, 100); ax.set_ylim(0, 12.5)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Vibration (mm/s RMS)")
    ax.set_title(f"Synthetic historical dataset ({len(hist)} readings): failing joints sit\n"
                 "well inside the hard safety limits, which is the model's job")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    save(fig, "01_dataset_scatter.png")


# 2 ---------------------------------------------------- feature distributions
def fig_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    cols = [("torque_pct", "Torque (% of rated)", LIM["torque_pct"]),
            ("temp_c", "Temperature (°C)", LIM["temp_c"]),
            ("vibration_mm_s", "Vibration (mm/s)", LIM["vibration_mm_s"]),
            ("hours_since_service", "Hours since service", None)]
    for ax, (c, lab, lim) in zip(axes.flat, cols):
        hi = max(hist[c].max(), lim or 0) * 1.03
        bins = np.linspace(hist[c].min(), hi, 28)
        ax.hist(hist[hist.failed_within_30d == 0][c], bins, alpha=.75, color=OK, label="healthy")
        ax.hist(hist[hist.failed_within_30d == 1][c], bins, alpha=.75, color=BAD, label="fails within 30 d")
        if lim:
            ax.axvline(lim, color="#d62828", ls="--")
            ax.text(lim, ax.get_ylim()[1] * .9, " hard limit", color="#d62828", ha="right", fontsize=9)
        ax.set_xlabel(lab); ax.set_ylabel("readings")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Feature distributions: healthy vs failing joints", fontsize=13)
    save(fig, "02_feature_distributions.png")


# 3 ---------------------------------------------------- model quality
def fig_model():
    y = hist.failed_within_30d.values
    pred = cross_val_predict(ram._pipe(), hist[ram.FULL], y, cv=5)
    proba = cross_val_predict(ram._pipe(), hist[ram.FULL], y, cv=5, method="predict_proba")[:, 1]
    cm = confusion_matrix(y, pred)
    fpr, tpr, _ = roc_curve(y, proba)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axes[0]
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=16,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["healthy", "fails"]); ax.set_yticklabels(["healthy", "fails"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (5-fold CV)")
    ax = axes[1]
    ax.plot(fpr, tpr, color="#4c8bd6", lw=2.5, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    ax = axes[2]
    coef = model.full.named_steps["logisticregression"].coef_[0]
    order = np.argsort(coef)
    ax.barh(np.array(ram.FULL)[order], coef[order],
            color=[BAD if c > 0 else OK for c in coef[order]])
    ax.set_xlabel("standardised coefficient"); ax.set_title("What drives predicted wear")
    fig.suptitle(f"Wear model: CV accuracy {model.cv_full:.0%} "
                 f"(degraded no-torque model {model.cv_no_torque:.0%})", fontsize=13)
    save(fig, "03_model_quality.png")


# 4 ---------------------------------------------------- safety margins
def fig_safety_margin():
    fail = hist[hist.failed_within_30d == 1]
    healthy = hist[hist.failed_within_30d == 0]
    names = ["Torque", "Temperature", "Vibration"]
    cols = ["torque_pct", "temp_c", "vibration_mm_s"]
    keys = ["torque_pct", "temp_c", "vibration_mm_s"]
    h = [healthy[c].max() / LIM[k] * 100 for c, k in zip(cols, keys)]
    f = [fail[c].max() / LIM[k] * 100 for c, k in zip(cols, keys)]
    x = np.arange(3); w = .35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, h, w, color=OK, label="worst healthy joint")
    ax.bar(x + w / 2, f, w, color=BAD, label="worst failing joint")
    ax.axhline(100, color="#d62828", ls="--", lw=2)
    ax.text(2.45, 101.5, "100 % = EMERGENCY_STOP", color="#d62828", ha="right")
    for xi, v in zip(x - w / 2, h): ax.text(xi, v + 1, f"{v:.0f}%", ha="center")
    for xi, v in zip(x + w / 2, f): ax.text(xi, v + 1, f"{v:.0f}%", ha="center")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 115)
    ax.set_ylabel("% of hard safety limit")
    ax.set_title("Safety thresholds sit above everything a failing joint reached")
    ax.legend(loc="lower right")
    save(fig, "04_safety_margins.png")


# 5 ---------------------------------------------------- spoof cross-sensor
def fig_spoof_scatter():
    exp_hist = np.array([spoof.expected_torque_pct(t, v)
                         for t, v in zip(hist.temp_c, hist.vibration_mm_s)])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(exp_hist, hist.torque_pct, s=22, color=GREY, alpha=.6, label="historical readings")
    lo, hi = 20, 115
    xs = np.array([lo, hi])
    ax.plot(xs, xs, "k-", lw=1)
    ax.fill_between(xs, xs - 3 * spoof.sigma, xs + 3 * spoof.sigma, color="#4c8bd6", alpha=.12,
                    label=f"believable zone (±3σ = ±{3 * spoof.sigma:.0f}% )")
    for arm, (_, wins) in scen.items():
        for j, jw in wins.items():
            rep = pct(j, jw.torque_nm).mean()
            exp = spoof.expected_torque_pct(np.mean(jw.temp_c), np.mean(jw.vibration_mm_s))
            flagged = j in reports[arm].spoof_flags
            if flagged:
                ax.scatter(exp, rep, marker="X", s=190, color="#d62828", zorder=5, edgecolor="white")
                ax.annotate(f"{arm} {j}", (exp, rep), xytext=(8, -16), textcoords="offset points",
                            color="#d62828", fontsize=9)
            else:
                ax.scatter(exp, rep, marker="D", s=26, color="#264653", zorder=4)
    ax.scatter([], [], marker="D", color="#264653", label="demo joints (accepted)")
    ax.scatter([], [], marker="X", s=100, color="#d62828", label="demo joints flagged as spoofed")
    ax.set_xlim(lo, hi); ax.set_ylim(0, 125)
    ax.set_xlabel("torque implied by temperature + vibration (% rated)")
    ax.set_ylabel("torque the sensor REPORTS (% rated)")
    ax.set_title("Cross-sensor spoof check: reported torque must agree with the physics")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "05_spoof_cross_sensor.png")


# 6 ---------------------------------------------------- telemetry traces
def fig_traces():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    t = np.arange(ram.N)
    # ARM-04 spike
    ax = axes[0, 0]
    tq = scen["ARM-04"][1]["shoulder"].torque_nm
    lim_nm = RATED["shoulder"] * LIM["torque_pct"] / 100
    ax.plot(t, tq, "-o", ms=4, color="#264653")
    ax.axhline(lim_nm, color="#d62828", ls="--"); ax.text(0.3, lim_nm + 2, "hard limit", color="#d62828")
    ax.scatter([11], [tq[11]], s=140, color="#d62828", zorder=5)
    ax.set_title("ARM-04 shoulder: one sample crosses the limit -> EMERGENCY_STOP")
    ax.set_xlabel("sample"); ax.set_ylabel("torque (N·m)")
    # ARM-06 spoof
    ax = axes[0, 1]
    w = scen["ARM-06"][1]["elbow"]
    rep = pct("elbow", w.torque_nm)
    exp = spoof.expected_torque_pct(np.mean(w.temp_c), np.mean(w.vibration_mm_s))
    ax.plot(t, rep, "-o", ms=4, color="#264653", label="reported torque")
    ax.axhline(exp, color=OK, lw=2, label="torque implied by temp/vibration")
    ax.axhspan(exp - 3 * spoof.sigma, exp + 3 * spoof.sigma, color=OK, alpha=.15)
    ax.set_title("ARM-06 elbow: reported torque far below what the heat says")
    ax.set_xlabel("sample"); ax.set_ylabel("torque (% rated)"); ax.legend(fontsize=8)
    # ARM-07 flatline
    ax = axes[1, 0]
    ax.plot(t, scen["ARM-01"][1]["wrist"].torque_nm, "-o", ms=4, color=OK, label="healthy wrist (ARM-01)")
    ax.plot(t, scen["ARM-07"][1]["wrist"].torque_nm, "-s", ms=4, color="#d62828", label="frozen wrist (ARM-07)")
    ax.set_title("ARM-07: real torque is noisy, a frozen value is not")
    ax.set_xlabel("sample"); ax.set_ylabel("torque (N·m)"); ax.legend(fontsize=8)
    # ARM-05 temp/vib
    ax = axes[1, 1]
    w = scen["ARM-05"][1]["elbow"]
    ax.plot(t, w.temp_c, color=BAD, label="temperature (°C)")
    ax.axhline(LIM["temp_c"], color=BAD, ls="--", alpha=.6)
    ax.set_ylabel("temperature (°C)", color=BAD); ax.set_ylim(40, 100)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(t, w.vibration_mm_s, color="#4c8bd6", label="vibration")
    ax2.axhline(LIM["vibration_mm_s"], color="#4c8bd6", ls="--", alpha=.6)
    ax2.set_ylabel("vibration (mm/s)", color="#4c8bd6"); ax2.set_ylim(0, 12)
    ax.set_title("ARM-05 elbow: hot and vibrating, but still under the hard limits")
    ax.set_xlabel("sample")
    fig.suptitle("Telemetry windows from the demo arms", fontsize=13)
    save(fig, "06_telemetry_traces.png")


# 7 ---------------------------------------------------- decision matrix
def fig_matrix():
    rows, cols = ["LOW", "MED", "HIGH"], ["NONE", "MINOR", "SEVERE"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid(False)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            d = ram.TABLE[(r, c)]
            ax.add_patch(plt.Rectangle((j, 2 - i), 1, 1, color=DEC_COL[d], alpha=.85, ec="white", lw=3))
            ax.text(j + .5, 2 - i + .8, d.replace("_", "\n"), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
            here = []
            for arm, rep in reports.items():
                if rep.safety_trips:
                    continue
                lvl = rep.model_level if rep.model_level != "UNAVAILABLE" else "LOW"
                if lvl == r and rep.note_level == c:
                    mark = "*" if rep.spoof_flags else ("†" if rep.model_level == "UNAVAILABLE" else "")
                    here.append(arm + mark)
            if here:
                ax.text(j + .5, 2 - i + .3, "  ".join(here), ha="center", va="center",
                        color="white", fontsize=8.5)
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xticks([.5, 1.5, 2.5]); ax.set_xticklabels(cols)
    ax.set_yticks([.5, 1.5, 2.5]); ax.set_yticklabels(rows[::-1])
    ax.set_xlabel("Maintenance-notes level"); ax.set_ylabel("Model wear level")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.set_title("Decision table with demo arms placed in their cells\n"
                 "* spoof override applied   † model down   ARM-04 bypasses the table (hard stop)",
                 fontsize=11, pad=34)
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, "07_decision_matrix.png")


# 8 ---------------------------------------------------- final decisions
def fig_decisions():
    arms = list(reports)
    rank = [ram.RANK[reports[a].decision] + 1 for a in arms]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(arms[::-1], rank[::-1], color=[DEC_COL[reports[a].decision] for a in arms[::-1]])
    for i, a in enumerate(arms[::-1]):
        ax.text(rank[::-1][i] + .05, i, f"{reports[a].decision}   ({scen[a][0]})", va="center", fontsize=8.5)
    ax.set_xlim(0, 8.5)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["SAFE", "MONITOR", "SCHEDULE", "STOP"])
    ax.set_xlabel("severity of final decision")
    ax.set_title("Final decision for every demo arm (all four outcomes reached)")
    save(fig, "08_final_decisions.png")


# 9 ---------------------------------------------------- wear probability
def fig_wear_probs():
    arms = list(reports)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, a in enumerate(arms):
        p = reports[a].wear_prob
        if p is None:
            hatch = "xx"
            label = "stopped before model" if reports[a].safety_trips else "model down"
            ax.bar(i, .04, color="none", edgecolor=GREY, hatch=hatch)
            ax.text(i, .08, label, rotation=90, va="bottom", ha="center", fontsize=8, color=GREY)
        else:
            col = "#d62828" if p >= ram.WEAR_HIGH else "#f4a261" if p >= ram.WEAR_MED else OK
            ax.bar(i, p, color=col)
            ax.text(i, p + .02, f"{p:.2f}", ha="center", fontsize=9)
    ax.axhline(ram.WEAR_MED, color="#f4a261", ls="--"); ax.text(len(arms) - .5, ram.WEAR_MED + .01, "MED", color="#f4a261", ha="right")
    ax.axhline(ram.WEAR_HIGH, color="#d62828", ls="--"); ax.text(len(arms) - .5, ram.WEAR_HIGH + .01, "HIGH", color="#d62828", ha="right")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms)
    ax.set_ylim(0, 1.1); ax.set_ylabel("predicted 30-day failure probability")
    ax.set_title("Model output per arm (worst joint)")
    save(fig, "09_wear_probability.png")


# 10 --------------------------------------------------- notes points
def fig_notes():
    arms = [f"ARM-{i:02d}" for i in range(1, 9)]
    pts = [notes[a].points if a in notes else 0 for a in arms]
    lvl = [notes[a].level if a in notes else "NONE" for a in arms]
    colmap = {"NONE": GREY, "MINOR": "#f4a261", "SEVERE": "#d62828"}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(arms, pts, color=[colmap[l] for l in lvl])
    for i, (p, l) in enumerate(zip(pts, lvl)):
        ax.text(i, p + .15, l, ha="center", fontsize=9)
        if arms[i] in notes and notes[arms[i]].evidence:
            ax.text(i, p / 2 if p else .3, "\n".join(e.split(": ")[1] for e in notes[arms[i]].evidence[:4]),
                    ha="center", va="center", fontsize=7, color="white" if p else "black")
    ax.axhline(ram.NOTE_SEVERE_POINTS, color="#d62828", ls="--")
    ax.text(7.45, ram.NOTE_SEVERE_POINTS + .1, "SEVERE at 3 pts", color="#d62828", ha="right")
    ax.set_ylabel("keyword points (recent notes only)")
    ax.set_title("Maintenance-log scan: negations, irrelevant and stale notes score zero")
    save(fig, "10_notes_scores.png")


# 11 --------------------------------------------------- spoof attack gallery
def fig_attacks():
    true = ram.window("elbow", 65, 0.85)
    N = ram.N
    cases = {}
    cases["Frozen value (35 %)"] = ram.JointWindow(ram.pct_to_nm("elbow", [35.0] * N), true.temp_c, true.vibration_mm_s)
    rec = ram.pct_to_nm("elbow", ram.rng.normal(35, 3, N // 2))
    cases["Replay of recorded window"] = ram.JointWindow(rec + rec, true.temp_c, true.vibration_mm_s)
    stepped = list(true.torque_nm[:10]) + ram.pct_to_nm("elbow", ram.rng.normal(35, 3, 10))
    cases["Sudden drop mid-window"] = ram.JointWindow(stepped, true.temp_c, true.vibration_mm_s)
    cases["Noisy fake far from truth"] = ram.JointWindow(ram.fake_torque("elbow", 35), true.temp_c, true.vibration_mm_s)
    cases["Modest lie (true - 10 %)"] = ram.JointWindow(
        ram.pct_to_nm("elbow", pct("elbow", true.torque_nm) - 10), true.temp_c, true.vibration_mm_s)
    healthy = ram.window("elbow", 50, 0.05)
    cases["Coherent forgery of all 3"] = ram.JointWindow(
        ram.fake_torque("elbow", 50), healthy.temp_c, healthy.vibration_mm_s)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    t = np.arange(N)
    for ax, (name, w) in zip(axes.flat, cases.items()):
        caught = bool(spoof.check("elbow", w))
        col = "#2a9d8f" if caught else "#d62828"
        exp = spoof.expected_torque_pct(np.mean(w.temp_c), np.mean(w.vibration_mm_s))
        ax.plot(t, pct("elbow", true.torque_nm), color=GREY, ls=":", label="true torque")
        ax.plot(t, pct("elbow", w.torque_nm), "-o", ms=3, color=col, label="reported torque")
        ax.axhspan(exp - 3 * spoof.sigma, exp + 3 * spoof.sigma, color="#4c8bd6", alpha=.13)
        ax.axhline(exp, color="#4c8bd6", lw=1)
        ax.set_title(f"{name}\n{'CAUGHT' if caught else 'MISSED (known limitation)'}", color=col, fontsize=10)
        ax.set_xlabel("sample")
    axes[0, 0].set_ylabel("torque (% rated)"); axes[1, 0].set_ylabel("torque (% rated)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Spoof-attack gallery (blue band = torque implied by temperature/vibration ±3σ)", fontsize=13)
    save(fig, "11_spoof_attack_gallery.png")


# 12 --------------------------------------------------- contact sheet
def dashboard():
    n = len(saved)
    cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(26, 5.2 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, p in zip(axes.flat, saved):
        ax.imshow(plt.imread(p))
    fig.suptitle("Robotic Arm Joint Health Monitoring - all visualisations", fontsize=24)
    fig.tight_layout()
    path = os.path.join(OUT, "00_dashboard.png")
    fig.savefig(path, dpi=80)
    print("saved", path)
    if not SHOW:
        plt.close(fig)


if __name__ == "__main__":
    for f in (fig_dataset, fig_distributions, fig_model, fig_safety_margin, fig_spoof_scatter,
              fig_traces, fig_matrix, fig_decisions, fig_wear_probs, fig_notes, fig_attacks):
        f()
    dashboard()
    print(f"\nDone. {len(saved)} charts + dashboard in: {OUT}")
    if SHOW:
        plt.show()
