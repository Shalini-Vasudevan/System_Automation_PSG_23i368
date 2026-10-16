# Robotic Arm Joint Health Monitoring System  (23IF02 Bonus Challenge)

## Quick start (Windows / Mac / Linux)
```
pip install numpy pandas scikit-learn matplotlib
python robotic_arm_monitor.py     # runs the full pipeline, prints all 4 decisions
python visualize.py               # draws 11 charts + dashboard into ./visuals/
```

## What is in this folder
| Item | Purpose |
|---|---|
| `robotic_arm_monitor.py` | MAIN FILE - whole system in one file (safety stop, ML wear model, notes scanner, spoof defence, decision logic, demo) |
| `visualize.py` | Draws all charts from the pipeline (needs matplotlib) |
| `historical_joint_data.csv` | Synthetic dataset: 150 labelled readings (64 failures) |
| `maintenance_log.txt` | Technician log: 12 entries, relevant + irrelevant + a negation + a stale note |
| `EXPLANATION.md` | Half-page written explanation (thresholds, combining logic, spoof defence) |
| `sample_output.txt` | Output of one full run |
| `visuals/` | Pre-generated PNG charts (`00_dashboard.png` = all on one page) |
| `modular_version/` | Same system split into modules (config, safety, model, notes, spoof, decision, demo) - run `python demo.py` inside it |

## Requirements covered
1. Immediate safety stop - deterministic, no ML (`check_safety`)
2. Predictive wear detection - logistic regression on 150 synthetic readings
3. Maintenance notes as second signal - keyword scan with negation + staleness handling
4. Spoofed-sensor defence - flatline, replay, impossible step, cross-sensor physics check
5. One decision per arm - SAFE / MONITOR / SCHEDULE_MAINTENANCE / EMERGENCY_STOP (all four demonstrated)

## Submit
Push the whole folder to your personal repo.
