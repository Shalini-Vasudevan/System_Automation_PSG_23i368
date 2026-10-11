# System Automation Workshop

This project demonstrates sensor monitoring and predictive maintenance using Python.

## Project Contents

- `Day1/hands_on_starter.py` simulates sensor readings for five machines, evaluates prioritized safety rules, reports machine severity, and logs each decision.
- `Day2/model_creation.py` loads `Day2/Data/day2_sample.csv`, trains a random forest classifier, and prints its test accuracy.
- `combined_output.py` runs the Day 2 model first and then the Day 1 monitoring simulation.
- `requirements.txt` lists the Python packages used by the project.

## Setup

Use Python 3.11 or 3.12, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

Run both demonstrations from the project root:

```bash
python combined_output.py
```

The launcher prints both programs' output and saves a copy to `cobined_output.txt` in the project root. The Day 1 simulation produces 50 readings per run and appends its decisions to `Day1/day1_log.txt`.

Run either demonstration on its own:

```bash
python Day2/model_creation.py
python Day1/hands_on_starter.py
```

## Day 1 Rules

The monitoring simulation generates nominal readings, occasional extreme spikes, and sensor dropouts. Rules are checked in priority order, with the lowest priority number evaluated first. Possible responses include stopping the machine, venting pressure, raising a maintenance alert, cutting power during a critical overload, or holding operation when sensor data is unavailable.