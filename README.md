# Adaptive Air

**Team Breezient · VIT Bhopal University · atomQUEST 2026, Track 1** (*Fan, Reimagined for the Next Decade: From Fans to Intelligent Comfort Ecosystems*)

> **This is a software-only simulation prototype.** Room temperature, humidity, occupancy, users and the fan are all
> simulated. No physical fan is connected or controlled, and nothing here claims real-world comfort or energy savings.

## 1. Problem
A fan is one of the most-used devices in an Indian home, yet it still waits for instructions. People get up to change
the speed when the afternoon heats up, forget to switch it off when they leave, and re-adjust the same setting day
after day. Smart fans add an app or voice control, but the *user is still the controller*. Comfort is also personal:
the same 29 °C humid evening feels different to different people.

## 2. Solution
Adaptive Air is a comfort engine that **decides** a fan speed (Level 1–5) from room conditions and **learns** from the
user's corrections:

1. A deterministic **rule baseline** gives a safe speed from day one (hotter/more humid → faster).
2. A small **per-user decision tree** learns how *this* user differs from the baseline and shifts future decisions.
3. An **occupancy policy** (kept separate from learning, and always applied last) reduces the speed and then idles the fan when the room is empty.

## 3. Architecture and data flow
```
 simulated room (temp, humidity, hour, occupancy)
        │
        ▼
 controller.py      rule baseline  → comfort level 1-5
        │
        ▼
 personalization.py + learned offset for the selected profile → adaptive comfort level (clipped 1-5)
        │
        ▼
 occupancy.py       occupancy policy (priority!) → applied level 0-5 + state RUNNING / REDUCED_SPEED / IDLE
        │
        ▼
 simulated fan (dashboard)  ──user correction──►  storage.py (CSV)  ──►  retrain that profile's model
```
Module-by-module detail: [docs/architecture.md](docs/architecture.md). Beginner-friendly code tour: [docs/code_walkthrough.md](docs/code_walkthrough.md).

## 4. Technologies
Python 3.12 · Streamlit (dashboard) · NumPy / pandas (data) · scikit-learn (decision tree) · Plotly (charts) · pytest (tests).
No hardware, cloud service, paid API or internet connection is needed at runtime.

## 5. Install
```bash
cd adaptive-air
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Run
```bash
streamlit run app.py                 # the dashboard (opens in your browser)
python -m pytest                     # the automated tests
python -m src.evaluation             # the evaluation experiment (prints results, writes data/experiments/evaluation_results.json)
```
Demo walkthrough: [docs/demo_script.md](docs/demo_script.md).

## 7. How the baseline works
`effective temperature = temperature + 0.03 × (humidity − 50)`, then
`<24 → L1 · 24–27 → L2 · 27–30 → L3 · 30–33 → L4 · ≥33 → L5`.
Example from the abstract: 29 °C with 80 % humidity → effective 29.9 → **Level 3**.
These thresholds are **prototype assumptions, not validated comfort standards** (`src/config.py`). The function is deterministic,
so baseline and adaptive controllers can be compared fairly.

## 8. How preference learning works
* Every feedback event stores: timestamp, profile, simulated hour, temperature, humidity, occupancy, baseline level, adaptive level, chosen level, whether a correction occurred, and its source.
* The model learns the **offset** `chosen level − baseline level` from `(temperature, humidity, hour)` using a depth-3 `DecisionTreeRegressor`, **one per profile**.
* Adaptive level = `clip(baseline + round(predicted offset), 1, 5)`; the offset itself is limited to ±2.
* **Safeguards:** nothing is learned until a profile has 10 events (the baseline is used and the UI says so); offsets are clipped; profiles never share data; retraining happens only when feedback is saved, not on page reruns.
* Both *corrections* and *"Accept as is"* clicks are training signals.

## 9. How occupancy-based energy saving works
| Occupancy | Behaviour |
|---|---|
| Occupied | Run at the comfort level (`RUNNING`) |
| Empty < idle delay (default 30 min) | Comfort level − 2, minimum Level 1 (`REDUCED_SPEED`) |
| Empty ≥ idle delay | Fan off (`IDLE`) |
| Unknown / missing / invalid | Conservative fallback: keep running. **Missing data is never treated as "empty".** |

This policy runs *after* the learned preference, so personalisation can never keep an empty room's fan running.

## 10. How the evaluation works
`python -m src.evaluation` (or the *Baseline comparison* tab). A simulated day has **96 decisions** (every 15 min).
For each of 5 seeds × 3 simulated profiles: Adaptive Air learns for **14 days** (retraining at the end of each day), is then **frozen**, and is scored on
**10 new days** drawn from independent random streams. Baseline and adaptive controllers see identical rooms and identical simulated user preferences.

* **Corrections/day:** occupied decisions where the recommended level ≠ the simulated user's preferred level.
* **Agreement:** share of occupied decisions within one level of the preferred level (exact match reported too).
* **Empty-room fan-on time:** empty-room decisions where the fan is not `IDLE`. The denominator is the conventional fan's fan-on decisions while empty; that fan has no occupancy awareness and never switches itself off (the abstract's baseline).

Simulated users (documented in `src/simulator.py`): *Warm-preferring* (+1 level, +2 if humidity ≥ 70 %), *Cool-preferring* (−1, −2 if humidity < 50 %), *Evening-oriented* (+2 between 18:00 and 23:00). 10 % of choices are randomly off by ±1.

## 11. Measured results (actual output, synthetic data)
| Metric | Abstract target | Baseline | Adaptive Air | Status |
|---|---|---|---|---|
| Corrections per day (14 days of learning, then held-out days) | ≥ 50 % fewer | 48.8 / day | 6.3 / day (**87.1 % fewer**) | Achieved |
| Agreement within one level (held-out) | ≥ 85 % | 82.1 % | 99.97 % (shown as 100.0 %) | Achieved |
| Fan-on time while room empty | ≥ 80 % less | 2,463 decisions | 444 decisions (**82.0 % less**) | Achieved |

Sample: 15 runs · 150 held-out days · 11,937 occupied and 2,463 empty-room decisions. **Read these results carefully:**
* The simulated users are learnable by construction (their preference is a simple rule of the model's own inputs). The results show the learning loop works on such users, **not** that real people behave this way.
* Within-one agreement is lenient: the rule baseline already scores 82.1 %. Exact agreement (38.6 % → 92.1 %) is more informative; roughly 90 % is the approximate limit set by the 10 % noise (slightly exceeded here because ±1 noise is absorbed when the preferred level is already 1 or 5).
* The energy result **passes narrowly** and depends on the 30-minute idle delay and the simulated absence pattern: a 45-minute delay gives 74.5 %, which would miss the target. If the conventional fan's user switched it off after 60 minutes instead of never, the reduction would be 43.7 %.

Full detail and the pytest results: [docs/testing_results.md](docs/testing_results.md).

## 12. Tests
`python -m pytest` → **95 tests, all passing** at the time of writing (see docs/testing_results.md for the exact run). They cover the baseline, validation, occupancy policy, storage/persistence/corruption handling, learning, profile separation, leakage checks, metric calculations and the Streamlit app flows.

## 13. Limitations
* Everything is simulated: sensors, occupancy, user preferences and the fan.
* The rule thresholds, simulated user preferences and absence patterns are assumptions requiring real-world validation.
* A "correction" here is a per-decision mismatch (every 15 min); a real person would not correct that often, so counts are an upper bound on real interventions.
* Energy is represented only by simulated fan-on time. No power or kWh is measured.
* The model is a tiny decision tree: noisy on few events, and it cannot extrapolate to conditions it has never seen.
* Real hardware integration would need its own safety design and testing; software tests do not establish physical safety.
* The dashboard's visual styling and fan animation were checked structurally (the app starts and renders without errors in automated tests) but not visually in a real browser by the author of the code; please eyeball it before recording.

## 14. Future work
Connect the decision output to a connected fan through a thin adapter (with a separate safety design); calibrate thresholds with real comfort data; learn from real occupancy sensing; compare other lightweight models; add a per-user "forget old preferences" decay.

## 15. Data handling
Feedback is saved to `data/feedback.csv` (git-ignored). Experiment outputs go to `data/experiments/`. Resetting demo data (sidebar) first copies the file to `data/backups/`. Corrupt files are renamed `*.corrupt-<time>.csv`, never deleted; malformed rows are skipped, counted and left on disk.
