"""Central configuration. Every tunable number in the project lives here.

IMPORTANT: all thresholds below are PROTOTYPE ASSUMPTIONS. They are not
scientifically validated comfort standards and need real-world validation.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"          # live / demo feedback only
BACKUP_DIR = DATA_DIR / "backups"                  # created before any reset
EXPERIMENT_DIR = DATA_DIR / "experiments"          # evaluation outputs only

# ---------------------------------------------------------------- fan levels
MIN_LEVEL = 1
MAX_LEVEL = 5

# ---------------------------------------------------------------- input ranges
# Plausible indoor ranges for the prototype; anything outside is rejected.
TEMP_MIN_C = 10.0
TEMP_MAX_C = 50.0
HUMIDITY_MIN = 0.0
HUMIDITY_MAX = 100.0

# ---------------------------------------------------------------- time model
STEP_MINUTES = 15                                  # one decision every 15 simulated minutes
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES            # = 96 decisions per simulated day

# ---------------------------------------------------------------- rule baseline
# effective_temperature = T + HUMIDITY_WEIGHT * (RH - HUMIDITY_REFERENCE)
HUMIDITY_REFERENCE = 50.0
HUMIDITY_WEIGHT = 0.03                             # deg C of "feels hotter" per 1% RH above 50
# effective temp <24 ->L1, 24-27 ->L2, 27-30 ->L3, 30-33 ->L4, >=33 ->L5
TEMP_THRESHOLDS = (24.0, 27.0, 30.0, 33.0)

# ---------------------------------------------------------------- occupancy / energy saving
IDLE_DELAY_MINUTES = 30.0      # empty this long -> IDLE (fan off). Configurable.
REDUCTION_LEVELS = 2           # while empty but inside the delay: drop 2 levels (floor = Level 1)
FORGETFUL_USER_OFF_MINUTES = 60.0   # sensitivity case only: user manually switches off after 1 h

# ---------------------------------------------------------------- personalisation model
MIN_FEEDBACK_EVENTS = 10       # below this the model stays inactive and the baseline is used
MAX_OFFSET = 2                 # learned shift is clipped to +/- 2 levels
TREE_MAX_DEPTH = 3
TREE_MIN_LEAF = 4
HOLDOUT_MIN_EVENTS = 20        # live data: minimum events before a chronological hold-out is shown
HOLDOUT_FRACTION = 0.3         # last 30 % of live events are held out

# ---------------------------------------------------------------- simulated users
PROFILES = ("Warm-preferring", "Cool-preferring", "Evening-oriented")
PREFERENCE_NOISE = 0.10        # 10 % of simulated choices are off by +/-1 (human inconsistency)

# ---------------------------------------------------------------- evaluation
EVAL_SEEDS = (11, 12, 13, 14, 15)
TRAIN_DAYS = 14
TEST_DAYS = 10

# ---------------------------------------------------------------- abstract targets
TARGET_CORRECTION_REDUCTION = 0.50
TARGET_AGREEMENT_WITHIN_ONE = 0.85
TARGET_EMPTY_FAN_REDUCTION = 0.80
