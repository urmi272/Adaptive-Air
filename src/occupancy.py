"""Occupancy handling and the energy-saving policy.

This module is deliberately independent of the learning model: the policy
below is applied AFTER a comfort level has been chosen, so occupancy always
wins over personalisation.

Policy (all simulated):
  OCCUPIED -> run at the comfort level, state RUNNING
  EMPTY    -> minutes_empty <  idle delay : REDUCED_SPEED (comfort level minus 2, floor Level 1)
              minutes_empty >= idle delay : IDLE (fan off, applied level 0)
  UNKNOWN  -> conservative fallback: behave as OCCUPIED (never switch off on missing data),
              and say so in the explanation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import config as C
from .validation import ValidationError, validate_level, validate_minutes


class Occupancy(str, Enum):
    OCCUPIED = "occupied"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class OperatingState(str, Enum):
    RUNNING = "RUNNING"
    REDUCED_SPEED = "REDUCED_SPEED"
    IDLE = "IDLE"


_OCCUPIED_WORDS = {"occupied", "present", "home", "true", "yes", "1"}
_EMPTY_WORDS = {"empty", "vacant", "away", "absent", "false", "no", "0"}
_UNKNOWN_WORDS = {"unknown", "none", "nan", "null", ""}


def parse_occupancy(value, strict: bool = False) -> Occupancy:
    """Turn anything into an Occupancy.

    Missing values (None, NaN, "unknown") -> UNKNOWN.
    Malformed values -> UNKNOWN by default, or ValidationError when strict=True.
    Crucially, nothing missing or malformed is ever turned into EMPTY.
    """
    if isinstance(value, Occupancy):
        return value
    if value is None:
        return Occupancy.UNKNOWN
    if isinstance(value, (bool, np.bool_)):
        return Occupancy.OCCUPIED if bool(value) else Occupancy.EMPTY
    if isinstance(value, (int, np.integer)):
        if int(value) == 1:
            return Occupancy.OCCUPIED
        if int(value) == 0:
            return Occupancy.EMPTY
    elif isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return Occupancy.UNKNOWN
        if value == 1.0:
            return Occupancy.OCCUPIED
        if value == 0.0:
            return Occupancy.EMPTY
    elif isinstance(value, str):
        word = value.strip().lower()
        if word in _OCCUPIED_WORDS:
            return Occupancy.OCCUPIED
        if word in _EMPTY_WORDS:
            return Occupancy.EMPTY
        if word in _UNKNOWN_WORDS:
            return Occupancy.UNKNOWN
    if strict:
        raise ValidationError(f"invalid occupancy value {value!r}")
    return Occupancy.UNKNOWN


@dataclass(frozen=True)
class OccupancyOutcome:
    applied_level: int      # 0 means the fan is off (IDLE)
    state: OperatingState
    reason: str


def apply_occupancy_policy(
    comfort_level: int,
    occupancy,
    minutes_empty: float = 0.0,
    idle_delay_minutes: float = C.IDLE_DELAY_MINUTES,
    reduction_levels: int = C.REDUCTION_LEVELS,
    explain: bool = True,
) -> OccupancyOutcome:
    level = validate_level(comfort_level, "comfort level")
    occ = parse_occupancy(occupancy)
    minutes = validate_minutes(minutes_empty, "minutes empty")
    delay = validate_minutes(idle_delay_minutes, "idle delay")

    if occ is Occupancy.OCCUPIED:
        return OccupancyOutcome(level, OperatingState.RUNNING, "Room occupied: running at the comfort level." if explain else "")
    if occ is Occupancy.UNKNOWN:
        return OccupancyOutcome(
            level, OperatingState.RUNNING,
            "Occupancy unknown or invalid: conservative fallback keeps the fan running (missing data is NOT treated as empty)." if explain else "",
        )
    # EMPTY
    if minutes >= delay:
        return OccupancyOutcome(
            0, OperatingState.IDLE,
            f"Room empty for {minutes:.0f} min (idle delay {delay:.0f} min): fan IDLE to save energy." if explain else "",
        )
    reduced = max(C.MIN_LEVEL, level - int(reduction_levels))
    return OccupancyOutcome(
        reduced, OperatingState.REDUCED_SPEED,
        f"Room empty for {minutes:.0f} min: reduced to Level {reduced}; IDLE after {delay:.0f} min." if explain else "",
    )
