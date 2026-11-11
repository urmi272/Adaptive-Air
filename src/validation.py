"""Input validation shared by every module. Invalid input raises ValidationError."""
from __future__ import annotations

import math

import numpy as np

from . import config as C


class ValidationError(ValueError):
    """Raised when an input is missing, malformed or outside the supported range."""


def _as_float(name: str, value) -> float:
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a number, got {value!r}")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationError(f"{name} must be a number, got {value!r}")
    x = float(value)
    if not math.isfinite(x):
        raise ValidationError(f"{name} is missing or not finite ({value!r})")
    return x


def validate_temperature(value) -> float:
    x = _as_float("temperature", value)
    if not C.TEMP_MIN_C <= x <= C.TEMP_MAX_C:
        raise ValidationError(f"temperature {x} C outside supported range {C.TEMP_MIN_C}-{C.TEMP_MAX_C} C")
    return x


def validate_humidity(value) -> float:
    x = _as_float("humidity", value)
    if not C.HUMIDITY_MIN <= x <= C.HUMIDITY_MAX:
        raise ValidationError(f"humidity {x} % outside supported range {C.HUMIDITY_MIN}-{C.HUMIDITY_MAX} %")
    return x


def validate_level(value, name: str = "fan level") -> int:
    x = _as_float(name, value)
    if x != int(x):
        raise ValidationError(f"{name} must be a whole number, got {value!r}")
    level = int(x)
    if not C.MIN_LEVEL <= level <= C.MAX_LEVEL:
        raise ValidationError(f"{name} {level} outside supported range {C.MIN_LEVEL}-{C.MAX_LEVEL}")
    return level


def validate_minutes(value, name: str = "minutes") -> float:
    x = _as_float(name, value)
    if x < 0:
        raise ValidationError(f"{name} cannot be negative ({x})")
    return x


def validate_hour(value) -> float:
    x = _as_float("hour of day", value)
    if not 0.0 <= x < 24.0:
        raise ValidationError(f"hour of day {x} must be in [0, 24)")
    return x


def clip_level(value) -> int:
    """Force any number into the supported 1-5 range (used after adding a learned offset)."""
    return int(min(C.MAX_LEVEL, max(C.MIN_LEVEL, round(float(value)))))
