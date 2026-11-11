"""Synthetic (SIMULATED) rooms and users.  Nothing here is real sensor or user data.

Simulated day: 96 decisions of 15 minutes. Weather, occupancy and user preferences
are generated from documented assumptions with seeded random generators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .controller import comfort_level_from_conditions
from .validation import ValidationError, clip_level

PROFILE_DESCRIPTIONS = {
    "Warm-preferring": "Feels warm easily. Wants +1 level vs the baseline, +2 when humidity is 70% or more.",
    "Cool-preferring": "Likes gentle airflow. Wants -1 level vs the baseline, -2 when humidity is below 50%.",
    "Evening-oriented": "Only deviates in the evening: +2 levels between 18:00 and 23:00, baseline otherwise.",
}


def preference_offset(profile: str, humidity: float, hour: float) -> int:
    """The simulated user's TRUE (noise-free) offset from the baseline. Hidden from the learner."""
    if profile == "Warm-preferring":
        return 2 if humidity >= 70 else 1
    if profile == "Cool-preferring":
        return -2 if humidity < 50 else -1
    if profile == "Evening-oriented":
        return 2 if 18.0 <= hour < 23.0 else 0
    raise ValidationError(f"unknown profile {profile!r}")


def preferred_level(profile: str, temperature: float, humidity: float, hour: float,
                    rng: np.random.Generator | None = None, noise: float = C.PREFERENCE_NOISE) -> int:
    """Level the simulated user actually wants. With probability `noise` the choice is off by +/-1."""
    level = comfort_level_from_conditions(temperature, humidity) + preference_offset(profile, humidity, hour)
    if rng is not None and noise > 0:
        u = rng.random()
        if u < noise / 2:
            level -= 1
        elif u < noise:
            level += 1
    return clip_level(level)


def generate_weather(rng: np.random.Generator, steps: int = C.STEPS_PER_DAY) -> pd.DataFrame:
    """One simulated day of room conditions: daily mean + diurnal swing (peak ~15:00) + small noise."""
    hours = np.arange(steps) * (24.0 / steps)
    t_mean, t_amp = rng.uniform(24.0, 34.0), rng.uniform(3.0, 6.0)
    rh_mean = rng.uniform(40.0, 80.0)
    temp = t_mean + t_amp * np.sin(2 * np.pi * (hours - 9.0) / 24.0) + rng.normal(0, 0.4, steps)
    hum = rh_mean - 2.0 * (temp - t_mean) + rng.normal(0, 2.5, steps)
    return pd.DataFrame({"hour": hours,
                         "temperature": np.clip(temp, C.TEMP_MIN_C, C.TEMP_MAX_C).round(1),
                         "humidity": np.clip(hum, 20.0, 95.0).round(1)})


def generate_occupancy(rng: np.random.Generator, steps: int = C.STEPS_PER_DAY) -> np.ndarray:
    """True = occupied. Assumptions: 70% of days have one 2-8 h absence starting 07:30-10:30,
    plus 0-3 short absences (15-90 min) during 07:00-22:00. Asleep at home counts as occupied."""
    occ = np.ones(steps, dtype=bool)
    per_hour = steps / 24.0
    if rng.random() < 0.7:
        start = int(rng.uniform(7.5, 10.5) * per_hour)
        length = int(rng.uniform(2.0, 8.0) * per_hour)
        occ[start:start + length] = False
    for _ in range(int(rng.integers(0, 4))):
        start = int(rng.uniform(7.0, 22.0) * per_hour)
        length = int(rng.integers(1, 7))
        occ[start:start + length] = False
    return occ


def generate_day(rng: np.random.Generator) -> pd.DataFrame:
    day = generate_weather(rng)
    day["occupied"] = generate_occupancy(rng)
    return day


def make_rng(seed: int, profile_index: int, stream: int) -> np.random.Generator:
    """Independent, reproducible random streams: stream 1/2 = train env/user, 3/4 = test env/user."""
    return np.random.default_rng(np.random.SeedSequence([seed, profile_index, stream]))
