import itertools

import numpy as np
import pytest

from src import config as C
from src.controller import RuleBasedController, comfort_level_from_conditions, effective_temperature
from src.occupancy import Occupancy, OperatingState
from src.validation import ValidationError, clip_level, validate_level


def test_abstract_example_29c_humid_gives_level_3():
    assert comfort_level_from_conditions(29, 80) == 3


def test_hotter_never_lowers_level_and_extremes_differ():
    levels = [comfort_level_from_conditions(t, 60) for t in np.arange(15, 45.5, 0.5)]
    assert levels == sorted(levels)
    assert levels[0] == 1 and levels[-1] == 5


def test_more_humidity_never_lowers_level():
    levels = [comfort_level_from_conditions(29, h) for h in range(0, 101, 5)]
    assert levels == sorted(levels)
    assert effective_temperature(30, 90) > effective_temperature(30, 30)


def test_level_always_in_range_over_full_grid():
    for t, h in itertools.product(np.linspace(C.TEMP_MIN_C, C.TEMP_MAX_C, 41), np.linspace(0, 100, 21)):
        assert C.MIN_LEVEL <= comfort_level_from_conditions(t, h) <= C.MAX_LEVEL


@pytest.mark.parametrize("t,h", [(float("nan"), 50), (None, 50), ("30", 50), (True, 50), (9.9, 50), (50.1, 50),
                                 (30, -1), (30, 100.5), (30, None), (30, float("inf"))])
def test_invalid_inputs_raise(t, h):
    with pytest.raises(ValidationError):
        comfort_level_from_conditions(t, h)


@pytest.mark.parametrize("bad", [0, 6, 2.5, None, "3", True, float("nan")])
def test_validate_level_rejects(bad):
    with pytest.raises(ValidationError):
        validate_level(bad)


def test_clip_level_bounds():
    assert clip_level(-3) == 1 and clip_level(9) == 5 and clip_level(3.4) == 3


def test_deterministic():
    c = RuleBasedController()
    assert c.decide(31, 70, "occupied") == c.decide(31, 70, "occupied")


def test_decision_has_explanation_and_running_state_when_occupied():
    d = RuleBasedController().decide(31, 70, "occupied")
    assert d.state is OperatingState.RUNNING and d.applied_level == d.comfort_level
    assert "Level" in d.explanation and d.occupancy is Occupancy.OCCUPIED
