import numpy as np
import pytest

from src.controller import RuleBasedController
from src.occupancy import Occupancy, OperatingState, apply_occupancy_policy, parse_occupancy
from src.personalization import PersonalizationEngine
from src.validation import ValidationError


def test_occupied_runs_at_comfort_level():
    o = apply_occupancy_policy(4, "occupied")
    assert (o.applied_level, o.state) == (4, OperatingState.RUNNING)


def test_empty_inside_delay_reduces_speed_then_idles():
    assert apply_occupancy_policy(4, "empty", 0).state is OperatingState.REDUCED_SPEED
    assert apply_occupancy_policy(4, "empty", 0).applied_level == 2
    assert apply_occupancy_policy(1, "empty", 15).applied_level == 1        # floor at Level 1
    idle = apply_occupancy_policy(4, "empty", 30)
    assert idle.state is OperatingState.IDLE and idle.applied_level == 0


def test_idle_delay_is_configurable():
    assert apply_occupancy_policy(3, "empty", 20, idle_delay_minutes=10).state is OperatingState.IDLE
    assert apply_occupancy_policy(3, "empty", 20, idle_delay_minutes=60).state is OperatingState.REDUCED_SPEED
    assert apply_occupancy_policy(3, "empty", 0, idle_delay_minutes=0).state is OperatingState.IDLE


@pytest.mark.parametrize("bad", [None, float("nan"), "unknown", "banana", 7, "", [1]])
def test_missing_or_invalid_occupancy_never_means_empty(bad):
    assert parse_occupancy(bad) is Occupancy.UNKNOWN
    o = apply_occupancy_policy(4, bad, minutes_empty=999)
    assert o.state is OperatingState.RUNNING and o.applied_level == 4
    assert "unknown" in o.reason.lower()


@pytest.mark.parametrize("bad", ["banana", 7, [1]])
def test_strict_parse_raises(bad):
    with pytest.raises(ValidationError):
        parse_occupancy(bad, strict=True)


def test_parse_known_values():
    assert parse_occupancy(True) is Occupancy.OCCUPIED and parse_occupancy(np.bool_(False)) is Occupancy.EMPTY
    assert parse_occupancy("Away") is Occupancy.EMPTY and parse_occupancy(" Occupied ") is Occupancy.OCCUPIED
    assert parse_occupancy(1) is Occupancy.OCCUPIED and parse_occupancy(0.0) is Occupancy.EMPTY


def test_negative_minutes_rejected():
    with pytest.raises(ValidationError):
        apply_occupancy_policy(3, "empty", -5)


def test_conventional_fan_keeps_running_in_empty_room():
    d = RuleBasedController(energy_saving=False).decide(30, 60, "empty", 500)
    assert d.state is OperatingState.RUNNING and d.applied_level == d.comfort_level


def test_occupancy_overrides_learned_preference():
    """A strongly learned +2 offset must not keep the fan running in an empty room."""
    eng = PersonalizationEngine()
    model = eng.model_for("Warm-preferring")

    class _Stub:                                     # force an active +2 offset
        offset = 2
    model.tree = object()
    model.predict_offset = lambda *a: type("P", (), {"offset": 2, "active": True, "support": 99})()
    occupied = eng.decide("Warm-preferring", 30, 60, 19, "occupied")
    empty_late = eng.decide("Warm-preferring", 30, 60, 19, "empty", 45)
    assert occupied.comfort_level == min(5, 4 + 0 + 2) and occupied.state is OperatingState.RUNNING
    assert empty_late.state is OperatingState.IDLE and empty_late.applied_level == 0
    assert eng.decide("Warm-preferring", 30, 60, 19, None, 45).state is OperatingState.RUNNING
