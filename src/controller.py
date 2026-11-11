"""Deterministic rule-based baseline controller.

Policy (PROTOTYPE ASSUMPTION, not a validated comfort standard):
  effective_temperature = temperature + 0.03 * (humidity - 50)
  < 24 -> L1 | 24-27 -> L2 | 27-30 -> L3 | 30-33 -> L4 | >= 33 -> L5
Occupancy is handled by occupancy.apply_occupancy_policy (energy saving).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config as C
from .occupancy import Occupancy, OperatingState, apply_occupancy_policy, parse_occupancy
from .validation import validate_humidity, validate_temperature

_BANDS = ("below 24", "24-27", "27-30", "30-33", "33 and above")


def effective_temperature(temperature, humidity) -> float:
    t = validate_temperature(temperature)
    h = validate_humidity(humidity)
    return t + C.HUMIDITY_WEIGHT * (h - C.HUMIDITY_REFERENCE)


def comfort_level_from_conditions(temperature, humidity) -> int:
    eff = effective_temperature(temperature, humidity)
    return 1 + sum(eff >= th for th in C.TEMP_THRESHOLDS)


@dataclass(frozen=True)
class Decision:
    controller: str            # "baseline", "conventional" or "adaptive"
    comfort_level: int         # level 1-5 the fan would use if the room is occupied
    applied_level: int         # level actually applied; 0 means IDLE (fan off)
    state: OperatingState
    occupancy: Occupancy
    explanation: str
    learned_offset: int = 0
    model_active: bool = False


def build_decision(
    controller: str,
    comfort_level: int,
    occupancy,
    minutes_empty: float,
    comfort_text: str,
    energy_saving: bool = True,
    learned_offset: int = 0,
    model_active: bool = False,
    idle_delay_minutes: float = C.IDLE_DELAY_MINUTES,
    explain: bool = True,
) -> Decision:
    """Combine a comfort level with the occupancy policy. Occupancy always has priority."""
    occ = parse_occupancy(occupancy)
    if energy_saving:
        out = apply_occupancy_policy(comfort_level, occ, minutes_empty, idle_delay_minutes, explain=explain)
        applied, state, reason = out.applied_level, out.state, out.reason
    else:
        applied, state = comfort_level, OperatingState.RUNNING
        reason = ("Conventional fan: no occupancy awareness, keeps running" +
                  (" although the room is empty." if occ is Occupancy.EMPTY else ".")) if explain else ""
    text = f"{comfort_text} {reason}".strip() if explain else ""
    return Decision(controller, comfort_level, applied, state, occ, text, learned_offset, model_active)


class RuleBasedController:
    """The deterministic baseline. energy_saving=False gives a conventional always-on fan."""

    def __init__(self, energy_saving: bool = True, idle_delay_minutes: float = C.IDLE_DELAY_MINUTES):
        self.energy_saving = energy_saving
        self.idle_delay_minutes = idle_delay_minutes

    @staticmethod
    def comfort_level(temperature, humidity) -> int:
        return comfort_level_from_conditions(temperature, humidity)

    def decide(self, temperature, humidity, occupancy=Occupancy.OCCUPIED, minutes_empty: float = 0.0,
               explain: bool = True) -> Decision:
        level = comfort_level_from_conditions(temperature, humidity)
        text = ""
        if explain:
            t, h = validate_temperature(temperature), validate_humidity(humidity)
            eff = effective_temperature(t, h)
            band = _BANDS[level - 1]
            text = (f"Rule baseline: effective temperature {eff:.1f} C ({t:.1f} C "
                    f"{C.HUMIDITY_WEIGHT * (h - C.HUMIDITY_REFERENCE):+.1f} humidity adjustment) is in the "
                    f"{band} band, so Level {level}.")
        name = "baseline" if self.energy_saving else "conventional"
        return build_decision(name, level, occupancy, minutes_empty, text, self.energy_saving,
                              idle_delay_minutes=self.idle_delay_minutes, explain=explain)
