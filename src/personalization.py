"""Preference learning: one small decision tree per user profile.

What is learned?  The user's OFFSET from the rule baseline:
        offset = level the user chose  -  level the baseline would have used
from three inputs: temperature, humidity, hour of day.

How is it used?   adaptive level = clip( baseline level + learned offset, 1, 5 )
The occupancy policy is applied afterwards (see controller.build_decision), so a
learned preference can never keep a fan running in an empty room.

Safeguards: no learning below MIN_FEEDBACK_EVENTS; offset clipped to +/-MAX_OFFSET;
final level always clipped to 1-5; one model per profile (no cross-contamination).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

from . import config as C
from .controller import Decision, RuleBasedController, build_decision
from .validation import ValidationError, clip_level, validate_hour, validate_humidity, validate_temperature

FEATURES = ("temperature", "humidity", "sim_hour")


@dataclass(frozen=True)
class TrainingSummary:
    profile: str
    n_events: int
    n_corrections: int
    active: bool
    train_exact: float | None
    train_within_one: float | None
    message: str


@dataclass(frozen=True)
class OffsetPrediction:
    offset: int
    active: bool
    support: int          # number of training events in the matching leaf (0 if inactive)


class ProfileModel:
    """Learns one profile's offset from the baseline."""

    def __init__(self, profile: str, min_events: int = C.MIN_FEEDBACK_EVENTS):
        if profile not in C.PROFILES:
            raise ValidationError(f"unknown profile {profile!r}")
        self.profile = profile
        self.min_events = min_events
        self.tree: DecisionTreeRegressor | None = None
        self.n_train = 0
        self.summary = TrainingSummary(profile, 0, 0, False, None, None,
                                       f"No feedback yet. Need {min_events} events before learning starts; baseline is used.")

    @property
    def active(self) -> bool:
        return self.tree is not None

    # ------------------------------------------------------------ training
    def fit(self, events: pd.DataFrame) -> TrainingSummary:
        """(Re)train from feedback events. Only rows of THIS profile with occupied rooms are used."""
        self.tree, self.n_train = None, 0
        needed = set(FEATURES) | {"baseline_level", "chosen_level"}
        if events is None or len(events) == 0 or not needed.issubset(events.columns):
            self.summary = TrainingSummary(self.profile, 0, 0, False, None, None,
                                           f"No feedback yet. Need {self.min_events} events before learning starts; baseline is used.")
            return self.summary
        d = events
        if "profile" in d.columns:
            d = d[d["profile"] == self.profile]
        if "occupancy" in d.columns:
            d = d[d["occupancy"] == "occupied"]
        d = d.dropna(subset=list(needed))
        d = d[d["baseline_level"].between(C.MIN_LEVEL, C.MAX_LEVEL) & d["chosen_level"].between(C.MIN_LEVEL, C.MAX_LEVEL)]
        n = len(d)
        n_corr = int(d["corrected"].sum()) if "corrected" in d.columns else int((d["chosen_level"] != d["baseline_level"]).sum())
        if n < self.min_events:
            self.summary = TrainingSummary(self.profile, n, n_corr, False, None, None,
                                           f"Learning not active yet: {n} of {self.min_events} feedback events collected. Baseline is used.")
            return self.summary

        X = d[list(FEATURES)].astype(float)
        y = (d["chosen_level"] - d["baseline_level"]).clip(-C.MAX_OFFSET, C.MAX_OFFSET).astype(float)
        tree = DecisionTreeRegressor(max_depth=C.TREE_MAX_DEPTH, min_samples_leaf=C.TREE_MIN_LEAF, random_state=0)
        tree.fit(X, y)
        self.tree, self.n_train = tree, n

        preds = np.array([clip_level(b + self.predict_offset(*r).offset)
                          for b, r in zip(d["baseline_level"], X.itertuples(index=False, name=None))])
        chosen = d["chosen_level"].to_numpy()
        exact = float(np.mean(preds == chosen))
        within = float(np.mean(np.abs(preds - chosen) <= 1))
        self.summary = TrainingSummary(self.profile, n, n_corr, True, exact, within,
                                       f"Model trained on {n} feedback events ({n_corr} corrections).")
        return self.summary

    # ------------------------------------------------------------ prediction
    def predict_offset(self, temperature, humidity, sim_hour) -> OffsetPrediction:
        t, h, hr = validate_temperature(temperature), validate_humidity(humidity), validate_hour(sim_hour)
        if self.tree is None:
            return OffsetPrediction(0, False, 0)
        # Walk the tree by hand: identical to sklearn's predict(), but fast and easy to explain.
        tr, x = self.tree.tree_, (t, h, hr)
        node = 0
        while tr.children_left[node] != tr.children_right[node]:       # not a leaf
            node = tr.children_left[node] if x[tr.feature[node]] <= tr.threshold[node] else tr.children_right[node]
        raw = float(tr.value[node][0][0])
        offset = int(np.clip(np.floor(raw + 0.5), -C.MAX_OFFSET, C.MAX_OFFSET))
        return OffsetPrediction(offset, True, int(tr.n_node_samples[node]))

    def rules_text(self) -> str:
        if self.tree is None:
            return "Model inactive: no rules learned yet."
        return export_text(self.tree, feature_names=list(FEATURES), decimals=1)


class PersonalizationEngine:
    """Holds one ProfileModel per profile and produces adaptive Decisions."""

    def __init__(self, idle_delay_minutes: float = C.IDLE_DELAY_MINUTES):
        self.idle_delay_minutes = idle_delay_minutes
        self.baseline = RuleBasedController(energy_saving=True, idle_delay_minutes=idle_delay_minutes)
        self.models = {p: ProfileModel(p) for p in C.PROFILES}

    def model_for(self, profile: str) -> ProfileModel:
        if profile not in self.models:
            raise ValidationError(f"unknown profile {profile!r}")
        return self.models[profile]

    def retrain(self, profile: str, events: pd.DataFrame) -> TrainingSummary:
        return self.model_for(profile).fit(events)

    def retrain_all(self, events: pd.DataFrame) -> dict:
        return {p: self.retrain(p, events) for p in C.PROFILES}

    def decide(self, profile: str, temperature, humidity, sim_hour, occupancy="occupied",
               minutes_empty: float = 0.0, explain: bool = True) -> Decision:
        model = self.model_for(profile)
        base = RuleBasedController.comfort_level(temperature, humidity)
        pred = model.predict_offset(temperature, humidity, sim_hour)
        level = clip_level(base + pred.offset)
        text = ""
        if explain:
            if not pred.active:
                text = (f"Adaptive Air: baseline Level {base}; learning not active yet "
                        f"({model.summary.n_events}/{model.min_events} feedback events), so no personal offset.")
            elif pred.offset == 0:
                text = (f"Adaptive Air: baseline Level {base}; learned preference for '{profile}' "
                        f"({pred.support} similar events) says no change, so Level {level}.")
            else:
                text = (f"Adaptive Air: baseline Level {base} {pred.offset:+d} learned from {pred.support} similar "
                        f"'{profile}' feedback events, so Level {level}.")
        return build_decision("adaptive", level, occupancy, minutes_empty, text, True, pred.offset,
                              pred.active, self.idle_delay_minutes, explain)
