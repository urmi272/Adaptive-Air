import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.controller import comfort_level_from_conditions as base_level
from src.personalization import FEATURES, PersonalizationEngine, ProfileModel
from src.storage import make_event
from src.validation import ValidationError


def events(profile, n, offset, hour=19.0, seed=0, temp_range=(26, 34)):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        t, h = round(rng.uniform(*temp_range), 1), round(rng.uniform(40, 80), 1)
        b = base_level(t, h)
        rows.append(make_event(profile, hour, t, h, "occupied", b, b, min(5, max(1, b + offset))))
    return pd.DataFrame(rows)


def test_inactive_below_minimum_events_uses_baseline():
    m = ProfileModel("Warm-preferring")
    s = m.fit(events("Warm-preferring", C.MIN_FEEDBACK_EVENTS - 1, 1))
    assert not m.active and not s.active and "not active" in s.message
    eng = PersonalizationEngine()
    eng.retrain("Warm-preferring", events("Warm-preferring", 3, 1))
    d = eng.decide("Warm-preferring", 30, 60, 19)
    assert d.comfort_level == base_level(30, 60) and d.learned_offset == 0 and not d.model_active


def test_enough_feedback_changes_future_recommendation():
    eng = PersonalizationEngine()
    before = eng.decide("Warm-preferring", 29, 60, 19).comfort_level
    s = eng.retrain("Warm-preferring", events("Warm-preferring", 30, +1))
    after = eng.decide("Warm-preferring", 29, 60, 19)
    assert s.active and after.model_active
    assert after.comfort_level == before + 1 and after.learned_offset == 1
    assert "learned" in after.explanation


def test_negative_preference_is_learned():
    eng = PersonalizationEngine()
    eng.retrain("Cool-preferring", events("Cool-preferring", 30, -1))
    assert eng.decide("Cool-preferring", 31, 60, 14).comfort_level == base_level(31, 60) - 1


def test_profiles_are_independent():
    eng = PersonalizationEngine()
    # temperature ranges chosen so a +2 / -1 preference is reachable without hitting the Level 1/5 limits
    df = pd.concat([events("Warm-preferring", 30, +2, seed=1, temp_range=(24, 28)),
                    events("Cool-preferring", 30, -1, seed=2, temp_range=(27, 31))], ignore_index=True)
    eng.retrain_all(df)
    assert eng.decide("Warm-preferring", 26, 60, 12).learned_offset == 2
    assert eng.decide("Cool-preferring", 29, 60, 12).learned_offset == -1
    assert not eng.model_for("Evening-oriented").active
    assert eng.decide("Evening-oriented", 30, 60, 12).learned_offset == 0
    # Training the warm model on a frame containing ONLY cool data must not use that data.
    warm = ProfileModel("Warm-preferring")
    assert not warm.fit(events("Cool-preferring", 40, -1)).active


def test_feedback_for_one_profile_does_not_change_another_profile_model():
    eng = PersonalizationEngine()
    eng.retrain("Cool-preferring", events("Cool-preferring", 30, -1))
    snapshot = eng.decide("Cool-preferring", 30, 60, 12).comfort_level
    eng.retrain("Warm-preferring", events("Warm-preferring", 30, +2))
    assert eng.decide("Cool-preferring", 30, 60, 12).comfort_level == snapshot


def test_evening_profile_learns_time_dependence():
    eve = pd.concat([events("Evening-oriented", 40, +2, hour=20.0, seed=3), events("Evening-oriented", 40, 0, hour=13.0, seed=4)],
                    ignore_index=True)
    eng = PersonalizationEngine()
    eng.retrain("Evening-oriented", eve)
    assert eng.decide("Evening-oriented", 30, 60, 20.0).learned_offset == 2
    assert eng.decide("Evening-oriented", 30, 60, 13.0).learned_offset == 0


def test_output_always_within_1_to_5():
    eng = PersonalizationEngine()
    eng.retrain("Warm-preferring", events("Warm-preferring", 30, +2, temp_range=(34, 40)))
    assert eng.decide("Warm-preferring", 45, 95, 12).comfort_level == 5          # 5 + 2 clipped
    eng.retrain("Cool-preferring", events("Cool-preferring", 30, -2, temp_range=(15, 20)))
    assert eng.decide("Cool-preferring", 15, 20, 12).comfort_level == 1           # 1 - 2 clipped


def test_offset_never_exceeds_max_even_for_extreme_feedback():
    df = events("Warm-preferring", 30, 0)
    df["chosen_level"] = 5
    df["baseline_level"] = 1
    df["adaptive_level"] = 1
    df["corrected"] = True
    m = ProfileModel("Warm-preferring")
    m.fit(df)
    assert m.predict_offset(30, 60, 12).offset == C.MAX_OFFSET


def test_manual_tree_walk_matches_sklearn_predict():
    m = ProfileModel("Warm-preferring")
    m.fit(pd.concat([events("Warm-preferring", 40, +1, seed=5), events("Warm-preferring", 40, +2, hour=21.0, seed=6)], ignore_index=True))
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = [rng.uniform(15, 45), rng.uniform(20, 95), rng.uniform(0, 23.9)]
        raw = m.tree.predict(pd.DataFrame([x], columns=list(FEATURES)))[0]
        expected = int(np.clip(np.floor(raw + 0.5), -C.MAX_OFFSET, C.MAX_OFFSET))
        assert m.predict_offset(*x).offset == expected


@pytest.mark.parametrize("df", [None, pd.DataFrame(), pd.DataFrame({"x": [1]})])
def test_empty_or_malformed_training_data_is_safe(df):
    m = ProfileModel("Warm-preferring")
    s = m.fit(df)
    assert not s.active and m.predict_offset(30, 60, 12).offset == 0


def test_nan_rows_are_ignored_not_crashing():
    df = events("Warm-preferring", 30, 1)
    df.loc[0, "temperature"] = np.nan
    assert ProfileModel("Warm-preferring").fit(df).n_events == 29


def test_unknown_profile_and_invalid_inputs_raise():
    with pytest.raises(ValidationError):
        ProfileModel("Mystery")
    with pytest.raises(ValidationError):
        PersonalizationEngine().decide("Mystery", 30, 60, 12)
    with pytest.raises(ValidationError):
        PersonalizationEngine().decide("Warm-preferring", 30, 60, 25)
    with pytest.raises(ValidationError):
        PersonalizationEngine().decide("Warm-preferring", float("nan"), 60, 12)
