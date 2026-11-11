import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import evaluation as E
from src.personalization import ProfileModel
from src.simulator import generate_day, make_rng, preferred_level


def test_agreement_rates_hand_computed():
    r = E.agreement_rates([1, 2, 3, 4, 5], [1, 3, 5, 4, 2])
    assert r["n"] == 5 and r["exact"] == pytest.approx(2 / 5) and r["within_one"] == pytest.approx(3 / 5)
    assert E.agreement_rates([], [])["exact"] is None


def test_reduction_and_classify():
    assert E.reduction(100, 25) == pytest.approx(0.75) and E.reduction(0, 5) is None
    assert E.classify(0.9, 0.8) == "ACHIEVED"
    assert E.classify(0.5, 0.8, 0.0) == "PARTIALLY ACHIEVED"
    assert E.classify(0.0, 0.8, 0.0) == "NOT ACHIEVED"
    assert E.classify(None, 0.8) == "NOT EVALUABLE"
    assert E.classify(0.84, 0.85, 0.82) == "PARTIALLY ACHIEVED" and E.classify(0.80, 0.85, 0.82) == "NOT ACHIEVED"


@pytest.fixture(scope="module")
def small():
    return E.run_experiment(seeds=(1, 2), train_days=5, test_days=3)


def test_experiment_is_reproducible(small):
    again = E.run_experiment(seeds=(1, 2), train_days=5, test_days=3)
    for k in ("overall", "per_profile", "learning_curve", "energy_sensitivity", "targets"):
        assert small[k] == again[k]


def test_sample_sizes_and_denominators(small):
    o = small["overall"]
    assert o["runs"] == 6 and o["test_days"] == 18
    assert o["conventional_on_empty"] == o["empty_decisions"]                  # denominator definition
    assert o["adaptive_on_empty"] <= o["empty_decisions"]
    assert o["occupied_decisions"] + o["empty_decisions"] == 18 * C.STEPS_PER_DAY


def test_every_day_has_96_decisions():
    day = generate_day(make_rng(1, 0, 3))
    assert len(day) == C.STEPS_PER_DAY == 96 and day["occupied"].dtype == bool


def test_decisions_partition_into_occupied_and_empty():
    model = ProfileModel("Warm-preferring")
    held = E.evaluate_heldout(model, "Warm-preferring", 4, make_rng(1, 0, 3), make_rng(1, 0, 4))
    assert held["occupied_decisions"] + held["empty_decisions"] == 4 * C.STEPS_PER_DAY


def test_inactive_model_means_adaptive_equals_baseline():
    held = E.evaluate_heldout(ProfileModel("Warm-preferring"), "Warm-preferring", 4, make_rng(2, 0, 3), make_rng(2, 0, 4))
    assert held["adaptive_corrections"] == held["baseline_corrections"]


def test_targets_are_computed_from_overall_not_hard_coded(small):
    o = small["overall"]
    by_name = {t["name"]: t for t in small["targets"]}
    assert by_name["Manual corrections per day (after 14 days)"]["measured"] == pytest.approx(1 - o["adaptive_corrections"] / o["baseline_corrections"])
    assert by_name["Fan-on time while room is empty"]["measured"] == pytest.approx(1 - o["adaptive_on_empty"] / o["conventional_on_empty"])
    assert by_name["Agreement within one level (held-out days)"]["measured"] == o["adaptive_within_one"]


def test_energy_metric_matches_independent_analytic_calculation():
    """Controller-path count (evaluate_heldout) must equal the closed-form count from run lengths."""
    res = E.run_experiment(seeds=(3,), train_days=3, test_days=6)
    at_default = [s for s in res["energy_sensitivity"] if s["idle_delay_minutes"] == C.IDLE_DELAY_MINUTES][0]
    assert at_default["on_steps"] == res["overall"]["adaptive_on_empty"]


def test_training_never_sees_held_out_days(monkeypatch):
    seen = []
    real_fit = ProfileModel.fit

    def spy(self, events):
        if events is not None and len(events) and "day" in events.columns:
            seen.append(int(events["day"].max()))
        return real_fit(self, events)
    monkeypatch.setattr(ProfileModel, "fit", spy)
    E.run_experiment(seeds=(4,), train_days=4, test_days=3, profiles=("Warm-preferring",))
    assert seen and max(seen) <= 4                                              # only training days 1..4 were fitted


def test_train_and_test_streams_are_independent():
    train = generate_day(make_rng(7, 0, 1))
    test = generate_day(make_rng(7, 0, 3))
    assert not train["temperature"].equals(test["temperature"])
    assert preferred_level("Warm-preferring", 30, 60, 12, make_rng(7, 0, 2)) == preferred_level("Warm-preferring", 30, 60, 12, make_rng(7, 0, 2))


def test_learning_actually_reduces_corrections_in_simulation():
    events, daily, model = E.run_online_phase("Warm-preferring", 6, make_rng(5, 0, 1), make_rng(5, 0, 2))
    assert model.active and daily[0]["adaptive_corrections"] == daily[0]["baseline_corrections"]   # nothing learned on day 1
    assert sum(d["adaptive_corrections"] for d in daily[2:]) < sum(d["baseline_corrections"] for d in daily[2:])


def _live(n, profile="Warm-preferring"):
    from src.storage import make_event
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        t, h = round(rng.uniform(26, 34), 1), round(rng.uniform(40, 80), 1)
        from src.controller import comfort_level_from_conditions as b
        base = b(t, h)
        rows.append(make_event(profile, 19.0, t, h, "occupied", base, base, min(5, base + 1)))
    df = pd.DataFrame(rows)
    df["step"] = range(1, n + 1)
    return df


def test_live_report_refuses_to_invent_results_with_little_data():
    assert E.live_feedback_report(_live(0), "Warm-preferring")["n_events"] == 0
    rep = E.live_feedback_report(_live(12), "Warm-preferring")
    assert rep["holdout"] is None and "needs at least" in rep["message"]


def test_live_report_uses_chronological_holdout_without_overlap():
    df = _live(40)
    rep = E.live_feedback_report(df, "Warm-preferring")
    h = rep["holdout"]
    assert h["n_train"] == int(40 * 0.7) and h["n_train"] + h["n_test"] == 40
    assert h["test_adaptive"]["exact"] >= h["test_baseline"]["exact"]
    assert E.live_feedback_report(df, "Cool-preferring")["n_events"] == 0        # other profile untouched


def test_rolling_agreement_columns():
    r = E.rolling_agreement(_live(30))
    assert {"Baseline", "Adaptive"} <= set(r.columns) and r[["Baseline", "Adaptive"]].max().max() <= 1.0
