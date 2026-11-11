"""Smoke tests: the Streamlit app starts and its controls change the real calculations."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import config as C
from src.storage import load_feedback

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "FEEDBACK_PATH", tmp_path / "feedback.csv")
    monkeypatch.setattr(C, "BACKUP_DIR", tmp_path / "backups")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_starts_without_exceptions(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Live control", "Learning", "Energy saving", "Baseline comparison", "About & limits"]


def test_controls_change_recommendation(app):
    html = lambda: " ".join(m.value for m in app.markdown)
    app.slider(key="live_temp").set_value(23.0).run()
    app.slider(key="live_hum").set_value(40).run()
    cool = html()
    app.slider(key="live_temp").set_value(40.0).run()
    app.slider(key="live_hum").set_value(90).run()
    hot = html()
    assert "Level 1" in cool and "Level 5" in hot and not app.exception


def badges(app):
    """State badges actually rendered (the About tab text also contains the word IDLE, so match the element)."""
    import re
    return re.findall(r'class="aa-badge (aa-\w+)">(\w+)<', " ".join(m.value for m in app.markdown))


def test_empty_room_goes_idle_after_delay(app):
    assert {b[1] for b in badges(app)} == {"RUNNING"}
    app.radio(key="live_occ").set_value("Empty").run()
    assert not app.exception
    assert {b[1] for b in badges(app)} == {"REDUCED_SPEED"}                 # empty 0 min: reduced, not off
    [s for s in app.slider if s.label == "Minutes the room has been empty"][0].set_value(60).run()
    assert {b[1] for b in badges(app)} == {"IDLE"}                          # past the 30 min idle delay


def test_unknown_occupancy_does_not_idle(app):
    app.radio(key="live_occ").set_value("Unknown (sensor missing)").run()
    assert {b[1] for b in badges(app)} == {"RUNNING"}
    assert "conservative fallback" in " ".join(m.value for m in app.markdown)


def test_apply_correction_saves_and_model_learns(app, tmp_path):
    app.slider(key="live_temp").set_value(26.0).run()
    app.slider(key="live_hum").set_value(50).run()
    base_level = int([m for m in app.markdown if "Baseline (rules)" in m.value][0].value.split("Level ")[1][0])
    target = min(5, base_level + 1)
    radio = [r for r in app.radio if r.label.startswith("Which speed")][0]
    radio.set_value(target).run()
    [n for n in app.number_input if n.label == "Repeat as separate events"][0].set_value(10).run()
    [b for b in app.button if b.label == "Apply Correction"][0].click().run()
    assert not app.exception
    saved = load_feedback(C.FEEDBACK_PATH).data
    assert len(saved) == 10 and set(saved["chosen_level"]) == {target} and set(saved["profile"]) == {"Warm-preferring"}
    assert any("went from Level" in i.value for i in app.info)           # effect on future recommendation shown
    # a fresh app instance (restart) still sees the saved feedback
    fresh = AppTest.from_file(APP, default_timeout=60).run()
    assert not fresh.exception
    assert fresh.metric[0].value == "10" and fresh.metric[2].value == "Active"


def test_simulated_feedback_and_other_profile_untouched(app):
    [b for b in app.button if b.label.startswith("Simulate")][0].click().run()
    assert not app.exception
    data = load_feedback(C.FEEDBACK_PATH).data
    assert len(data) > 0 and set(data["profile"]) == {"Warm-preferring"} and set(data["source"]) == {"simulated_profile"}
    app.selectbox[0].set_value("Cool-preferring").run()
    assert app.metric[0].value == "0"                                      # profiles are separate


def test_comparison_tab_renders_all_three_targets(app):
    assert not app.exception
    assert any("Press **Run experiment**" in i.value for i in app.info)         # nothing shown before it is run
    [b for b in app.button if b.label == "Run experiment"][0].click().run()
    assert not app.exception
    frames = [d.value for d in app.dataframe]
    assert any("ACHIEVED" in " ".join(map(str, f.to_numpy().ravel())) or "NOT" in " ".join(map(str, f.to_numpy().ravel())) for f in frames)
