import pandas as pd
import pytest

from src import storage as S
from src.validation import ValidationError


def ev(**kw):
    base = dict(profile="Warm-preferring", sim_hour=19.0, temperature=30.0, humidity=70.0, occupancy="occupied",
                baseline_level=4, adaptive_level=4, chosen_level=5, source="manual")
    base.update(kw)
    return S.make_event(**base)


def test_event_records_all_required_fields_and_correction_flag():
    e = ev()
    for col in S.COLUMNS:
        assert col in e
    assert e["corrected"] is True and ev(chosen_level=4)["corrected"] is False
    assert e["baseline_level"] == 4 and e["adaptive_level"] == 4 and e["chosen_level"] == 5


@pytest.mark.parametrize("kw", [dict(chosen_level=6), dict(chosen_level=0), dict(temperature=99),
                                dict(humidity=-5), dict(occupancy="empty"), dict(occupancy="unknown"),
                                dict(profile="Nobody"), dict(source="hacked"), dict(sim_hour=24)])
def test_make_event_rejects_invalid(kw):
    with pytest.raises(ValidationError):
        ev(**kw)


def test_roundtrip_and_persistence_across_restart(tmp_path):
    p = tmp_path / "fb.csv"
    S.append_events([ev(), ev(profile="Cool-preferring", chosen_level=3)], p)
    first = S.load_feedback(p)
    S.append_events([ev(chosen_level=4)], p)
    second = S.load_feedback(p)                      # a fresh load = an application restart
    assert len(first.data) == 2 and len(second.data) == 3
    assert list(second.data["step"]) == [1, 2, 1] or sorted(second.data["step"]) == [1, 1, 2]
    assert set(second.data["profile"]) == {"Warm-preferring", "Cool-preferring"}
    assert second.data["corrected"].dtype == bool


def test_missing_file_gives_empty_frame(tmp_path):
    r = S.load_feedback(tmp_path / "nope.csv")
    assert len(r.data) == 0 and list(r.data.columns) == S.COLUMNS


def test_malformed_rows_skipped_counted_and_kept_on_disk(tmp_path):
    p = tmp_path / "fb.csv"
    S.append_events([ev(), ev()], p)
    with open(p, "a") as fh:
        fh.write("zzz,2026-01-01,Warm-preferring,notanumber,30,70,occupied,4,4,5,True,manual\n")
        fh.write("dup,2026-01-01,Nobody,19,30,70,occupied,4,4,5,True,manual\n")
    r = S.load_feedback(p)
    assert len(r.data) == 2 and r.skipped_rows == 2 and r.notes
    assert "zzz" in p.read_text()                    # never silently deleted


def test_inconsistent_corrected_flag_is_skipped(tmp_path):
    p = tmp_path / "fb.csv"
    S.append_events([ev()], p)
    with open(p, "a") as fh:
        fh.write("bad1,2026-01-01T00:00:00,Warm-preferring,19,30,70,occupied,4,4,5,False,manual\n")
    assert S.load_feedback(p).skipped_rows == 1


def test_corrupted_file_is_quarantined_not_deleted(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text("this,is,not\nthe,right,columns\n")
    r = S.load_feedback(p)
    assert len(r.data) == 0 and r.notes
    assert not p.exists()
    kept = list(tmp_path.glob("fb.corrupt-*.csv"))
    assert len(kept) == 1 and "right,columns" in kept[0].read_text()


def test_binary_garbage_is_quarantined(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_bytes(b"\xff\xfe\x00\x01garbage\x80\x81")
    r = S.load_feedback(p)
    assert len(r.data) == 0 and list(tmp_path.glob("*.corrupt-*"))


def test_append_after_missing_trailing_newline(tmp_path):
    p = tmp_path / "fb.csv"
    S.append_events([ev()], p)
    p.write_text(p.read_text().rstrip("\n"))
    S.append_events([ev(chosen_level=3)], p)
    assert len(S.load_feedback(p).data) == 2


def test_reset_creates_backup_then_empties(tmp_path):
    p, b = tmp_path / "fb.csv", tmp_path / "backups"
    S.append_events([ev(), ev()], p)
    backup = S.reset_feedback(p, b)
    assert backup is not None and len(S.load_feedback(backup).data) == 2
    assert len(S.load_feedback(p).data) == 0
    assert S.reset_feedback(p, b) is None            # nothing to back up the second time
