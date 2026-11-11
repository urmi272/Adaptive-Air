"""Local CSV persistence for user feedback.

Design rules:
  * Append-only writes: existing lines (even malformed ones) are never rewritten or dropped.
  * Malformed rows are skipped when loading and COUNTED, never silently deleted from disk.
  * An unreadable file is renamed to *.corrupt-<timestamp>.csv (kept), not deleted.
  * Reset is deliberate and always writes a backup copy first.
  * Synthetic evaluation data never goes here (see config.EXPERIMENT_DIR).
"""
from __future__ import annotations

import csv
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config as C
from .occupancy import Occupancy, parse_occupancy
from .validation import ValidationError, validate_hour, validate_humidity, validate_level, validate_temperature

COLUMNS = [
    "event_id", "recorded_at", "profile", "sim_hour", "temperature", "humidity", "occupancy",
    "baseline_level", "adaptive_level", "chosen_level", "corrected", "source",
]
SOURCES = ("manual", "accepted", "simulated_profile")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_event(profile: str, sim_hour, temperature, humidity, occupancy, baseline_level,
               adaptive_level, chosen_level, source: str = "manual", recorded_at: str | None = None) -> dict:
    """Validate everything and return one feedback event (a plain dict)."""
    if profile not in C.PROFILES:
        raise ValidationError(f"unknown profile {profile!r}")
    occ = parse_occupancy(occupancy, strict=True)
    if occ is not Occupancy.OCCUPIED:
        raise ValidationError("feedback is only recorded while the room is occupied "
                              "(occupancy policy overrides speed when the room is empty or unknown)")
    if source not in SOURCES:
        raise ValidationError(f"unknown source {source!r}")
    adaptive = validate_level(adaptive_level, "adaptive level")
    chosen = validate_level(chosen_level, "chosen level")
    return {
        "event_id": uuid.uuid4().hex[:10],
        "recorded_at": recorded_at or _now(),
        "profile": profile,
        "sim_hour": validate_hour(sim_hour),
        "temperature": validate_temperature(temperature),
        "humidity": validate_humidity(humidity),
        "occupancy": occ.value,
        "baseline_level": validate_level(baseline_level, "baseline level"),
        "adaptive_level": adaptive,
        "chosen_level": chosen,
        "corrected": chosen != adaptive,
        "source": source,
    }


def _parse_row(row: dict) -> dict:
    """Validate one CSV row (strings) and return a typed event, or raise ValidationError."""
    try:
        event = {
            "event_id": str(row["event_id"]).strip(),
            "recorded_at": str(row["recorded_at"]).strip(),
            "profile": str(row["profile"]).strip(),
            "sim_hour": validate_hour(float(row["sim_hour"])),
            "temperature": validate_temperature(float(row["temperature"])),
            "humidity": validate_humidity(float(row["humidity"])),
            "occupancy": parse_occupancy(row["occupancy"], strict=True).value,
            "baseline_level": validate_level(float(row["baseline_level"]), "baseline level"),
            "adaptive_level": validate_level(float(row["adaptive_level"]), "adaptive level"),
            "chosen_level": validate_level(float(row["chosen_level"]), "chosen level"),
            "source": str(row["source"]).strip(),
        }
        corrected_text = str(row["corrected"]).strip().lower()
    except (KeyError, TypeError, ValueError) as exc:      # ValidationError is a ValueError
        raise ValidationError(f"malformed row: {exc}") from exc
    if not event["event_id"] or not event["recorded_at"]:
        raise ValidationError("empty event_id or timestamp")
    if event["profile"] not in C.PROFILES:
        raise ValidationError(f"unknown profile {event['profile']!r}")
    if event["occupancy"] != Occupancy.OCCUPIED.value:
        raise ValidationError("non-occupied feedback is not valid")
    if event["source"] not in SOURCES:
        raise ValidationError(f"unknown source {event['source']!r}")
    if corrected_text not in ("true", "false"):
        raise ValidationError("corrected must be True/False")
    event["corrected"] = corrected_text == "true"
    if event["corrected"] != (event["chosen_level"] != event["adaptive_level"]):
        raise ValidationError("corrected flag contradicts chosen/adaptive levels")
    return event


@dataclass
class LoadResult:
    data: pd.DataFrame
    skipped_rows: int = 0
    notes: list = field(default_factory=list)


def empty_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: [] for c in COLUMNS})
    return df.astype({"sim_hour": float, "temperature": float, "humidity": float, "baseline_level": int,
                      "adaptive_level": int, "chosen_level": int, "corrected": bool})


def _quarantine(path: Path, why: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
    path.rename(target)
    return f"Feedback file was unreadable ({why}); kept as {target.name} and started a fresh file."


def load_feedback(path: Path | None = None) -> LoadResult:
    path = Path(path or C.FEEDBACK_PATH)      # resolved at call time so tests can redirect it
    if not path.exists() or path.stat().st_size == 0:
        return LoadResult(empty_frame())
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        return LoadResult(empty_frame(), 0, [_quarantine(path, type(exc).__name__)])
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        return LoadResult(empty_frame(), 0, [_quarantine(path, f"missing columns {missing}")])

    events, skipped, seen = [], 0, set()
    for row in raw.to_dict("records"):
        try:
            ev = _parse_row(row)
        except ValidationError:
            skipped += 1
            continue
        if ev["event_id"] in seen:
            skipped += 1
            continue
        seen.add(ev["event_id"])
        events.append(ev)
    notes = [f"Skipped {skipped} malformed or duplicate row(s); they remain in the file on disk."] if skipped else []
    if not events:
        return LoadResult(empty_frame(), skipped, notes)
    df = pd.DataFrame(events, columns=COLUMNS)
    df["step"] = df.groupby("profile").cumcount() + 1          # per-profile sequence number
    return LoadResult(df, skipped, notes)


def append_events(events: list[dict], path: Path | None = None) -> int:
    """Append already-validated events. Returns the number written."""
    path = Path(path or C.FEEDBACK_PATH)
    if not events:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (not path.exists()) or path.stat().st_size == 0
    needs_newline = False
    if not needs_header:
        with open(path, "rb") as fh:
            fh.seek(-1, 2)
            needs_newline = fh.read(1) not in (b"\n", b"\r")
    with open(path, "a", newline="", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(events)
    return len(events)


def reset_feedback(path: Path | None = None, backup_dir: Path | None = None) -> Path | None:
    """Deliberate reset: copy the file to backups/ first, then start a fresh file. Returns backup path."""
    path = Path(path or C.FEEDBACK_PATH)
    backup = None
    has_rows = path.exists() and len([ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]) > 1
    if has_rows:                                       # back up anything beyond a bare header line
        backup_dir = Path(backup_dir or C.BACKUP_DIR)
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = backup_dir / f"feedback-{stamp}.csv"
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=COLUMNS).writeheader()
    return backup
