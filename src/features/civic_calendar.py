"""Toronto civic calendar loader and timestamp feature helpers."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CALENDAR_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "toronto_civic_calendar.yaml"
)

HOLIDAY_KINDS = frozenset({"public_holiday", "civic_holiday"})
MEGA_KINDS = frozenset({"mega_event", "parade", "shopping_peak", "school_break"})


@dataclass(frozen=True)
class CivicEvent:
    """One civic-calendar entry with an inclusive date range.

    Attributes:
        id: Stable event identifier from config.
        name: Human-readable event name.
        kind: Event category (``public_holiday``, ``parade``, ...).
        start_date: Inclusive start date.
        end_date: Inclusive end date.
        scope: Spatial scope label (currently ``citywide``).
        notes: Optional free-text notes.
    """

    id: str
    name: str
    kind: str
    start_date: date
    end_date: date
    scope: str = "citywide"
    notes: str | None = None

    def active_on(self, day: date) -> bool:
        """Return whether this event covers ``day`` (inclusive range)."""
        return self.start_date <= day <= self.end_date


@dataclass(frozen=True)
class CivicCalendar:
    """In-memory civic calendar indexed for fast per-day lookups.

    Attributes:
        timezone: IANA timezone name from config (informational for now).
        start_year: First year covered.
        end_year: Last year covered.
        events: All configured events.
        by_date: Mapping of calendar date -> events active that day.
    """

    timezone: str
    start_year: int
    end_year: int
    events: tuple[CivicEvent, ...]
    by_date: dict[date, tuple[CivicEvent, ...]]

    def events_on(self, day: date) -> tuple[CivicEvent, ...]:
        """Return events active on ``day`` (empty tuple if none)."""
        return self.by_date.get(day, ())

    def features_for_date(self, day: date) -> dict[str, Any]:
        """Build model-ready civic features for a calendar date.

        Args:
            day: Local calendar date (America/Toronto civil date).

        Returns:
            Feature dict with boolean flags, event counts, and name lists.
        """
        active = self.events_on(day)
        kinds = {event.kind for event in active}
        holiday_names = [e.name for e in active if e.kind in HOLIDAY_KINDS]
        mega_names = [e.name for e in active if e.kind in MEGA_KINDS]
        return {
            "date": day.isoformat(),
            "is_public_holiday": any(e.kind == "public_holiday" for e in active),
            "is_civic_holiday": any(e.kind == "civic_holiday" for e in active),
            "is_holiday": bool(kinds & HOLIDAY_KINDS),
            "is_school_break": any(e.kind == "school_break" for e in active),
            "is_mega_event": any(e.kind == "mega_event" for e in active),
            "is_parade": any(e.kind == "parade" for e in active),
            "is_shopping_peak": any(e.kind == "shopping_peak" for e in active),
            "n_civic_events": len(active),
            "holiday_names": holiday_names,
            "mega_event_names": mega_names,
            "event_ids": [e.id for e in active],
            "event_kinds": sorted(kinds),
        }

    def features_for_timestamp(self, ts: datetime | date | str) -> dict[str, Any]:
        """Build civic features for a timestamp by using its calendar date.

        Args:
            ts: ``datetime``, ``date``, or ISO date/datetime string. Datetimes
                are converted with ``.date()`` (caller should localize first).

        Returns:
            Same feature mapping as ``features_for_date``.
        """
        return self.features_for_date(_as_date(ts))


def _as_date(value: datetime | date | str) -> date:
    """Normalize supported inputs to a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if "T" in text:
        return datetime.fromisoformat(text).date()
    return date.fromisoformat(text[:10])


def _parse_event(raw: dict[str, Any]) -> CivicEvent:
    """Parse one YAML event mapping into a ``CivicEvent``."""
    start = date.fromisoformat(str(raw["start_date"]))
    end = date.fromisoformat(str(raw.get("end_date", raw["start_date"])))
    if end < start:
        raise ValueError(f"Event {raw.get('id')} has end_date before start_date")
    return CivicEvent(
        id=str(raw["id"]),
        name=str(raw["name"]),
        kind=str(raw["kind"]),
        start_date=start,
        end_date=end,
        scope=str(raw.get("scope", "citywide")),
        notes=raw.get("notes"),
    )


def load_civic_calendar(path: Path | str = DEFAULT_CALENDAR_PATH) -> CivicCalendar:
    """Load and index the Toronto civic calendar YAML.

    Args:
        path: Path to ``toronto_civic_calendar.yaml``.

    Returns:
        ``CivicCalendar`` with a per-day index for O(1) lookups.

    Raises:
        ValueError: If config structure is invalid.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "events" not in config:
        raise ValueError(f"Invalid civic calendar config: {config_path}")

    events = tuple(_parse_event(item) for item in config["events"])
    by_date: dict[date, list[CivicEvent]] = {}
    for event in events:
        day = event.start_date
        while day <= event.end_date:
            by_date.setdefault(day, []).append(event)
            day = date.fromordinal(day.toordinal() + 1)

    return CivicCalendar(
        timezone=str(config.get("timezone", "America/Toronto")),
        start_year=int(config["start_year"]),
        end_year=int(config["end_year"]),
        events=events,
        by_date={day: tuple(items) for day, items in by_date.items()},
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for inspecting civic-calendar features."""
    parser = argparse.ArgumentParser(
        description="Inspect Toronto civic calendar features for a date."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CALENDAR_PATH,
        help="Path to toronto_civic_calendar.yaml",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date to inspect (YYYY-MM-DD).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: print civic features for one date.

    Args:
        argv: Optional argument list (without program name).

    Returns:
        Process exit code ``0``.
    """
    args = _build_parser().parse_args(argv)
    calendar = load_civic_calendar(args.config)
    features = calendar.features_for_date(date.fromisoformat(args.date))
    for key, value in features.items():
        print(f"{key}\t{value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
