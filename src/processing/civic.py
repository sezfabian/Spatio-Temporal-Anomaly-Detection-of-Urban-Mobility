"""Export civic calendar features to an interim daily Parquet table."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.features.civic_calendar import DEFAULT_CALENDAR_PATH, load_civic_calendar
from src.processing.paths import DEFAULT_INTERIM_DIR


def normalize_civic_days(
    calendar_path: Path = DEFAULT_CALENDAR_PATH,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> Path:
    """Materialize one civic-feature row per calendar day.

    Args:
        calendar_path: Path to ``toronto_civic_calendar.yaml``.
        interim_dir: Interim output directory.
        start_date: Optional inclusive start override.
        end_date: Optional inclusive end override.

    Returns:
        Path to ``civic_days.parquet``.
    """
    calendar = load_civic_calendar(calendar_path)
    start = start_date or date(calendar.start_year, 1, 1)
    end = end_date or date(calendar.end_year, 12, 31)

    rows: list[dict] = []
    day = start
    while day <= end:
        features = calendar.features_for_date(day)
        rows.append(
            {
                "date": day,
                "is_public_holiday": features["is_public_holiday"],
                "is_civic_holiday": features["is_civic_holiday"],
                "is_holiday": features["is_holiday"],
                "is_school_break": features["is_school_break"],
                "is_mega_event": features["is_mega_event"],
                "is_parade": features["is_parade"],
                "is_shopping_peak": features["is_shopping_peak"],
                "n_civic_events": features["n_civic_events"],
                "holiday_names": "|".join(features["holiday_names"]),
                "mega_event_names": "|".join(features["mega_event_names"]),
                "event_ids": "|".join(features["event_ids"]),
                "event_kinds": "|".join(features["event_kinds"]),
            }
        )
        day += timedelta(days=1)

    table = pd.DataFrame(rows)
    table["date"] = pd.to_datetime(table["date"]).dt.date
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "civic_days.parquet"
    table.to_parquet(out_path, index=False)
    return out_path
