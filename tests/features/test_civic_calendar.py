"""Unit tests for src.features.civic_calendar."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

from src.features.civic_calendar import load_civic_calendar, main


SAMPLE = {
    "timezone": "America/Toronto",
    "start_year": 2014,
    "end_year": 2014,
    "events": [
        {
            "id": "2014_christmas_day",
            "name": "Christmas Day",
            "kind": "public_holiday",
            "start_date": "2014-12-25",
            "end_date": "2014-12-25",
            "scope": "citywide",
        },
        {
            "id": "2014_santa_claus_parade",
            "name": "Santa Claus Parade",
            "kind": "parade",
            "start_date": "2014-11-16",
            "end_date": "2014-11-16",
            "scope": "citywide",
        },
        {
            "id": "2014_tiff",
            "name": "Toronto International Film Festival",
            "kind": "mega_event",
            "start_date": "2014-09-04",
            "end_date": "2014-09-14",
            "scope": "citywide",
        },
    ],
}


@pytest.fixture
def calendar_path(tmp_path: Path) -> Path:
    path = tmp_path / "toronto_civic_calendar.yaml"
    path.write_text(yaml.safe_dump(SAMPLE), encoding="utf-8")
    return path


def test_load_civic_calendar_indexes_multi_day_events(calendar_path: Path) -> None:
    calendar = load_civic_calendar(calendar_path)
    assert calendar.events_on(date(2014, 9, 4))
    assert calendar.events_on(date(2014, 9, 14))
    assert not calendar.events_on(date(2014, 9, 15))


def test_features_for_holiday(calendar_path: Path) -> None:
    calendar = load_civic_calendar(calendar_path)
    features = calendar.features_for_date(date(2014, 12, 25))
    assert features["is_holiday"] is True
    assert features["is_public_holiday"] is True
    assert features["holiday_names"] == ["Christmas Day"]
    assert features["n_civic_events"] == 1


def test_features_for_parade(calendar_path: Path) -> None:
    calendar = load_civic_calendar(calendar_path)
    features = calendar.features_for_timestamp("2014-11-16T12:00:00")
    assert features["is_parade"] is True
    assert features["is_holiday"] is False
    assert "Santa Claus Parade" in features["mega_event_names"]


def test_features_for_ordinary_day(calendar_path: Path) -> None:
    calendar = load_civic_calendar(calendar_path)
    features = calendar.features_for_timestamp(datetime(2014, 4, 1, 8, 0))
    assert features["is_holiday"] is False
    assert features["n_civic_events"] == 0
    assert features["event_ids"] == []


def test_invalid_end_before_start(tmp_path: Path) -> None:
    bad = {
        "timezone": "America/Toronto",
        "start_year": 2014,
        "end_year": 2014,
        "events": [
            {
                "id": "bad",
                "name": "Bad",
                "kind": "public_holiday",
                "start_date": "2014-12-26",
                "end_date": "2014-12-25",
            }
        ],
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="end_date before start_date"):
        load_civic_calendar(path)


def test_repo_calendar_loads_and_has_christmas_2014() -> None:
    calendar = load_civic_calendar()
    features = calendar.features_for_date(date(2014, 12, 25))
    assert features["is_public_holiday"] is True
    assert any("Christmas" in name for name in features["holiday_names"])


def test_main_prints_features(calendar_path: Path, capsys) -> None:
    code = main(["--config", str(calendar_path), "--date", "2014-12-25"])
    assert code == 0
    out = capsys.readouterr().out
    assert "is_holiday\tTrue" in out
    assert "Christmas Day" in out
