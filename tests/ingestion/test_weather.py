"""Unit tests for src.ingestion.weather (ECCC HTTP calls are mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml

from src.ingestion.weather import (
    build_hourly_bulk_url,
    download_weather_month,
    download_weather_range,
    iter_month_requests,
    load_weather_config,
    local_path_for_month,
    main,
    month_range,
)


SAMPLE_WEATHER = {
    "provider": "eccc",
    "dataset_key": "weather_toronto_city",
    "bulk_data_url": "https://climate.weather.gc.ca/climate_data/bulk_data_e.html",
    "climate_id": "6158355",
    "timeframe": 1,
    "start_year": 2014,
    "end_year": 2014,
}


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "toronto_datasets.yaml"
    path.write_text(yaml.safe_dump({"weather": SAMPLE_WEATHER}), encoding="utf-8")
    return path


def test_load_weather_config(config_path: Path) -> None:
    weather = load_weather_config(config_path)
    assert weather["climate_id"] == "6158355"


def test_load_weather_config_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("datasets: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing weather"):
        load_weather_config(path)


def test_month_range_inclusive() -> None:
    assert list(month_range(2014, 2014))[:2] == [(2014, 1), (2014, 2)]
    assert list(month_range(2014, 2014))[-1] == (2014, 12)
    assert len(list(month_range(2014, 2015))) == 24


def test_build_hourly_bulk_url_contains_expected_params() -> None:
    url = build_hourly_bulk_url(
        SAMPLE_WEATHER["bulk_data_url"],
        "6158355",
        2014,
        7,
    )
    assert "climate_id=6158355" in url
    assert "Year=2014" in url
    assert "Month=7" in url
    assert "timeframe=1" in url
    assert "format=csv" in url


def test_local_path_for_month(tmp_path: Path) -> None:
    path = local_path_for_month(tmp_path, "weather_toronto_city", 2014, 7)
    assert path == tmp_path / "weather_toronto_city" / "hourly_2014_07.csv"


def test_iter_month_requests_builds_12_months_for_one_year(tmp_path: Path) -> None:
    requests_ = iter_month_requests(SAMPLE_WEATHER, tmp_path)
    assert len(requests_) == 12
    assert requests_[0].destination.name == "hourly_2014_01.csv"
    assert requests_[-1].destination.name == "hourly_2014_12.csv"


def test_download_weather_month_streams_csv(tmp_path: Path) -> None:
    request = iter_month_requests(SAMPLE_WEATHER, tmp_path)[0]
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    response.headers = {"content-type": "application/force-download"}
    response.iter_content.return_value = [b"a,b\n", b"1,2\n"]

    session = MagicMock(spec=requests.Session)
    session.get.return_value = response

    result = download_weather_month(request, session)
    assert result.status == "downloaded"
    assert request.destination.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_download_weather_month_skips_existing(tmp_path: Path) -> None:
    request = iter_month_requests(SAMPLE_WEATHER, tmp_path)[0]
    request.destination.parent.mkdir(parents=True)
    request.destination.write_text("already", encoding="utf-8")
    session = MagicMock(spec=requests.Session)

    result = download_weather_month(request, session, force=False)
    assert result.status == "skipped"
    session.get.assert_not_called()


def test_download_weather_month_rejects_html_payload(tmp_path: Path) -> None:
    request = iter_month_requests(SAMPLE_WEATHER, tmp_path)[0]
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    response.headers = {"content-type": "text/html; charset=UTF-8"}
    response.iter_content.return_value = [b"<html>"]

    session = MagicMock(spec=requests.Session)
    session.get.return_value = response

    result = download_weather_month(request, session)
    assert result.status == "failed"
    assert "HTML" in (result.error or "")


def test_download_weather_range_uses_year_overrides(tmp_path: Path) -> None:
    session = MagicMock(spec=requests.Session)
    with patch("src.ingestion.weather.download_weather_month") as download_month:
        download_month.side_effect = lambda req, *_a, **_k: MagicMock(
            request=req, status="skipped", error=None
        )
        results = download_weather_range(
            SAMPLE_WEATHER,
            tmp_path,
            session,
            start_year=2014,
            end_year=2014,
        )
    assert len(results) == 12
    assert download_month.call_count == 12


def test_main_returns_nonzero_on_failure(config_path: Path, tmp_path: Path) -> None:
    failed = MagicMock()
    failed.status = "failed"
    failed.request = MagicMock(year=2014, month=1, destination=tmp_path / "x.csv")
    failed.error = "boom"

    with patch("src.ingestion.weather.download_weather_range", return_value=[failed]):
        code = main(["--config", str(config_path), "--raw-dir", str(tmp_path)])
    assert code == 1
