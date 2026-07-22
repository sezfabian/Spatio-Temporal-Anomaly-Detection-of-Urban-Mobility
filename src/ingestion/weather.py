"""Download hourly ECCC climate observations for Toronto City."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

import requests
import yaml

from src.ingestion.ckan import DEFAULT_CONFIG_PATH
from src.ingestion.download import CHUNK_SIZE, DEFAULT_RAW_DIR

DEFAULT_DAY = 1  # ECCC bulk hourly downloads are month-scoped; Day is required but unused.


@dataclass(frozen=True)
class WeatherMonthRequest:
    """One month of hourly weather data to fetch from ECCC.

    Attributes:
        year: Calendar year.
        month: Calendar month (1-12).
        url: Fully constructed bulk-download URL.
        destination: Local CSV path under ``data/raw``.
    """

    year: int
    month: int
    url: str
    destination: Path


@dataclass(frozen=True)
class WeatherDownloadResult:
    """Outcome of downloading one weather-month CSV.

    Attributes:
        request: The month request that was attempted.
        status: One of ``downloaded``, ``skipped``, or ``failed``.
        bytes_written: Bytes written on a successful download.
        error: Error message when ``status`` is ``failed``.
    """

    request: WeatherMonthRequest
    status: str
    bytes_written: int = 0
    error: str | None = None


def load_weather_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the ``weather`` section from the Toronto datasets YAML config.

    Args:
        path: Path to ``toronto_datasets.yaml``.

    Returns:
        Mapping of weather settings (``climate_id``, year range, bulk URL, ...).

    Raises:
        ValueError: If the file has no ``weather`` mapping.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    weather = (config or {}).get("weather")
    if not isinstance(weather, dict):
        raise ValueError(f"Missing weather config mapping in {config_path}")
    return weather


def month_range(start_year: int, end_year: int) -> Iterator[tuple[int, int]]:
    """Yield ``(year, month)`` pairs inclusively from start_year to end_year.

    Args:
        start_year: First year to include.
        end_year: Last year to include.

    Yields:
        ``(year, month)`` tuples with month in ``1..12``.
    """
    if end_year < start_year:
        raise ValueError(f"end_year {end_year} is before start_year {start_year}")
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def build_hourly_bulk_url(
    bulk_data_url: str,
    climate_id: str,
    year: int,
    month: int,
    *,
    timeframe: int = 1,
    day: int = DEFAULT_DAY,
) -> str:
    """Build an ECCC bulk CSV URL for one month of hourly observations.

    Args:
        bulk_data_url: ECCC bulk endpoint
            (``https://climate.weather.gc.ca/climate_data/bulk_data_e.html``).
        climate_id: Station climate identifier (Toronto City = ``6158355``).
        year: Year to download.
        month: Month to download (1-12).
        timeframe: ECCC timeframe code (``1`` = hourly).
        day: Day query param required by ECCC; ignored for monthly hourly pulls.

    Returns:
        Absolute URL including query string.
    """
    query = urlencode(
        {
            "format": "csv",
            "climate_id": climate_id,
            "Year": year,
            "Month": month,
            "Day": day,
            "timeframe": timeframe,
            "submit": "Download Data",
        }
    )
    return f"{bulk_data_url}?{query}"


def local_path_for_month(
    raw_dir: Path,
    dataset_key: str,
    year: int,
    month: int,
) -> Path:
    """Build the local path for one hourly weather-month CSV.

    Args:
        raw_dir: Raw data root (typically ``data/raw``).
        dataset_key: Subdirectory name (e.g. ``weather_toronto_city``).
        year: Year.
        month: Month (1-12).

    Returns:
        Path like ``data/raw/weather_toronto_city/hourly_2014_07.csv``.
    """
    return raw_dir / dataset_key / f"hourly_{year}_{month:02d}.csv"


def iter_month_requests(
    weather_config: dict[str, Any],
    raw_dir: Path,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[WeatherMonthRequest]:
    """Expand weather config into concrete month download requests.

    Args:
        weather_config: Parsed ``weather`` config mapping.
        raw_dir: Raw data root directory.
        start_year: Optional override for config ``start_year``.
        end_year: Optional override for config ``end_year``.

    Returns:
        List of ``WeatherMonthRequest`` objects in chronological order.
    """
    start = start_year if start_year is not None else int(weather_config["start_year"])
    end = end_year if end_year is not None else int(weather_config["end_year"])
    dataset_key = weather_config["dataset_key"]
    bulk_url = weather_config["bulk_data_url"]
    climate_id = str(weather_config["climate_id"])
    timeframe = int(weather_config.get("timeframe", 1))

    requests_: list[WeatherMonthRequest] = []
    for year, month in month_range(start, end):
        url = build_hourly_bulk_url(
            bulk_url,
            climate_id,
            year,
            month,
            timeframe=timeframe,
        )
        destination = local_path_for_month(raw_dir, dataset_key, year, month)
        requests_.append(
            WeatherMonthRequest(
                year=year,
                month=month,
                url=url,
                destination=destination,
            )
        )
    return requests_


def download_weather_month(
    request: WeatherMonthRequest,
    session: requests.Session,
    *,
    force: bool = False,
    chunk_size: int = CHUNK_SIZE,
    timeout: int = 120,
) -> WeatherDownloadResult:
    """Stream one monthly hourly CSV to disk.

    Args:
        request: Month URL + destination.
        session: Shared ``requests.Session``.
        force: Re-download when the destination already exists.
        chunk_size: Streaming chunk size in bytes.
        timeout: Per-request timeout in seconds.

    Returns:
        ``WeatherDownloadResult`` with status ``downloaded``, ``skipped``,
        or ``failed``.
    """
    destination = request.destination
    if destination.exists() and not force:
        return WeatherDownloadResult(request=request, status="skipped")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    bytes_written = 0

    try:
        with session.get(request.url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            if "html" in content_type:
                raise RuntimeError(
                    f"ECCC returned HTML instead of CSV for {request.year}-{request.month:02d}"
                )
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_written += len(chunk)
        partial.replace(destination)
    except Exception as exc:  # noqa: BLE001 - surface transfer failures as results
        if partial.exists():
            partial.unlink(missing_ok=True)
        return WeatherDownloadResult(
            request=request,
            status="failed",
            bytes_written=bytes_written,
            error=str(exc),
        )

    return WeatherDownloadResult(
        request=request,
        status="downloaded",
        bytes_written=bytes_written,
    )


def download_weather_range(
    weather_config: dict[str, Any],
    raw_dir: Path,
    session: requests.Session,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    force: bool = False,
) -> list[WeatherDownloadResult]:
    """Download all configured hourly weather months.

    Args:
        weather_config: Parsed ``weather`` config mapping.
        raw_dir: Raw data root directory.
        session: Shared HTTP session.
        start_year: Optional year-range override.
        end_year: Optional year-range override.
        force: Re-download existing month files.

    Returns:
        One result per month request, in chronological order.
    """
    results: list[WeatherDownloadResult] = []
    for request in iter_month_requests(
        weather_config,
        raw_dir,
        start_year=start_year,
        end_year=end_year,
    ):
        results.append(download_weather_month(request, session, force=force))
    return results


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ECCC hourly weather downloads.

    Args:
        None.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download hourly ECCC climate CSVs for Toronto City into data/raw."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to toronto_datasets.yaml",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Destination root for raw files (default: data/raw).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Override config start_year.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Override config end_year.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download month files even if they already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: download hourly Toronto City weather months.

    Args:
        argv: Optional argument list (without the program name).

    Returns:
        ``0`` if every month succeeded or was skipped; ``1`` if any failed.
    """
    args = _build_parser().parse_args(argv)
    weather_config = load_weather_config(args.config)
    session = requests.Session()
    results = download_weather_range(
        weather_config,
        args.raw_dir,
        session,
        start_year=args.start_year,
        end_year=args.end_year,
        force=args.force,
    )

    failures = 0
    for result in results:
        req = result.request
        print(
            f"{result.status}\t{req.year}-{req.month:02d}\t{req.destination}"
            + (f"\t{result.error}" if result.error else "")
        )
        if result.status == "failed":
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
