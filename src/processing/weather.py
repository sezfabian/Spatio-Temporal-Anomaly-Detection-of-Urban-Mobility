"""Normalize ECCC hourly weather CSVs into interim Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.processing.paths import DEFAULT_INTERIM_DIR, DEFAULT_RAW_DIR, LOCAL_TZ

# Raw ECCC header -> tidy name
COLUMN_MAP = {
    "Date/Time (LST)": "ts_local",
    "Temp (°C)": "temp_c",
    "Dew Point Temp (°C)": "dew_point_c",
    "Rel Hum (%)": "rel_hum_pct",
    "Precip. Amount (mm)": "precip_mm",
    "Wind Dir (10s deg)": "wind_dir_10s",
    "Wind Spd (km/h)": "wind_spd_kmh",
    "Visibility (km)": "visibility_km",
    "Stn Press (kPa)": "stn_press_kpa",
    "Hmdx": "humidex",
    "Wind Chill": "wind_chill",
    "Weather": "weather_desc",
}


def list_weather_csvs(raw_dir: Path = DEFAULT_RAW_DIR) -> list[Path]:
    """List monthly Toronto City hourly weather CSVs under raw.

    Args:
        raw_dir: Raw data root.

    Returns:
        Sorted ``hourly_YYYY_MM.csv`` paths.
    """
    folder = raw_dir / "weather_toronto_city"
    return sorted(folder.glob("hourly_*.csv"))


def read_weather_csv(path: Path) -> pd.DataFrame:
    """Read and normalize one monthly ECCC hourly CSV.

    Args:
        path: Path to ``hourly_YYYY_MM.csv``.

    Returns:
        Tidy hourly weather DataFrame.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(col).strip().strip('"') for col in frame.columns]
    rename = {src: dst for src, dst in COLUMN_MAP.items() if src in frame.columns}
    out = frame.rename(columns=rename)
    keep = [col for col in COLUMN_MAP.values() if col in out.columns]
    out = out.loc[:, keep].copy()
    out["ts_local"] = pd.to_datetime(out["ts_local"], errors="coerce")
    out = out.dropna(subset=["ts_local"])
    # ECCC LST timestamps are naive local times.
    out["ts_local"] = out["ts_local"].dt.tz_localize(
        LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward"
    )
    for col in keep:
        if col in {"ts_local", "weather_desc"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)


def normalize_weather(
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
) -> Path:
    """Concatenate monthly weather CSVs into ``weather_hourly.parquet``.

    Args:
        raw_dir: Raw data root.
        interim_dir: Interim output directory.

    Returns:
        Path to written Parquet file.
    """
    files = list_weather_csvs(raw_dir)
    if not files:
        raise FileNotFoundError(f"No weather CSVs found under {raw_dir}")
    table = pd.concat([read_weather_csv(path) for path in files], ignore_index=True)
    table = table.sort_values("ts_local").drop_duplicates("ts_local", keep="last")
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "weather_hourly.parquet"
    table.to_parquet(out_path, index=False)
    return out_path
