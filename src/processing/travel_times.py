"""Normalize Bluetooth travel-time ZIP CSVs into interim Parquet."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.processing.paths import DEFAULT_INTERIM_DIR, DEFAULT_RAW_DIR, LOCAL_TZ


def list_travel_time_zips(raw_dir: Path = DEFAULT_RAW_DIR) -> list[Path]:
    """List citywide Bluetooth travel-time ZIP archives under raw.

    Args:
        raw_dir: Raw data root (typically ``data/raw``).

    Returns:
        Sorted paths matching ``travel_times_bluetooth/travel-time-*.zip``.
    """
    folder = raw_dir / "travel_times_bluetooth"
    return sorted(folder.glob("travel-time-*.zip"))


def _year_from_zip_name(path: Path) -> int | None:
    """Extract a 4-digit year from a travel-time ZIP filename if present."""
    for part in path.stem.split("-"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def read_travel_time_zip(path: Path) -> pd.DataFrame:
    """Read and normalize one travel-time ZIP into a tidy DataFrame.

    Args:
        path: Path to a ``travel-time-YYYY.zip`` archive containing a CSV.

    Returns:
        DataFrame with columns ``route_id``, ``travel_time_s``, ``sample_count``,
        ``ts_local``, ``year``.
    """
    year = _year_from_zip_name(path)
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and "__macosx" not in name.lower()
            and not Path(name).name.startswith("._")
        ]
        if not members:
            raise FileNotFoundError(f"No CSV member found in {path}")
        for member in members:
            with archive.open(member) as handle:
                frame = pd.read_csv(handle)
            frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    required = {"resultId", "timeInSeconds", "count", "updated"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    # Mixed EST/EDT offsets (-05/-04) become object dtype unless parsed via UTC first.
    ts_utc = pd.to_datetime(raw["updated"], utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "route_id": raw["resultId"].astype("string"),
            "travel_time_s": pd.to_numeric(raw["timeInSeconds"], errors="coerce"),
            "sample_count": pd.to_numeric(raw["count"], errors="coerce"),
            "ts_local": ts_utc.dt.tz_convert(LOCAL_TZ),
        }
    )
    out["year"] = year if year is not None else out["ts_local"].dt.year
    out = out.dropna(subset=["route_id", "ts_local", "travel_time_s"])
    out = out.loc[out["travel_time_s"] > 0].reset_index(drop=True)
    return out


def normalize_travel_times(
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    *,
    years: list[int] | None = None,
) -> Path:
    """Normalize all (or selected) travel-time years to interim Parquet.

    Args:
        raw_dir: Raw data root.
        interim_dir: Interim output directory.
        years: Optional year filter; ``None`` means all discovered ZIPs.

    Returns:
        Path to ``travel_times.parquet``.
    """
    zips = list_travel_time_zips(raw_dir)
    if years is not None:
        wanted = set(years)
        zips = [path for path in zips if _year_from_zip_name(path) in wanted]
    if not zips:
        raise FileNotFoundError(f"No travel-time ZIPs found under {raw_dir}")

    frames = [read_travel_time_zip(path) for path in zips]
    table = pd.concat(frames, ignore_index=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "travel_times.parquet"
    table.to_parquet(out_path, index=False)
    return out_path
