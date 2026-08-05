"""Build per-route congestion values over time for animated maps."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

INTERVAL_OPTIONS: dict[str, str] = {
    "5 min": "5min",
    "15 min": "15min",
    "30 min": "30min",
    "1 hour": "h",
}


def parse_day(value: str | date | datetime | pd.Timestamp) -> date:
    """Normalize a date-like value to ``datetime.date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def panel_date_bounds(panel: pd.DataFrame | Path) -> tuple[date, date]:
    """Return ``(min_date, max_date)`` covered by the panel."""
    if isinstance(panel, (str, Path)):
        frame = pd.read_parquet(panel, columns=["date"])
    else:
        if "date" in panel.columns:
            frame = panel.loc[:, ["date"]]
        elif "ts_local" in panel.columns:
            frame = pd.DataFrame({"date": pd.to_datetime(panel["ts_local"]).dt.date})
        else:
            raise KeyError("panel needs date or ts_local to determine bounds")
    dates = pd.to_datetime(frame["date"]).dt.date
    return dates.min(), dates.max()


def build_congestion_frames(
    panel: pd.DataFrame | Path,
    *,
    start: str | date | datetime,
    end: str | date | datetime,
    freq: str = "h",
    metric: str = "delay_s",
    max_frames: int = 500,
) -> tuple[list[str], pd.DataFrame]:
    """Aggregate ``metric`` per route for each time bucket in ``[start, end]``.

    Args:
        panel: Processed panel path or DataFrame.
        start, end: Inclusive calendar dates.
        freq: Pandas offset alias (``5min``, ``15min``, ``30min``, ``h``).
        metric: Congestion metric column.
        max_frames: Safety cap on animation frames.

    Returns:
        ``(frame_labels, values)`` where ``values`` is indexed by ``route_id``
        with one column per frame.
    """
    start_day = parse_day(start)
    end_day = parse_day(end)
    if end_day < start_day:
        raise ValueError(f"end date {end_day} is before start date {start_day}")

    columns = ["route_id", "ts_local", "date", metric]
    if isinstance(panel, (str, Path)):
        try:
            frame = pd.read_parquet(
                panel,
                columns=columns,
                filters=[
                    [
                        ("date", ">=", start_day),
                        ("date", "<=", end_day),
                    ]
                ],
            )
        except Exception:
            frame = pd.read_parquet(panel, columns=columns)
            as_dates = pd.to_datetime(frame["date"]).dt.date
            frame = frame.loc[(as_dates >= start_day) & (as_dates <= end_day)]
    else:
        frame = panel.copy()
        if "date" not in frame.columns and "ts_local" in frame.columns:
            frame["date"] = pd.to_datetime(frame["ts_local"]).dt.date
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            raise KeyError(f"panel missing columns: {missing}")
        as_dates = pd.to_datetime(frame["date"]).dt.date
        frame = frame.loc[(as_dates >= start_day) & (as_dates <= end_day)]

    if frame.empty:
        raise ValueError("No panel rows in the selected date period")

    bucket = pd.to_datetime(frame["ts_local"]).dt.floor(freq)
    grouped = (
        frame.assign(_bucket=bucket)
        .groupby(["_bucket", "route_id"], sort=True)[metric]
        .mean()
        .rename("value")
        .reset_index()
    )
    buckets = sorted(grouped["_bucket"].unique())
    if not buckets:
        raise ValueError("No time buckets available for the selected period")
    if len(buckets) > max_frames:
        raise ValueError(
            f"{len(buckets)} frames exceeds max_frames={max_frames}. "
            "Use a coarser interval or a shorter date period."
        )

    labels = [pd.Timestamp(b).strftime("%Y-%m-%d %H:%M") for b in buckets]
    wide = grouped.pivot(index="route_id", columns="_bucket", values="value")
    wide = wide.reindex(columns=buckets)
    wide.columns = labels
    return labels, wide
