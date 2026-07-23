"""Build the joined route×time analysis panel for statistical modeling."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.processing.paths import DEFAULT_INTERIM_DIR, DEFAULT_PROCESSED_DIR, LOCAL_TZ


def _ensure_tz(series: pd.Series) -> pd.Series:
    """Ensure a datetime series is timezone-aware in America/Toronto."""
    if getattr(series.dt, "tz", None) is None:
        return series.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return series.dt.tz_convert(LOCAL_TZ)


def floor_to_local_hour(series: pd.Series) -> pd.Series:
    """Floor timestamps to the hour without DST ambiguous-time failures.

    Args:
        series: Timezone-aware datetime series (America/Toronto).

    Returns:
        Same timezone, floored to the hour. Flooring is done in UTC so
        fall-back transitions (e.g. 2014-11-02 01:00) do not raise.
    """
    return series.dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(LOCAL_TZ)


def build_event_day_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse events into per-date citywide exposure counts.

    Args:
        events: Normalized events table with ``start_local`` / ``end_local``.

    Returns:
        DataFrame keyed by ``date`` with active-event counts.
    """
    if events.empty:
        return pd.DataFrame(
            columns=["date", "n_events_active", "n_events_road_close", "n_events_with_coords"]
        )

    rows: list[dict] = []
    for _, event in events.iterrows():
        start = event.get("start_local")
        end = event.get("end_local")
        if pd.isna(start):
            continue
        if pd.isna(end):
            end = start
        start_day = pd.Timestamp(start).tz_convert(LOCAL_TZ).date()
        end_day = pd.Timestamp(end).tz_convert(LOCAL_TZ).date()
        day = pd.Timestamp(start_day)
        end_ts = pd.Timestamp(end_day)
        while day <= end_ts:
            rows.append(
                {
                    "date": day.date(),
                    "road_close": bool(event.get("road_close")),
                    "has_coords": pd.notna(event.get("lat")) and pd.notna(event.get("lon")),
                }
            )
            day = day + pd.Timedelta(1, unit="D")

    if not rows:
        return pd.DataFrame(
            columns=["date", "n_events_active", "n_events_road_close", "n_events_with_coords"]
        )

    exploded = pd.DataFrame(rows)
    summary = (
        exploded.groupby("date", as_index=False)
        .agg(
            n_events_active=("road_close", "size"),
            n_events_road_close=("road_close", "sum"),
            n_events_with_coords=("has_coords", "sum"),
        )
    )
    return summary


def build_route_time_panel(
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    years: list[int] | None = None,
) -> tuple[Path, dict]:
    """Join interim tables into ``route_time_panel.parquet``.

    Args:
        interim_dir: Directory containing interim Parquet tables.
        processed_dir: Output directory for the analysis panel.
        years: Optional travel-time year filter applied before joins.

    Returns:
        Tuple of ``(panel_path, qa_summary_dict)``.
    """
    travel = pd.read_parquet(interim_dir / "travel_times.parquet")
    routes = pd.read_parquet(interim_dir / "routes.parquet")
    weather = pd.read_parquet(interim_dir / "weather_hourly.parquet")
    civic = pd.read_parquet(interim_dir / "civic_days.parquet")
    events = pd.read_parquet(interim_dir / "events.parquet")

    if years is not None:
        travel = travel.loc[travel["year"].isin(years)].copy()

    travel["ts_local"] = _ensure_tz(pd.to_datetime(travel["ts_local"], utc=False))
    weather["ts_local"] = _ensure_tz(pd.to_datetime(weather["ts_local"], utc=False))

    panel = travel.merge(routes, on="route_id", how="left")
    panel["ts_hour"] = floor_to_local_hour(panel["ts_local"])
    panel = panel.merge(
        weather.add_prefix("wx_").rename(columns={"wx_ts_local": "ts_hour"}),
        on="ts_hour",
        how="left",
    )

    panel["date"] = panel["ts_local"].dt.date
    civic = civic.copy()
    civic["date"] = pd.to_datetime(civic["date"]).dt.date
    panel = panel.merge(civic, on="date", how="left")

    event_days = build_event_day_summary(events)
    panel = panel.merge(event_days, on="date", how="left")
    for col in ("n_events_active", "n_events_road_close", "n_events_with_coords"):
        panel[col] = panel[col].fillna(0).astype("int64")

    panel["hour"] = panel["ts_local"].dt.hour.astype("int16")
    panel["dow"] = panel["ts_local"].dt.dayofweek.astype("int16")
    panel["is_weekend"] = panel["dow"].isin([5, 6])
    if "free_flow_s" in panel.columns:
        panel["delay_s"] = panel["travel_time_s"] - panel["free_flow_s"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "route_time_panel.parquet"
    panel.to_parquet(out_path, index=False)

    qa = {
        "n_rows": int(len(panel)),
        "n_routes": int(panel["route_id"].nunique()),
        "year_min": int(panel["year"].min()) if len(panel) else None,
        "year_max": int(panel["year"].max()) if len(panel) else None,
        "weather_match_rate": float(panel["wx_temp_c"].notna().mean()) if "wx_temp_c" in panel else 0.0,
        "civic_match_rate": float(panel["is_holiday"].notna().mean()) if "is_holiday" in panel else 0.0,
        "routes_match_rate": float(panel["length_m"].notna().mean()) if "length_m" in panel else 0.0,
        "event_days_with_exposure": int((panel["n_events_active"] > 0).sum()),
        "null_travel_time_rate": float(panel["travel_time_s"].isna().mean()) if len(panel) else 0.0,
    }
    qa_path = processed_dir / "route_time_panel_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return out_path, qa
