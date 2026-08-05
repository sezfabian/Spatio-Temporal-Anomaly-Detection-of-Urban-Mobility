"""Drop sparse weather / free-text name columns for ST-GAE v1 training panels."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.processing.paths import DEFAULT_PROCESSED_DIR

# Exact column names removed for v1 training readiness.
# Keep is_weekend (calendar-derived) and event_ids / event_kinds / counts;
# drop free-text name fields (holiday_names, mega_event_names).
DROP_COLUMNS_EXACT: frozenset[str] = frozenset(
    {
        "wx_visibility_km",
        "wx_weather_desc",
        "wx_wind_chill",
        "wx_humidex",
        "holiday_names",
        "mega_event_names",
    }
)

# Prefixes dropped for v1 (covers wx_wind_dir_10s, wx_wind_spd_kmh, …).
DROP_COLUMN_PREFIXES: tuple[str, ...] = ("wx_wind_",)

DEFAULT_PANEL = DEFAULT_PROCESSED_DIR / "route_time_panel.parquet"
DEFAULT_V1_PANEL = DEFAULT_PROCESSED_DIR / "route_time_panel_v1.parquet"
DEFAULT_V1_QA = DEFAULT_PROCESSED_DIR / "route_time_panel_v1_qa.json"


def columns_to_drop(columns: list[str] | pd.Index) -> list[str]:
    """Return column names that should be dropped for the v1 training panel."""
    drop: list[str] = []
    for name in columns:
        if name in DROP_COLUMNS_EXACT:
            drop.append(str(name))
            continue
        if any(str(name).startswith(prefix) for prefix in DROP_COLUMN_PREFIXES):
            drop.append(str(name))
    return drop


def _ensure_calendar_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Populate weekend fields from the local calendar timestamp."""
    out = frame.copy()
    if "ts_local" not in out.columns:
        raise KeyError("panel must include ts_local to derive calendar fields")

    ts = pd.to_datetime(out["ts_local"], utc=False)
    out["dow"] = ts.dt.dayofweek.astype("int16")
    out["is_weekend"] = out["dow"].isin([5, 6]).astype("bool")
    return out


def clean_panel_for_v1(
    panel: pd.DataFrame | Path = DEFAULT_PANEL,
    *,
    out_path: Path = DEFAULT_V1_PANEL,
    qa_path: Path = DEFAULT_V1_QA,
) -> tuple[Path, dict]:
    """Write a v1 panel with sparse weather and free-text name columns removed.

    Re-derives ``is_weekend`` / ``dow`` from ``ts_local``. Drops
    ``holiday_names`` and ``mega_event_names``; retains ``event_ids``,
    ``event_kinds``, and numeric event / holiday flags.

    Args:
        panel: Full ``route_time_panel`` path or DataFrame.
        out_path: Destination Parquet for the cleaned panel.
        qa_path: Destination JSON summarizing dropped / retained columns.

    Returns:
        ``(out_path, qa_summary)``.
    """
    frame = pd.read_parquet(panel) if isinstance(panel, (str, Path)) else panel.copy()
    frame = _ensure_calendar_fields(frame)
    dropped = columns_to_drop(frame.columns)
    cleaned = frame.drop(columns=dropped, errors="ignore")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out_path, index=False)

    weekend_rate = float(cleaned["is_weekend"].mean()) if len(cleaned) else 0.0
    event_id_rate = (
        float((cleaned["event_ids"].fillna("").astype(str).str.len() > 0).mean())
        if "event_ids" in cleaned.columns and len(cleaned)
        else 0.0
    )
    qa = {
        "source_rows": int(len(frame)),
        "output_rows": int(len(cleaned)),
        "n_cols_before": int(frame.shape[1]),
        "n_cols_after": int(cleaned.shape[1]),
        "dropped_columns": dropped,
        "retained_columns": list(cleaned.columns),
        "is_weekend_rate": weekend_rate,
        "event_ids_nonempty_rate": event_id_rate,
    }
    Path(qa_path).write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return out_path, qa
