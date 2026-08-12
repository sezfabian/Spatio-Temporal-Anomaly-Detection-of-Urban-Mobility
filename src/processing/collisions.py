"""Normalize Toronto Police traffic-collision CSV into interim Parquet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from src.processing.paths import DEFAULT_INTERIM_DIR, DEFAULT_RAW_DIR, LOCAL_TZ
from src.processing.routes import DEFAULT_ROUTES_ZIP, load_route_geometries

DEFAULT_CSV_NAME = "Traffic Collisions - 4326.csv"
DEFAULT_YEARS = (2014, 2015, 2016, 2017)
# Discard collisions farther than this from the nearest Bluetooth corridor.
DEFAULT_MAX_MATCH_DISTANCE_M = 150.0
# UTM zone 17N — appropriate projected meters for Toronto.
_PROJECTED_CRS = "EPSG:32617"

_YES_NO_COLUMNS = (
    "INJURY_COLLISIONS",
    "FTR_COLLISIONS",
    "PD_COLLISIONS",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "PASSENGER",
    "BICYCLE",
    "PEDESTRIAN",
)


def _yes_no_to_bool(series: pd.Series) -> pd.Series:
    """Map YES/NO (and common variants) to nullable boolean."""
    normalized = series.astype("string").str.strip().str.upper()
    mapped = normalized.map({"YES": True, "NO": False, "Y": True, "N": False})
    return mapped.astype("boolean")


def _parse_occ_date(series: pd.Series) -> pd.Series:
    """Parse OCC_DATE as epoch milliseconds or ordinary datetime strings."""
    numeric = pd.to_numeric(series, errors="coerce")
    # Epoch ms values are ~1e12; ordinary year numbers are tiny by comparison.
    if numeric.notna().any() and float(numeric.dropna().median()) > 1e11:
        return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _project_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 lon/lat arrays to local UTM meters."""
    transformer = Transformer.from_crs("EPSG:4326", _PROJECTED_CRS, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return np.asarray(easting, dtype=np.float64), np.asarray(northing, dtype=np.float64)


def _point_to_polyline_distance_m(
    px: np.ndarray,
    py: np.ndarray,
    vertices_xy: np.ndarray,
) -> np.ndarray:
    """Minimum Euclidean distance from each point to a polyline (meters).

    Args:
        px, py: Point coordinates in a projected CRS, shape ``(n,)``.
        vertices_xy: Polyline vertices as ``(m, 2)`` in the same CRS.

    Returns:
        Array of shape ``(n,)`` with minimum distance to any segment.
    """
    if vertices_xy.shape[0] == 0:
        return np.full(px.shape[0], np.inf, dtype=np.float64)
    if vertices_xy.shape[0] == 1:
        return np.hypot(px - vertices_xy[0, 0], py - vertices_xy[0, 1])

    ax = vertices_xy[:-1, 0]
    ay = vertices_xy[:-1, 1]
    bx = vertices_xy[1:, 0]
    by = vertices_xy[1:, 1]
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby

    # Broadcast points against segments: (n, s)
    apx = px[:, None] - ax[None, :]
    apy = py[:, None] - ay[None, :]
    denom = np.maximum(ab2, 1e-12)
    t = np.clip((apx * abx[None, :] + apy * aby[None, :]) / denom[None, :], 0.0, 1.0)
    cx = ax[None, :] + t * abx[None, :]
    cy = ay[None, :] + t * aby[None, :]
    return np.min(np.hypot(px[:, None] - cx, py[:, None] - cy), axis=1)


def match_collisions_to_routes(
    collisions: pd.DataFrame,
    route_geometries: dict[str, list[tuple[float, float]]],
    *,
    max_distance_m: float = DEFAULT_MAX_MATCH_DISTANCE_M,
) -> pd.DataFrame:
    """Assign each collision to its nearest Bluetooth route within ``max_distance_m``.

    Collisions without coordinates, or farther than ``max_distance_m`` from every
    route polyline, are dropped.

    Args:
        collisions: Occurrence table with ``lat`` / ``lon``.
        route_geometries: ``route_id`` -> WGS84 ``(lon, lat)`` vertices.
        max_distance_m: Maximum snap distance in meters.

    Returns:
        Filtered table with added ``route_id`` and ``dist_to_route_m`` columns.
    """
    if collisions.empty:
        out = collisions.copy()
        out["route_id"] = pd.Series(dtype="string")
        out["dist_to_route_m"] = pd.Series(dtype="float64")
        return out

    located = collisions.loc[collisions["lat"].notna() & collisions["lon"].notna()].copy()
    if located.empty or not route_geometries:
        return located.iloc[0:0].assign(
            route_id=pd.Series(dtype="string"),
            dist_to_route_m=pd.Series(dtype="float64"),
        )

    lon = located["lon"].to_numpy(dtype=np.float64)
    lat = located["lat"].to_numpy(dtype=np.float64)
    px, py = _project_lonlat(lon, lat)

    n = len(located)
    best_dist = np.full(n, np.inf, dtype=np.float64)
    best_route = np.empty(n, dtype=object)

    # Stable route order so equal distances resolve deterministically.
    for route_id in sorted(route_geometries):
        vertices = route_geometries[route_id]
        if not vertices:
            continue
        lon_v = np.asarray([p[0] for p in vertices], dtype=np.float64)
        lat_v = np.asarray([p[1] for p in vertices], dtype=np.float64)
        vx, vy = _project_lonlat(lon_v, lat_v)
        vertices_xy = np.column_stack([vx, vy])
        dist = _point_to_polyline_distance_m(px, py, vertices_xy)
        better = dist < best_dist
        best_dist[better] = dist[better]
        best_route[better] = route_id

    keep = best_dist <= float(max_distance_m)
    matched = located.loc[keep].copy()
    matched["route_id"] = pd.Series(best_route[keep], index=matched.index, dtype="string")
    matched["dist_to_route_m"] = best_dist[keep]
    return matched.reset_index(drop=True)


def parse_collisions_csv(
    path: Path,
    *,
    years: tuple[int, ...] | list[int] | None = DEFAULT_YEARS,
) -> pd.DataFrame:
    """Parse the WGS84 traffic-collisions CSV into a tidy occurrence table.

    Rows without usable coordinates (missing or near ``(0, 0)``) are dropped.

    Args:
        path: Path to ``Traffic Collisions - 4326.csv``.
        years: Optional occurrence-year filter. ``None`` keeps all years.
            Defaults to the Bluetooth study window 2014–2017.

    Returns:
        DataFrame with local timestamps, coordinates, severity flags, and
        involvement indicators.
    """
    raw = pd.read_csv(path, low_memory=False)
    empty_cols = [
        "collision_id",
        "ts_local",
        "date",
        "year",
        "hour",
        "month",
        "dow",
        "division",
        "hood_id",
        "neighbourhood",
        "lat",
        "lon",
        "fatalities",
        "is_injury",
        "is_ftr",
        "is_pd",
        "involves_automobile",
        "involves_motorcycle",
        "involves_passenger",
        "involves_bicycle",
        "involves_pedestrian",
    ]
    if raw.empty:
        return pd.DataFrame(columns=empty_cols)

    year = pd.to_numeric(raw["OCC_YEAR"], errors="coerce").astype("Int64")
    if years is not None:
        keep = year.isin(list(years))
        raw = raw.loc[keep].copy()
        year = year.loc[keep]

    occ_utc = _parse_occ_date(raw["OCC_DATE"])
    hour = pd.to_numeric(raw["OCC_HOUR"], errors="coerce").fillna(0).astype("int16")
    # OCC_DATE is midnight Eastern expressed as UTC epoch; attach OCC_HOUR in local TZ.
    date_local = occ_utc.dt.tz_convert(LOCAL_TZ).dt.normalize()
    ts_local = date_local + pd.to_timedelta(hour, unit="h")

    lat = pd.to_numeric(raw["LAT_WGS84"], errors="coerce")
    lon = pd.to_numeric(raw["LONG_WGS84"], errors="coerce")
    # NSA / suppressed locations are published near (0, 0).
    missing_coords = lat.isna() | lon.isna() | ((lat.abs() < 1e-6) & (lon.abs() < 1e-6))
    lat = lat.mask(missing_coords)
    lon = lon.mask(missing_coords)

    flags = {col: _yes_no_to_bool(raw[col]) for col in _YES_NO_COLUMNS if col in raw.columns}

    collision_id = (
        raw["_id"]
        if "_id" in raw.columns
        else pd.Series(range(1, len(raw) + 1), index=raw.index)
    )

    table = pd.DataFrame(
        {
            "collision_id": pd.to_numeric(collision_id, errors="coerce").astype("Int64"),
            "ts_local": ts_local,
            "date": ts_local.dt.date,
            "year": year.astype("Int64"),
            "hour": hour,
            "month": raw["OCC_MONTH"].astype("string") if "OCC_MONTH" in raw.columns else pd.NA,
            "dow": raw["OCC_DOW"].astype("string") if "OCC_DOW" in raw.columns else pd.NA,
            "division": raw["DIVISION"].astype("string") if "DIVISION" in raw.columns else pd.NA,
            "hood_id": raw["HOOD_158"].astype("string") if "HOOD_158" in raw.columns else pd.NA,
            "neighbourhood": (
                raw["NEIGHBOURHOOD_158"].astype("string")
                if "NEIGHBOURHOOD_158" in raw.columns
                else pd.NA
            ),
            "lat": lat.astype("float64"),
            "lon": lon.astype("float64"),
            "fatalities": pd.to_numeric(raw.get("FATALITIES"), errors="coerce").astype("Int64"),
            "is_injury": flags.get("INJURY_COLLISIONS"),
            "is_ftr": flags.get("FTR_COLLISIONS"),
            "is_pd": flags.get("PD_COLLISIONS"),
            "involves_automobile": flags.get("AUTOMOBILE"),
            "involves_motorcycle": flags.get("MOTORCYCLE"),
            "involves_passenger": flags.get("PASSENGER"),
            "involves_bicycle": flags.get("BICYCLE"),
            "involves_pedestrian": flags.get("PEDESTRIAN"),
        }
    )
    table = table.loc[table["lat"].notna() & table["lon"].notna()].copy()
    return table.reset_index(drop=True)


def normalize_collisions(
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    *,
    csv_name: str = DEFAULT_CSV_NAME,
    years: tuple[int, ...] | list[int] | None = DEFAULT_YEARS,
    routes_zip: Path | None = None,
    max_distance_m: float = DEFAULT_MAX_MATCH_DISTANCE_M,
) -> tuple[Path, dict]:
    """Normalize collision CSV, snap to Bluetooth routes, write ``collisions.parquet``.

    This is the processing-pipeline step for collisions. Route matching is
    required here: the interim table only contains points snapped to a
    Bluetooth corridor within ``max_distance_m``.

    Pipeline:
      1. Parse + keep study years.
      2. Drop rows without coordinates.
      3. Snap each remaining point to the nearest route polyline.
      4. Discard points farther than ``max_distance_m`` from the network.

    Args:
        raw_dir: Raw data root.
        interim_dir: Interim output directory.
        csv_name: Filename under ``traffic_collisions/``.
        years: Optional occurrence-year filter (default 2014–2017).
        routes_zip: Bluetooth routes shapefile ZIP. Defaults to the standard
            path under ``raw_dir``.
        max_distance_m: Maximum distance (m) to accept a route match.

    Returns:
        Tuple of ``(parquet_path, qa_summary_dict)``.
    """
    csv_path = raw_dir / "traffic_collisions" / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Collisions CSV not found: {csv_path}")

    zip_path = routes_zip or (raw_dir / "travel_times_bluetooth" / DEFAULT_ROUTES_ZIP.name)
    if not zip_path.exists():
        raise FileNotFoundError(f"Routes ZIP not found: {zip_path}")

    table = parse_collisions_csv(csv_path, years=years)
    geometries = load_route_geometries(zip_path)
    matched = match_collisions_to_routes(
        table,
        geometries,
        max_distance_m=max_distance_m,
    )

    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "collisions.parquet"
    matched.to_parquet(out_path, index=False)

    qa = {
        "n_with_coords": int(len(table)),
        "n_route_matched": int(len(matched)),
        "n_discarded_far_from_network": int(len(table) - len(matched)),
        "match_rate": float(len(matched) / len(table)) if len(table) else 0.0,
        "n_routes_matched": int(matched["route_id"].nunique()) if len(matched) else 0,
        "max_distance_m": float(max_distance_m),
        "dist_to_route_m_p50": (
            float(matched["dist_to_route_m"].median()) if len(matched) else None
        ),
        "dist_to_route_m_p90": (
            float(matched["dist_to_route_m"].quantile(0.9)) if len(matched) else None
        ),
    }
    qa_path = interim_dir / "collisions_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return out_path, qa
