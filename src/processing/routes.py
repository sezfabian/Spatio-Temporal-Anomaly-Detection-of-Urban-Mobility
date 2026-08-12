"""Normalize Bluetooth route attributes from the WGS84 shapefile ZIP."""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pandas as pd
import shapefile

from src.processing.paths import DEFAULT_INTERIM_DIR, DEFAULT_RAW_DIR

DEFAULT_ROUTES_ZIP = (
    DEFAULT_RAW_DIR / "travel_times_bluetooth" / "bluetooth-routes-wgs84.zip"
)


def load_route_geometries(routes_zip: Path) -> dict[str, list[tuple[float, float]]]:
    """Load WGS84 polyline coordinates keyed by ``route_id``.

    Args:
        routes_zip: Path to ``bluetooth-routes-wgs84.zip``.

    Returns:
        Mapping of ``route_id`` -> list of ``(lon, lat)`` vertices.
    """
    with zipfile.ZipFile(routes_zip) as archive:
        names = {
            Path(name).suffix.lower(): name
            for name in archive.namelist()
            if Path(name).suffix.lower() in {".shp", ".dbf", ".shx"}
            and "__macosx" not in name.lower()
            and not Path(name).name.startswith("._")
        }
        missing = {ext for ext in (".shp", ".dbf", ".shx") if ext not in names}
        if missing:
            raise FileNotFoundError(f"Routes ZIP missing members: {sorted(missing)}")
        shp = io.BytesIO(archive.read(names[".shp"]))
        dbf = io.BytesIO(archive.read(names[".dbf"]))
        shx = io.BytesIO(archive.read(names[".shx"]))

    reader = shapefile.Reader(shp=shp, dbf=dbf, shx=shx)
    field_names = [field[0] for field in reader.fields[1:]]
    if "resultId" not in field_names:
        raise KeyError("Shapefile DBF missing resultId field")
    id_idx = field_names.index("resultId")

    geometries: dict[str, list[tuple[float, float]]] = {}
    for shape_rec in reader.iterShapeRecords():
        route_id = str(shape_rec.record[id_idx]).strip()
        points = [(float(x), float(y)) for x, y in shape_rec.shape.points]
        if points:
            geometries[route_id] = points
    return geometries


def _read_dbf_records(data: bytes) -> pd.DataFrame:
    """Parse a dBase III/IV DBF byte payload into a DataFrame.

    Args:
        data: Raw ``.dbf`` file bytes.

    Returns:
        DataFrame of attribute rows (geometry is not included).
    """
    if len(data) < 32:
        raise ValueError("DBF too small")
    record_count = struct.unpack("<I", data[4:8])[0]
    header_length = struct.unpack("<H", data[8:10])[0]
    record_length = struct.unpack("<H", data[10:12])[0]

    fields: list[tuple[str, str, int, int]] = []
    offset = 32
    while offset + 32 <= header_length and data[offset] != 0x0D:
        name = data[offset : offset + 11].split(b"\x00", 1)[0].decode("ascii", "replace")
        ftype = chr(data[offset + 11])
        size = data[offset + 16]
        decimal = data[offset + 17]
        fields.append((name, ftype, size, decimal))
        offset += 32

    rows: list[dict] = []
    cursor = header_length
    for _ in range(record_count):
        record = data[cursor : cursor + record_length]
        cursor += record_length
        if not record or record[0:1] == b"*":
            continue
        pos = 1
        row: dict = {}
        for name, ftype, size, decimal in fields:
            raw = record[pos : pos + size]
            pos += size
            text = raw.decode("latin-1", errors="replace").strip()
            if ftype in {"N", "F"}:
                if text in {"", "."}:
                    row[name] = None
                else:
                    row[name] = float(text) if decimal or "." in text else int(float(text))
            else:
                row[name] = text or None
        rows.append(row)
    return pd.DataFrame(rows)


def read_routes_zip(path: Path) -> pd.DataFrame:
    """Read route attributes from ``bluetooth-routes-wgs84.zip``.

    Args:
        path: Path to the routes shapefile ZIP.

    Returns:
        DataFrame with ``route_id``, ``free_flow_s``, ``length_m``.
    """
    with zipfile.ZipFile(path) as archive:
        dbf_names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".dbf")
            and "__macosx" not in name.lower()
            and not Path(name).name.startswith("._")
        ]
        if not dbf_names:
            raise FileNotFoundError(f"No DBF member in {path}")
        data = archive.read(dbf_names[0])

    raw = _read_dbf_records(data)
    rename = {}
    if "resultId" in raw.columns:
        rename["resultId"] = "route_id"
    if "normalDriv" in raw.columns:
        rename["normalDriv"] = "free_flow_s"
    if "length_m" in raw.columns:
        rename["length_m"] = "length_m"
    out = raw.rename(columns=rename)
    keep = [col for col in ("route_id", "free_flow_s", "length_m") if col in out.columns]
    out = out.loc[:, keep].copy()
    out["route_id"] = out["route_id"].astype("string")
    for col in ("free_flow_s", "length_m"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)


def normalize_routes(
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
) -> Path:
    """Normalize Bluetooth route attributes to ``routes.parquet``.

    Args:
        raw_dir: Raw data root.
        interim_dir: Interim output directory.

    Returns:
        Path to written Parquet file.
    """
    zip_path = raw_dir / "travel_times_bluetooth" / DEFAULT_ROUTES_ZIP.name
    if not zip_path.exists():
        raise FileNotFoundError(f"Routes ZIP not found: {zip_path}")
    table = read_routes_zip(zip_path)
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "routes.parquet"
    table.to_parquet(out_path, index=False)
    return out_path
