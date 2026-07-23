"""Normalize Toronto festivals/events historical XML into interim Parquet."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.processing.paths import DEFAULT_INTERIM_DIR, DEFAULT_RAW_DIR, LOCAL_TZ

DEFAULT_XML_NAME = (
    "festivals-and-events-historical-xml-feed-jan-2014-dec-2016.xml"
)


def _entry_value(entrydata: ET.Element) -> str | None:
    """Extract the first text payload from a Domino ``entrydata`` node."""
    texts = [text.strip() for text in entrydata.itertext() if text and text.strip()]
    if not texts:
        return None
    return " | ".join(texts)


def _parse_event_datetime(date_text: str | None, time_text: str | None) -> pd.Timestamp | pd.NaT:
    """Parse organizer date/time strings into a timezone-aware timestamp."""
    if not date_text:
        return pd.NaT
    date_text = date_text.strip()
    time_text = (time_text or "").strip()
    candidates = []
    if time_text:
        candidates.append(f"{date_text} {time_text}")
    candidates.append(date_text)
    for raw in candidates:
        for fmt in (
            "%b %d, %Y %I:%M %p",
            "%B %d, %Y %I:%M %p",
            "%b %d, %Y",
            "%B %d, %Y",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(raw, fmt)
                return pd.Timestamp(dt, tz=LOCAL_TZ)
            except ValueError:
                continue
    return pd.NaT


def _to_float(value: str | None) -> float | None:
    """Parse a float from a possibly messy XML text field."""
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def parse_festivals_xml(path: Path) -> pd.DataFrame:
    """Parse the historical festivals XML export into a tidy events table.

    Args:
        path: Path to the Domino ``viewentries`` XML file.

    Returns:
        DataFrame with event identity, schedule, location, and flags.
    """
    root = ET.parse(path).getroot()
    rows: list[dict] = []
    for index, entry in enumerate(root.findall("viewentry")):
        fields = {
            ed.attrib.get("name"): _entry_value(ed)
            for ed in entry.findall("entrydata")
            if ed.attrib.get("name")
        }
        start_local = _parse_event_datetime(
            fields.get("DateBeginShow"), fields.get("TimeBegin")
        )
        end_local = _parse_event_datetime(
            fields.get("DateEndShow"), fields.get("TimeEnd")
        )
        if pd.isna(end_local) and not pd.isna(start_local):
            end_local = start_local

        road_close_raw = (fields.get("RoadClose") or "").strip().lower()
        rows.append(
            {
                "event_id": entry.attrib.get("unid") or f"row-{index}",
                "name": fields.get("EventName"),
                "category": fields.get("CategoryList"),
                "area": fields.get("Area"),
                "location": fields.get("Location"),
                "address": fields.get("Address"),
                "intersection": fields.get("Intersection"),
                "lat": _to_float(fields.get("txtLat")),
                "lon": _to_float(fields.get("txtLong")),
                "start_local": start_local,
                "end_local": end_local,
                "road_close": road_close_raw not in {"", "none", "null"},
                "admission": fields.get("Admission"),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["name"] = table["name"].astype("string")
    table["road_close"] = table["road_close"].astype(bool)
    return table.reset_index(drop=True)


def normalize_events(
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    xml_name: str = DEFAULT_XML_NAME,
) -> Path:
    """Normalize festivals XML into ``events.parquet``.

    Args:
        raw_dir: Raw data root.
        interim_dir: Interim output directory.
        xml_name: Historical XML filename under ``festivals_events/``.

    Returns:
        Path to written Parquet file.
    """
    xml_path = raw_dir / "festivals_events" / xml_name
    if not xml_path.exists():
        raise FileNotFoundError(f"Festivals XML not found: {xml_path}")
    table = parse_festivals_xml(xml_path)
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "events.parquet"
    table.to_parquet(out_path, index=False)
    return out_path
