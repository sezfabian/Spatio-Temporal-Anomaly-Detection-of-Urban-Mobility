"""Unit tests for processing normalizers using tiny synthetic fixtures."""

from __future__ import annotations

import struct
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.processing.civic import normalize_civic_days
from src.processing.collisions import (
    match_collisions_to_routes,
    normalize_collisions,
    parse_collisions_csv,
)
from src.processing.events import normalize_events, parse_festivals_xml
from src.processing.clean_panel import clean_panel_for_v1, columns_to_drop
from src.processing.panel import (
    build_collision_route_hour_summary,
    build_event_day_summary,
    build_route_time_panel,
    floor_to_local_hour,
)
from src.processing.routes import normalize_routes, read_routes_zip
from src.processing.travel_times import normalize_travel_times, read_travel_time_zip
from src.processing.weather import normalize_weather, read_weather_csv


def _write_travel_zip(path: Path) -> None:
    csv = (
        "resultId,timeInSeconds,count,updated\n"
        "J_I,56,16,2017-01-01T00:05:00-05\n"
        "J_I,60,9,2017-01-01T01:10:00-05\n"
        "A_B,0,1,2017-01-01T00:05:00-05\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("travel-time-2017.csv", csv)


def _write_dbf(path: Path) -> None:
    fields = [
        (b"resultId", b"C", 8, 0),
        (b"normalDriv", b"N", 5, 0),
        (b"length_m", b"N", 8, 1),
    ]
    header_len = 32 + 32 * len(fields) + 1
    record_len = 1 + sum(field[2] for field in fields)
    header = bytearray(32)
    header[0] = 0x03
    struct.pack_into("<I", header, 4, 1)
    struct.pack_into("<H", header, 8, header_len)
    struct.pack_into("<H", header, 10, record_len)
    chunks = [bytes(header)]
    for name, ftype, size, decimal in fields:
        field = bytearray(32)
        field[:11] = name.ljust(11, b"\x00")
        field[11:12] = ftype
        field[16] = size
        field[17] = decimal
        chunks.append(bytes(field))
    chunks.append(b"\x0d")
    record = bytearray(record_len)
    record[0:1] = b" "
    pos = 1
    values = [b"J_I", b"90", b"250.5"]
    for (_name, _t, size, _d), value in zip(fields, values):
        record[pos : pos + size] = value.ljust(size, b" ")[:size]
        pos += size
    path.write_bytes(b"".join(chunks) + bytes(record))


def test_read_travel_time_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "travel-time-2017.zip"
    _write_travel_zip(zip_path)
    frame = read_travel_time_zip(zip_path)
    assert list(frame.columns) == [
        "route_id",
        "travel_time_s",
        "sample_count",
        "ts_local",
        "year",
    ]
    assert len(frame) == 2
    assert set(frame["route_id"]) == {"J_I"}
    assert int(frame["year"].iloc[0]) == 2017
    assert str(frame["ts_local"].dt.tz) == "America/Toronto"


def test_read_travel_time_zip_mixed_offsets(tmp_path: Path) -> None:
    """Winter (-05) and summer (-04) offsets must parse into one tz-aware series."""
    zip_path = tmp_path / "travel-time-2016.zip"
    csv = (
        "resultId,timeInSeconds,count,updated\n"
        "J_I,56,16,2016-01-15T00:05:00-05\n"
        "J_I,60,9,2016-07-15T00:05:00-04\n"
    )
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("travel-time-2016.csv", csv)
    frame = read_travel_time_zip(zip_path)
    assert len(frame) == 2
    assert str(frame["ts_local"].dtype).startswith("datetime64")
    assert str(frame["ts_local"].dt.tz) == "America/Toronto"


def test_normalize_travel_times(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "travel_times_bluetooth"
    raw.mkdir(parents=True)
    _write_travel_zip(raw / "travel-time-2017.zip")
    out = normalize_travel_times(tmp_path / "raw", tmp_path / "interim", years=[2017])
    assert out.exists()
    assert len(pd.read_parquet(out)) == 2


def test_read_weather_csv(tmp_path: Path) -> None:
    path = tmp_path / "hourly_2014_01.csv"
    path.write_text(
        "Date/Time (LST),Temp (°C),Rel Hum (%),Precip. Amount (mm),Wind Spd (km/h),Stn Press (kPa)\n"
        "2014-01-01 00:00,-5.0,80,0.0,10,100.1\n"
        "2014-01-01 01:00,-5.5,81,,12,100.0\n",
        encoding="utf-8",
    )
    frame = read_weather_csv(path)
    assert "temp_c" in frame.columns
    assert frame["ts_local"].dt.tz is not None
    assert len(frame) == 2


def test_normalize_weather(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "weather_toronto_city"
    raw.mkdir(parents=True)
    (raw / "hourly_2014_01.csv").write_text(
        "Date/Time (LST),Temp (°C)\n2014-01-01 00:00,-5.0\n",
        encoding="utf-8",
    )
    out = normalize_weather(tmp_path / "raw", tmp_path / "interim")
    assert pd.read_parquet(out)["temp_c"].iloc[0] == -5.0


def test_normalize_civic_days(tmp_path: Path) -> None:
    calendar = {
        "timezone": "America/Toronto",
        "start_year": 2014,
        "end_year": 2014,
        "events": [
            {
                "id": "2014_christmas_day",
                "name": "Christmas Day",
                "kind": "public_holiday",
                "start_date": "2014-12-25",
                "end_date": "2014-12-25",
            }
        ],
    }
    path = tmp_path / "cal.yaml"
    path.write_text(yaml.safe_dump(calendar), encoding="utf-8")
    out = normalize_civic_days(
        path,
        tmp_path / "interim",
        start_date=date(2014, 12, 24),
        end_date=date(2014, 12, 26),
    )
    table = pd.read_parquet(out)
    assert len(table) == 3
    xmas = table.loc[table["date"].astype(str) == "2014-12-25"].iloc[0]
    assert bool(xmas["is_public_holiday"]) is True


def test_parse_festivals_xml(tmp_path: Path) -> None:
    xml = """<?xml version='1.0' encoding='UTF-8'?>
<viewentries toplevelentries="1">
  <viewentry unid="ABC123">
    <entrydata name="EventName"><text>Test Parade</text></entrydata>
    <entrydata name="Area"><text>Downtown</text></entrydata>
    <entrydata name="CategoryList"><text>Music</text></entrydata>
    <entrydata name="DateBeginShow"><text>Nov 16, 2014</text></entrydata>
    <entrydata name="TimeBegin"><text>12:00 PM</text></entrydata>
    <entrydata name="DateEndShow"><text>Nov 16, 2014</text></entrydata>
    <entrydata name="TimeEnd"><text>3:00 PM</text></entrydata>
    <entrydata name="Address"><text>1 Front St</text></entrydata>
    <entrydata name="txtLat"><text>43.65</text></entrydata>
    <entrydata name="txtLong"><text>-79.38</text></entrydata>
    <entrydata name="RoadClose"><text>Road closure</text></entrydata>
  </viewentry>
</viewentries>
"""
    path = tmp_path / "events.xml"
    path.write_text(xml, encoding="utf-8")
    table = parse_festivals_xml(path)
    assert len(table) == 1
    assert table.iloc[0]["name"] == "Test Parade"
    assert bool(table.iloc[0]["road_close"]) is True
    assert table.iloc[0]["lat"] == pytest.approx(43.65)


def test_normalize_events(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "festivals_events"
    raw.mkdir(parents=True)
    xml_name = "festivals-and-events-historical-xml-feed-jan-2014-dec-2016.xml"
    (raw / xml_name).write_text(
        """<?xml version='1.0'?><viewentries>
        <viewentry unid="1">
          <entrydata name="EventName"><text>A</text></entrydata>
          <entrydata name="DateBeginShow"><text>Jan 1, 2014</text></entrydata>
        </viewentry></viewentries>""",
        encoding="utf-8",
    )
    out = normalize_events(tmp_path / "raw", tmp_path / "interim")
    assert len(pd.read_parquet(out)) == 1


def test_read_routes_zip(tmp_path: Path) -> None:
    dbf = tmp_path / "routes.dbf"
    _write_dbf(dbf)
    zip_path = tmp_path / "bluetooth-routes-wgs84.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(dbf, arcname="bluetooth_routes_wgs84.dbf")
    frame = read_routes_zip(zip_path)
    assert frame.iloc[0]["route_id"] == "J_I"
    assert frame.iloc[0]["free_flow_s"] == 90
    assert frame.iloc[0]["length_m"] == pytest.approx(250.5)


def test_normalize_routes(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "travel_times_bluetooth"
    raw.mkdir(parents=True)
    dbf = tmp_path / "routes.dbf"
    _write_dbf(dbf)
    with zipfile.ZipFile(raw / "bluetooth-routes-wgs84.zip", "w") as archive:
        archive.write(dbf, arcname="bluetooth_routes_wgs84.dbf")
    out = normalize_routes(tmp_path / "raw", tmp_path / "interim")
    assert pd.read_parquet(out).iloc[0]["route_id"] == "J_I"


def test_floor_to_local_hour_handles_dst_fallback() -> None:
    """Fall-back 01:xx local must floor without AmbiguousTimeError."""
    # Build via UTC so construction itself is unambiguous around DST end.
    ts = pd.Series(
        [
            pd.Timestamp("2014-11-02 05:05:00", tz="UTC").tz_convert("America/Toronto"),
            pd.Timestamp("2014-11-02 06:35:00", tz="UTC").tz_convert("America/Toronto"),
        ]
    )
    floored = floor_to_local_hour(ts)
    assert str(floored.dt.tz) == "America/Toronto"
    assert floored.dt.minute.eq(0).all()


def test_build_event_day_summary() -> None:
    events = pd.DataFrame(
        [
            {
                "start_local": pd.Timestamp("2014-11-16 12:00", tz="America/Toronto"),
                "end_local": pd.Timestamp("2014-11-16 15:00", tz="America/Toronto"),
                "road_close": True,
                "lat": 43.65,
                "lon": -79.38,
            }
        ]
    )
    summary = build_event_day_summary(events)
    assert summary.iloc[0]["n_events_active"] == 1
    assert summary.iloc[0]["n_events_road_close"] == 1


def test_parse_collisions_csv(tmp_path: Path) -> None:
    csv = tmp_path / "collisions.csv"
    csv.write_text(
        "_id,OCC_DATE,OCC_MONTH,OCC_DOW,OCC_YEAR,OCC_HOUR,DIVISION,"
        "FATALITIES,INJURY_COLLISIONS,FTR_COLLISIONS,PD_COLLISIONS,"
        "HOOD_158,NEIGHBOURHOOD_158,LONG_WGS84,LAT_WGS84,"
        "AUTOMOBILE,MOTORCYCLE,PASSENGER,BICYCLE,PEDESTRIAN\n"
        "1,1388552400000,January,Wednesday,2014,15,D43,,"
        "NO,NO,YES,133,Centennial Scarborough (133),"
        "-79.17246,43.78218,YES,NO,NO,NO,NO\n"
        "2,1388552400000,January,Wednesday,2014,11,NSA,,"
        "YES,NO,NO,NSA,NSA,0,0,YES,NO,NO,NO,NO\n"
        "3,1514782800000,January,Monday,2018,9,D51,1,"
        "YES,NO,NO,075,Church-Yonge Corridor (75),"
        "-79.378,43.661,YES,NO,NO,NO,YES\n",
        encoding="utf-8",
    )
    frame = parse_collisions_csv(csv, years=(2014, 2015, 2016, 2017))
    # Year filter drops 2018; missing/(0,0) coords are dropped.
    assert len(frame) == 1
    assert frame.iloc[0]["hour"] == 15
    assert bool(frame.iloc[0]["is_pd"]) is True
    assert pd.notna(frame.iloc[0]["lat"])
    assert str(frame["ts_local"].dt.tz) == "America/Toronto"
    assert frame.iloc[0]["ts_local"].date().isoformat() == "2014-01-01"


def test_match_collisions_to_routes_keeps_nearby_only() -> None:
    # Short east-west corridor near downtown Toronto.
    geometries = {
        "A_B": [(-79.4000, 43.6500), (-79.3900, 43.6500)],
        "B_A": [(-79.3900, 43.6500), (-79.4000, 43.6500)],
    }
    collisions = pd.DataFrame(
        {
            "collision_id": [1, 2, 3],
            "lat": [43.6501, 43.70, None],
            "lon": [-79.3950, -79.3950, None],
            "ts_local": pd.to_datetime(
                ["2017-01-01 00:00:00"] * 3
            ).tz_localize("America/Toronto"),
        }
    )
    matched = match_collisions_to_routes(collisions, geometries, max_distance_m=150.0)
    assert len(matched) == 1
    assert matched.iloc[0]["collision_id"] == 1
    assert matched.iloc[0]["route_id"] in {"A_B", "B_A"}
    assert float(matched.iloc[0]["dist_to_route_m"]) < 50.0


def test_normalize_collisions(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "traffic_collisions"
    routes = tmp_path / "raw" / "travel_times_bluetooth"
    raw.mkdir(parents=True)
    routes.mkdir(parents=True)

    # Reuse the real routes ZIP when available; otherwise skip spatial snap via tiny zip.
    from shutil import copyfile

    repo_zip = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "travel_times_bluetooth"
        / "bluetooth-routes-wgs84.zip"
    )
    if not repo_zip.exists():
        pytest.skip("bluetooth-routes-wgs84.zip not available")
    copyfile(repo_zip, routes / "bluetooth-routes-wgs84.zip")

    csv = raw / "Traffic Collisions - 4326.csv"
    # Point on / near corridor B_A first vertex (~-79.4935, 43.6247).
    csv.write_text(
        "_id,OCC_DATE,OCC_MONTH,OCC_DOW,OCC_YEAR,OCC_HOUR,DIVISION,"
        "FATALITIES,INJURY_COLLISIONS,FTR_COLLISIONS,PD_COLLISIONS,"
        "HOOD_158,NEIGHBOURHOOD_158,LONG_WGS84,LAT_WGS84,"
        "AUTOMOBILE,MOTORCYCLE,PASSENGER,BICYCLE,PEDESTRIAN\n"
        "10,1483246800000,January,Sunday,2017,0,D52,,"
        "NO,NO,YES,078,Kensington-Chinatown (78),"
        "-79.4935,43.6247,YES,NO,NO,NO,NO\n"
        "11,1483246800000,January,Sunday,2017,1,NSA,,"
        "NO,NO,YES,NSA,NSA,0,0,YES,NO,NO,NO,NO\n",
        encoding="utf-8",
    )
    out, qa = normalize_collisions(tmp_path / "raw", tmp_path / "interim")
    table = pd.read_parquet(out)
    assert len(table) == 1
    assert int(table.iloc[0]["year"]) == 2017
    assert pd.notna(table.iloc[0]["route_id"])
    assert float(table.iloc[0]["dist_to_route_m"]) <= 150.0
    assert qa["n_route_matched"] == 1
    assert qa["n_with_coords"] == 1
    assert (tmp_path / "interim" / "collisions_qa.json").exists()


def test_build_collision_route_hour_summary() -> None:
    collisions = pd.DataFrame(
        {
            "collision_id": [1, 2, 3],
            "route_id": ["J_I", "J_I", "A_B"],
            "ts_local": pd.to_datetime(
                [
                    "2017-01-01 00:15:00",
                    "2017-01-01 00:45:00",
                    "2017-01-01 00:20:00",
                ]
            ).tz_localize("America/Toronto"),
            "is_injury": [True, False, False],
            "fatalities": [0, 1, 0],
            "dist_to_route_m": [12.0, 40.0, 8.0],
        }
    )
    summary = build_collision_route_hour_summary(collisions)
    ji = summary.loc[summary["route_id"] == "J_I"].iloc[0]
    assert int(ji["n_collisions"]) == 2
    assert int(ji["n_injury_collisions"]) == 1
    assert int(ji["n_fatal_collisions"]) == 1
    assert float(ji["min_dist_to_route_m"]) == pytest.approx(12.0)


def test_build_route_time_panel(tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir()

    travel = pd.DataFrame(
        {
            "route_id": ["J_I", "J_I"],
            "travel_time_s": [56.0, 60.0],
            "sample_count": [16, 9],
            "ts_local": pd.to_datetime(
                ["2017-01-01 00:05:00-05:00", "2017-01-01 01:10:00-05:00"]
            ),
            "year": [2017, 2017],
        }
    )
    routes = pd.DataFrame(
        {"route_id": ["J_I"], "free_flow_s": [50.0], "length_m": [250.5]}
    )
    weather = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(
                ["2017-01-01 00:00:00", "2017-01-01 01:00:00"]
            ).tz_localize("America/Toronto"),
            "temp_c": [-5.0, -4.0],
        }
    )
    civic = pd.DataFrame(
        {
            "date": [date(2017, 1, 1)],
            "is_public_holiday": [True],
            "is_civic_holiday": [False],
            "is_holiday": [True],
            "is_school_break": [False],
            "is_mega_event": [False],
            "is_parade": [False],
            "is_shopping_peak": [False],
            "n_civic_events": [1],
            "holiday_names": ["New Year's Day"],
            "mega_event_names": [""],
            "event_ids": ["2017_new_years_day"],
            "event_kinds": ["public_holiday"],
        }
    )
    events = pd.DataFrame(
        {
            "start_local": pd.to_datetime(["2017-01-01 00:00:00"]).tz_localize(
                "America/Toronto"
            ),
            "end_local": pd.to_datetime(["2017-01-01 23:00:00"]).tz_localize(
                "America/Toronto"
            ),
            "road_close": [False],
            "lat": [43.65],
            "lon": [-79.38],
        }
    )
    collisions = pd.DataFrame(
        {
            "collision_id": [1],
            "route_id": ["J_I"],
            "ts_local": pd.to_datetime(["2017-01-01 00:20:00"]).tz_localize(
                "America/Toronto"
            ),
            "lat": [43.65],
            "lon": [-79.38],
            "is_injury": [True],
            "fatalities": [0],
            "dist_to_route_m": [25.0],
        }
    )

    travel.to_parquet(interim / "travel_times.parquet", index=False)
    routes.to_parquet(interim / "routes.parquet", index=False)
    weather.to_parquet(interim / "weather_hourly.parquet", index=False)
    civic.to_parquet(interim / "civic_days.parquet", index=False)
    events.to_parquet(interim / "events.parquet", index=False)
    collisions.to_parquet(interim / "collisions.parquet", index=False)

    out, qa = build_route_time_panel(interim, processed, years=[2017])
    panel = pd.read_parquet(out)
    assert len(panel) == 2
    assert "wx_temp_c" in panel.columns
    assert "delay_s" in panel.columns
    assert "n_collisions" in panel.columns
    assert int(panel.loc[panel["hour"] == 0, "n_collisions"].iloc[0]) == 1
    assert float(panel.loc[panel["hour"] == 0, "min_dist_to_route_m"].iloc[0]) == 25.0
    assert int(panel.loc[panel["hour"] == 1, "n_collisions"].iloc[0]) == 0
    assert qa["n_rows"] == 2
    assert qa["weather_match_rate"] == 1.0
    assert qa["collision_route_hours_with_exposure"] == 1


def test_columns_to_drop_v1() -> None:
    cols = [
        "route_id",
        "wx_temp_c",
        "wx_wind_spd_kmh",
        "wx_wind_dir_10s",
        "wx_visibility_km",
        "wx_weather_desc",
        "wx_wind_chill",
        "wx_humidex",
        "holiday_names",
        "mega_event_names",
        "event_ids",
        "event_kinds",
        "is_weekend",
        "delay_s",
    ]
    dropped = columns_to_drop(cols)
    assert "wx_wind_spd_kmh" in dropped
    assert "wx_humidex" in dropped
    assert "mega_event_names" in dropped
    assert "holiday_names" in dropped
    assert "event_ids" not in dropped
    assert "event_kinds" not in dropped
    assert "is_weekend" not in dropped
    assert "wx_temp_c" not in dropped
    assert "delay_s" not in dropped


def test_clean_panel_for_v1(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "route_id": ["A_B", "B_C"],
            "ts_local": pd.to_datetime(
                ["2016-06-18 10:00:00-04:00", "2016-06-19 10:00:00-04:00"]  # Sat, Sun
            ),
            "delay_s": [1.0, 2.0],
            "is_weekend": [False, False],  # wrong on purpose; cleaner should fix
            "wx_temp_c": [10.0, 11.0],
            "wx_wind_spd_kmh": [None, None],
            "wx_visibility_km": [None, None],
            "wx_weather_desc": [None, None],
            "wx_wind_chill": [None, None],
            "wx_humidex": [None, None],
            "holiday_names": ["Canada Day", None],
            "mega_event_names": ["Pride", ""],
            "event_ids": ["e1", None],
            "event_kinds": ["festival", None],
        }
    )
    src = tmp_path / "route_time_panel.parquet"
    panel.to_parquet(src, index=False)
    out, qa = clean_panel_for_v1(
        src,
        out_path=tmp_path / "route_time_panel_v1.parquet",
        qa_path=tmp_path / "route_time_panel_v1_qa.json",
    )
    cleaned = pd.read_parquet(out)
    assert cleaned["is_weekend"].tolist() == [True, True]
    assert "event_ids" in cleaned.columns
    assert "event_kinds" in cleaned.columns
    assert "holiday_names" not in cleaned.columns
    assert "mega_event_names" not in cleaned.columns
    for col in (
        "wx_wind_spd_kmh",
        "wx_visibility_km",
        "wx_weather_desc",
        "wx_wind_chill",
        "wx_humidex",
    ):
        assert col not in cleaned.columns
    assert qa["is_weekend_rate"] == 1.0
    assert qa["event_ids_nonempty_rate"] == 0.5
