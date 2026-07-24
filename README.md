# Spatio-Temporal Anomaly Detection of Urban Mobility

Research on spatio-temporal ML for urban mobility: detecting anomalies in Toronto Bluetooth travel-time data and learning early indicators of collisions and emerging network disruptions.

This repo currently covers **data ingestion** (raw downloads) and **processing** (tidy interim tables + a joined analysis panel).

## Pipeline overview

```text
configs/                     # dataset + civic calendar definitions
        │
        ▼
ingestion CLIs  ──────────►  data/raw/          (downloaded archives/CSVs)
        │
        ▼
processing CLIs ──────────►  data/interim/      (tidy Parquet tables)
        │
        ▼
                             data/processed/    (joined route×time panel)
```

| Stage | Purpose | Typical outputs |
|---|---|---|
| **Ingestion** | Fetch external sources into local disk | ZIPs, CSVs, XML under `data/raw/` |
| **Processing** | Normalize + join into analysis-ready tables | Parquet under `data/interim/` and `data/processed/` |

Downloaded/generated data under `data/raw/**`, `data/interim/**`, and `data/processed/**` is gitignored (directory placeholders are kept).

---

## Setup

Requires **Python 3.11+** (developed with 3.13).

```bash
git clone https://github.com/sezfabian/Spatio-Temporal-Anomaly-Detection-of-Urban-Mobility.git
cd Spatio-Temporal-Anomaly-Detection-of-Urban-Mobility

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Always run commands from the repo root with the venv active.

Prefer invoking tools via the project interpreter so the correct packages are used:

```bash
python -m pytest -q
python -m src.ingestion.download --help
python -m src.processing --help
```

---

## Project layout

```text
configs/
  toronto_datasets.yaml          # CKAN datasets + ECCC weather config
  toronto_civic_calendar.yaml    # holidays / mega-events (2014–2017)
src/
  ingestion/                     # download / resolve / preview CLIs
  features/                      # civic calendar feature helpers
  processing/                    # raw → interim → processed
tests/
data/
  raw/                           # ingested files (not committed)
  interim/                       # tidy Parquet tables
  processed/                     # analysis panel + QA JSON
```

---

## Ingestion

Ingestion pulls configured sources into `data/raw/{dataset_key}/...`.

### Configured sources

Defined in [`configs/toronto_datasets.yaml`](configs/toronto_datasets.yaml):

| Config key | Source | What gets downloaded |
|---|---|---|
| `travel_times_bluetooth` | [Toronto Open Data](https://open.toronto.ca/) CKAN | Travel-time ZIPs (2014–2017), routes shapefile, readme |
| `king_st_bluetooth_segments` | Toronto Open Data CKAN | King St segment geometries |
| `king_st_bluetooth_travel_times` | Toronto Open Data CKAN | King St travel times |
| `festivals_events` | Toronto Open Data CKAN | Historical festivals XML (2014–2016) + readme |
| `weather` (top-level block) | [ECCC climate](https://climate.weather.gc.ca/) | Hourly Toronto City CSVs (`climate_id=6158355`, 2014–2017) |

Civic calendar events are **not downloaded**; they live in [`configs/toronto_civic_calendar.yaml`](configs/toronto_civic_calendar.yaml) and are materialized during processing.

### 1) Resolve CKAN resource URLs

Lists configured resources and their live download URLs (no files written):

```bash
# all CKAN datasets in config
python -m src.ingestion.ckan

# one dataset
python -m src.ingestion.ckan --dataset travel_times_bluetooth
python -m src.ingestion.ckan --dataset festivals_events
```

Example output (TSV):

```text
travel_times_bluetooth  travel-time-2014  travel_times  2014  ZIP  https://...
```

### 2) Download Toronto Open Data (CKAN) files

```bash
# everything listed under datasets: in toronto_datasets.yaml
python -m src.ingestion.download

# one dataset
python -m src.ingestion.download --dataset travel_times_bluetooth
python -m src.ingestion.download --dataset festivals_events

# only certain resource kinds
python -m src.ingestion.download --dataset travel_times_bluetooth --kind travel_times --kind geo

# re-download even if files already exist
python -m src.ingestion.download --dataset travel_times_bluetooth --force
```

Outputs land under:

```text
data/raw/travel_times_bluetooth/
  travel-time-2014.zip
  ...
  bluetooth-routes-wgs84.zip
data/raw/festivals_events/
  festivals-and-events-historical-xml-feed-jan-2014-dec-2016.xml
  festivals-and-events-readme.xls
```

Existing files are **skipped** unless you pass `--force`.

### 3) Download hourly weather (ECCC)

Weather uses a separate CLI (month-by-month bulk CSV API):

```bash
# years from config (default 2014–2017)
python -m src.ingestion.weather

# subset of years
python -m src.ingestion.weather --start-year 2014 --end-year 2014

python -m src.ingestion.weather --force
```

Outputs:

```text
data/raw/weather_toronto_city/
  hourly_2014_01.csv
  hourly_2014_02.csv
  ...
```

### 4) Preview raw files (optional)

Inspect schemas/samples without fully extracting large archives:

```bash
python -m src.ingestion.preview --dataset travel_times_bluetooth --rows 5
python -m src.ingestion.preview --dataset weather_toronto_city --rows 5
python -m src.ingestion.preview --file data/raw/festivals_events/festivals-and-events-historical-xml-feed-jan-2014-dec-2016.xml
```

### Adding a new ingestion source

#### A) New Toronto Open Data (CKAN) dataset

1. Find the package on [open.toronto.ca](https://open.toronto.ca/) and note the **package id** (URL slug) and exact **resource names**.
2. Add a block under `datasets:` in `configs/toronto_datasets.yaml`:

```yaml
datasets:
  my_new_dataset:
    package_id: some-ckan-package-id
    description: Short description of the dataset
    resources:
      - name: exact-resource-name-from-ckan
        kind: travel_times   # free label: geo | docs | events | ...
        year: 2016           # optional
```

3. Confirm names resolve:

```bash
python -m src.ingestion.ckan --dataset my_new_dataset
```

4. Download:

```bash
python -m src.ingestion.download --dataset my_new_dataset
```

Files are written to `data/raw/my_new_dataset/`.

Tips:
- Resource `name` values must match CKAN **exactly** (case/spacing).
- Supported formats for local extensions include ZIP, CSV, JSON, XML, XLS/XLSX, GeoJSON, etc. (`src/ingestion/download.py`).
- If you need the new source in the analysis panel, also add a **processing** normalizer (see below).

#### B) New weather station / year range

Edit the top-level `weather:` block in `configs/toronto_datasets.yaml` (`climate_id`, `start_year`, `end_year`, …), then rerun:

```bash
python -m src.ingestion.weather
```

#### C) New civic holidays / mega-events

Edit [`configs/toronto_civic_calendar.yaml`](configs/toronto_civic_calendar.yaml), then regenerate interim civic days during processing (`--step civic`).

Inspect a date:

```bash
python -m src.features.civic_calendar --date 2014-12-25
```

---

## Processing

Processing turns `data/raw` (+ civic calendar config) into tidy Parquet tables, then joins them into one analysis panel.

### Steps (what each one is for)

| `--step` | Reads from | Writes | Purpose |
|---|---|---|---|
| `travel_times` | `data/raw/travel_times_bluetooth/travel-time-*.zip` | `data/interim/travel_times.parquet` | Unzip/normalize 5‑minute Bluetooth observations → `route_id`, `ts_local`, `travel_time_s`, `sample_count`, `year` |
| `weather` | `data/raw/weather_toronto_city/hourly_*.csv` | `data/interim/weather_hourly.parquet` | Concatenate monthly ECCC files into one hourly weather table (`temp_c`, `precip_mm`, …) |
| `civic` | `configs/toronto_civic_calendar.yaml` | `data/interim/civic_days.parquet` | One row per calendar day with holiday / parade / mega-event flags |
| `events` | festivals historical XML under `data/raw/festivals_events/` | `data/interim/events.parquet` | Parse festivals/events into tidy rows (`name`, `start_local`, `end_local`, `lat`, `lon`, `road_close`, …) |
| `routes` | `bluetooth-routes-wgs84.zip` | `data/interim/routes.parquet` | Route attributes (`route_id`, `free_flow_s`, `length_m`) for delay features |
| `panel` | all interim tables above | `data/processed/route_time_panel.parquet` (+ QA JSON) | Join everything into a route × timestamp analysis table |
| `all` | — | all of the above, in order | Full rebuild |

**Panel joins (conceptually):**
1. travel times ⟕ routes on `route_id`
2. ⟕ weather on hour (`ts_local` floored to hour, DST-safe)
3. ⟕ civic flags on calendar `date`
4. ⟕ citywide event exposure counts on `date`
5. derived fields: `hour`, `dow`, `is_weekend`, `delay_s` (= travel − free-flow when available)

### Run processing

```bash
# full pipeline (can take a while / several GB of RAM for all years)
python -m src.processing --step all

# run steps one by one (recommended while developing)
python -m src.processing --step travel_times
python -m src.processing --step weather
python -m src.processing --step civic
python -m src.processing --step events
python -m src.processing --step routes
python -m src.processing --step panel
```

Process / panel only one travel-time year (faster smoke runs):

```bash
python -m src.processing --step travel_times --year 2017
python -m src.processing --step panel --year 2017

# multiple years
python -m src.processing --step all --year 2014 --year 2015
```

### Outputs

**Interim (tidy, separately analyzable):**

| File | Grain | Main columns |
|---|---|---|
| `data/interim/travel_times.parquet` | route × 5‑min timestamp | `route_id`, `ts_local`, `travel_time_s`, `sample_count`, `year` |
| `data/interim/weather_hourly.parquet` | hour | `ts_local`, `temp_c`, `rel_hum_pct`, `precip_mm`, `wind_spd_kmh`, … |
| `data/interim/civic_days.parquet` | day | `date`, `is_holiday`, `is_parade`, `is_mega_event`, … |
| `data/interim/events.parquet` | event | `event_id`, `name`, `start_local`, `end_local`, `lat`, `lon`, `area`, `road_close`, … |
| `data/interim/routes.parquet` | route | `route_id`, `free_flow_s`, `length_m` |

**Processed (main stats / ML table):**

| File | Description |
|---|---|
| `data/processed/route_time_panel.parquet` | Joined route×time panel with weather (`wx_*`), civic flags, event counts, `delay_s`, calendar fields |
| `data/processed/route_time_panel_qa.json` | Row counts, year range, join match rates, null rates |

Quick peek in Python:

```python
import pandas as pd

panel = pd.read_parquet("data/processed/route_time_panel.parquet")
print(panel.shape)
print(panel.columns.tolist())
print(panel.head())
```

### Adding processing for a new source

1. Ingest it to `data/raw/...` (see above).
2. Add a normalizer under `src/processing/` that writes a tidy Parquet to `data/interim/`.
3. Wire it into `src/processing/__main__.py` (`STEPS` + `run_step`).
4. If it belongs on the analysis table, join it in `src/processing/panel.py`.
5. Add fixture-based tests under `tests/processing/`.

---

## Tests

```bash
source venv/bin/activate
python -m pytest -q
```

---

## Notes / known limits

- Festivals **live JSON feed** on Toronto Open Data is currently blocked upstream; ingestion uses the **historical XML (Jan 2014–Dec 2016)** only. 2017 travel times still get civic-calendar features.
- Full `travel_times` + `panel` for 2014–2017 is large (tens of millions of rows). Use `--year` while iterating.
- Event exposure on the panel is **citywide by date** in v1 (not yet spatially buffered to nearby routes).
