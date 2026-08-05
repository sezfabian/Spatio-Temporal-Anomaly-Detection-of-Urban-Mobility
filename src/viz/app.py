"""Streamlit app: choose date period + interval, play congestion on a zoomable map."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

# ``streamlit run src/viz/app.py`` does not put the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.viz.frames import INTERVAL_OPTIONS, panel_date_bounds, parse_day
from src.viz.interactive import MAP_STYLES, build_animated_congestion_figure
from src.viz.network import DEFAULT_PANEL, DEFAULT_ROUTES_ZIP

st.set_page_config(
    page_title="Toronto congestion map",
    layout="wide",
)
st.title("Toronto Bluetooth congestion")
st.caption(
    "Pick a date period and interval, then play congestion over time. "
    "Zoom/pan the map with scroll and drag."
)

panel_path = Path(
    st.sidebar.text_input("Panel parquet", str(DEFAULT_PANEL))
)
routes_zip = Path(
    st.sidebar.text_input("Routes ZIP", str(DEFAULT_ROUTES_ZIP))
)
metric = st.sidebar.selectbox("Metric", ["delay_s", "travel_time_s"], index=0)
map_style = st.sidebar.selectbox("Basemap", list(MAP_STYLES), index=0)

if not panel_path.exists():
    st.error(f"Panel not found: {panel_path}")
    st.stop()

data_min, data_max = panel_date_bounds(panel_path)
default_start = data_min + timedelta(days=max((data_max - data_min).days // 2, 0))
default_end = min(default_start + timedelta(days=1), data_max)

col_a, col_b, col_c = st.columns(3)
with col_a:
    start = st.date_input(
        "Start date",
        value=default_start,
        min_value=data_min,
        max_value=data_max,
    )
with col_b:
    end = st.date_input(
        "End date",
        value=default_end,
        min_value=data_min,
        max_value=data_max,
    )
with col_c:
    interval_label = st.selectbox("Interval", list(INTERVAL_OPTIONS.keys()), index=3)

freq = INTERVAL_OPTIONS[interval_label]
st.caption(f"Data coverage: {data_min} → {data_max}")

if end < start:
    st.error("End date must be on or after start date.")
    st.stop()

run = st.button("Build map", type="primary")
if run or "fig" in st.session_state:
    if run:
        with st.spinner("Building animated map…"):
            try:
                fig = build_animated_congestion_figure(
                    panel_path,
                    start=parse_day(start),
                    end=parse_day(end),
                    freq=freq,
                    metric=metric,
                    routes_zip=routes_zip if routes_zip.exists() else None,
                    map_style=map_style,
                )
            except Exception as exc:
                st.error(str(exc))
                st.stop()
            st.session_state["fig"] = fig
            st.session_state["meta"] = f"{start} → {end} · {interval_label} · {metric}"

    st.subheader(st.session_state.get("meta", ""))
    st.plotly_chart(
        st.session_state["fig"],
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["zoomInMap", "zoomOutMap"],
        },
    )
    st.info(
        "Use the Play / Pause buttons and time slider under the map. "
        "Scroll to zoom, drag to pan."
    )
else:
    st.write("Choose a period and interval, then click **Build map**.")
