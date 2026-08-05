"""Tests for congestion frames and interactive Plotly maps."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.viz.frames import build_congestion_frames, panel_date_bounds
from src.viz.interactive import build_animated_congestion_figure, save_interactive_map


def _period_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route_id": ["A_B", "A_B", "A_B", "B_C"],
            "ts_local": pd.to_datetime(
                [
                    "2016-06-15 07:00:00",
                    "2016-06-15 08:00:00",
                    "2016-06-16 07:00:00",
                    "2016-06-16 07:00:00",
                ]
            ),
            "date": [
                date(2016, 6, 15),
                date(2016, 6, 15),
                date(2016, 6, 16),
                date(2016, 6, 16),
            ],
            "delay_s": [10.0, 40.0, 20.0, 80.0],
        }
    )


def test_build_congestion_frames_period() -> None:
    labels, wide = build_congestion_frames(
        _period_panel(),
        start="2016-06-15",
        end="2016-06-16",
        freq="h",
    )
    assert len(labels) == 3
    assert wide.loc["A_B", "2016-06-15 07:00"] == 10.0
    assert wide.loc["B_C", "2016-06-16 07:00"] == 80.0


def test_panel_date_bounds() -> None:
    lo, hi = panel_date_bounds(_period_panel())
    assert lo == date(2016, 6, 15)
    assert hi == date(2016, 6, 16)


def test_build_animated_figure_requires_geometry() -> None:
    with pytest.raises(FileNotFoundError):
        build_animated_congestion_figure(
            _period_panel(),
            start="2016-06-15",
            end="2016-06-16",
            freq="h",
            routes_zip=Path("/nonexistent/routes.zip"),
        )


def test_save_interactive_map_html(tmp_path: Path) -> None:
    # Minimal figure path: only test HTML writer helper with a tiny figure.
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Scattermap(lon=[-79.4], lat=[43.65], mode="markers")])
    out = save_interactive_map(fig, tmp_path / "map.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "plotly" in text.lower()
