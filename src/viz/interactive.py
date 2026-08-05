"""Interactive Plotly congestion map with play slider and zoom/pan."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from src.viz.frames import (
    INTERVAL_OPTIONS,
    build_congestion_frames,
    panel_date_bounds,
    parse_day,
)
from src.viz.network import DEFAULT_PANEL, DEFAULT_ROUTES_ZIP, load_route_geometries

MAP_STYLES = {
    "positron": "carto-positron",
    "dark": "carto-darkmatter",
    "streets": "open-street-map",
}


def _color_for_value(value: float, vmin: float, vmax: float, colorscale: str) -> str:
    if not np.isfinite(value) or vmax <= vmin:
        return "rgba(160,160,160,0.35)"
    t = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    return sample_colorscale(colorscale, [t])[0]


def _width_for_value(value: float, vmin: float, vmax: float) -> float:
    if not np.isfinite(value) or vmax <= vmin:
        return 1.5
    t = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    return 1.5 + 5.0 * t


def build_animated_congestion_figure(
    panel: pd.DataFrame | Path = DEFAULT_PANEL,
    *,
    start: str | date | datetime,
    end: str | date | datetime,
    freq: str = "h",
    metric: str = "delay_s",
    routes_zip: Path | None = DEFAULT_ROUTES_ZIP,
    map_style: str = "positron",
    colorscale: str = "YlOrRd",
    max_frames: int = 500,
    title: str | None = None,
) -> go.Figure:
    """Build a Plotly map that animates congestion over a date period.

    The figure supports zoom/pan on the basemap, a time slider, and play/pause.
    """
    labels, wide = build_congestion_frames(
        panel,
        start=start,
        end=end,
        freq=freq,
        metric=metric,
        max_frames=max_frames,
    )
    if routes_zip is None or not Path(routes_zip).exists():
        raise FileNotFoundError(
            "Interactive map needs route geometry. "
            f"Missing routes ZIP: {routes_zip}"
        )
    geometries = load_route_geometries(Path(routes_zip))

    route_ids = [rid for rid in wide.index.astype(str) if rid in geometries]
    if not route_ids:
        raise ValueError("No routes with both panel values and geometry")

    flat = wide.loc[route_ids].to_numpy(dtype=float)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("No finite congestion values in selected period")
    vmin = float(np.nanpercentile(finite, 5))
    vmax = float(np.nanpercentile(finite, 95))
    if vmax <= vmin:
        vmax = vmin + 1.0

    def _style_traces(frame_label: str, *, include_geometry: bool) -> list[go.Scattermap]:
        traces: list[go.Scattermap] = []
        col = wide[frame_label]
        for route_id in route_ids:
            value = float(col.get(route_id, np.nan))
            line = dict(
                width=_width_for_value(value, vmin, vmax),
                color=_color_for_value(value, vmin, vmax, colorscale),
            )
            if include_geometry:
                points = geometries[route_id]
                traces.append(
                    go.Scattermap(
                        lon=[p[0] for p in points],
                        lat=[p[1] for p in points],
                        mode="lines",
                        line=line,
                        name=route_id,
                        hovertemplate=(
                            f"{route_id}<br>{metric}: "
                            f"{'n/a' if not np.isfinite(value) else f'{value:.0f}s'}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
            else:
                # Frames only restyle color/width; geometry stays on base traces.
                traces.append(go.Scattermap(line=line))
        return traces

    all_lons = [p[0] for rid in route_ids for p in geometries[rid]]
    all_lats = [p[1] for rid in route_ids for p in geometries[rid]]
    center = {"lon": float(np.mean(all_lons)), "lat": float(np.mean(all_lats))}

    style_token = MAP_STYLES.get(map_style, MAP_STYLES["positron"])
    fig = go.Figure(data=_style_traces(labels[0], include_geometry=True))
    fig.frames = [
        go.Frame(
            data=_style_traces(label, include_geometry=False),
            name=label,
            traces=list(range(len(route_ids))),
        )
        for label in labels
    ]

    slider_steps = [
        {
            "args": [
                [label],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": label,
            "method": "animate",
        }
        for label in labels
    ]

    fig.update_layout(
        title=title
        or (
            f"Toronto congestion · {metric} · "
            f"{parse_day(start)} → {parse_day(end)} · {freq}"
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        map=dict(
            style=style_token,
            center=center,
            zoom=11,
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.02,
                "xanchor": "left",
                "yanchor": "bottom",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 350, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {"b": 10, "t": 40},
                "len": 0.9,
                "x": 0.05,
                "y": 0,
                "steps": slider_steps,
            }
        ],
        coloraxis=dict(
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title=metric),
        ),
    )
    # Invisible scatter keeps the colorbar present.
    fig.add_trace(
        go.Scattermap(
            lon=[center["lon"]],
            lat=[center["lat"]],
            mode="markers",
            marker=dict(size=0.1, color=[vmin], coloraxis="coloraxis", opacity=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


def save_interactive_map(
    fig: go.Figure,
    path: Path | str,
) -> Path:
    """Write a self-contained interactive HTML map."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: build an interactive HTML congestion map for a date period."""
    parser = argparse.ArgumentParser(
        description="Interactive Toronto congestion map (zoom, pan, play).",
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--routes-zip", type=Path, default=DEFAULT_ROUTES_ZIP)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--freq",
        default="h",
        choices=list(INTERVAL_OPTIONS.values()),
        help="Time interval / animation frame size",
    )
    parser.add_argument("--metric", default="delay_s")
    parser.add_argument(
        "--map-style",
        choices=list(MAP_STYLES),
        default="positron",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("figures/congestion_interactive.html"),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure in a browser",
    )
    args = parser.parse_args(argv)

    data_min, data_max = panel_date_bounds(args.panel)
    if args.start is None or args.end is None:
        mid = data_min + timedelta(days=max((data_max - data_min).days // 2, 0))
        start = args.start or mid
        end = args.end or min(mid + timedelta(days=1), data_max)
    else:
        start, end = args.start, args.end

    fig = build_animated_congestion_figure(
        args.panel,
        start=start,
        end=end,
        freq=args.freq,
        metric=args.metric,
        routes_zip=args.routes_zip,
        map_style=args.map_style,
    )
    out = save_interactive_map(fig, args.save)
    print(f"Wrote {out}")
    if args.show:
        fig.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
