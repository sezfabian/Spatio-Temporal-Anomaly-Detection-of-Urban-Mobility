"""Bluetooth route network visualization on a Toronto street basemap."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shapefile
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from src.processing.paths import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR

DEFAULT_PANEL = DEFAULT_PROCESSED_DIR / "route_time_panel.parquet"
DEFAULT_ROUTES_ZIP = (
    DEFAULT_RAW_DIR / "travel_times_bluetooth" / "bluetooth-routes-wgs84.zip"
)

BASEMAP_STYLES = ("positron", "voyager", "streets", "dark")


def split_route_id(route_id: str) -> tuple[str, str]:
    """Split ``SOURCE_TARGET`` route ids into detector node ids."""
    parts = str(route_id).split("_")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Expected SOURCE_TARGET route_id, got {route_id!r}")
    return parts[0], parts[1]


def aggregate_route_metrics(
    panel: pd.DataFrame | Path,
    *,
    metric: str = "delay_s",
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate a panel metric to one value per ``route_id``.

    Returns:
        DataFrame with ``route_id``, ``source``, ``target``, ``value``.
    """
    if isinstance(panel, (str, Path)):
        frame = pd.read_parquet(panel, columns=["route_id", metric])
    else:
        frame = panel
    if "route_id" not in frame.columns:
        raise KeyError("panel must include route_id")
    if metric not in frame.columns:
        raise KeyError(f"panel missing metric column {metric!r}")

    series = frame.groupby("route_id", sort=False)[metric]
    if agg == "p95":
        values = series.quantile(0.95)
    else:
        values = series.agg(agg)

    out = values.rename("value").reset_index()
    ends = out["route_id"].map(split_route_id)
    out["source"] = ends.map(lambda x: x[0])
    out["target"] = ends.map(lambda x: x[1])
    return out


def load_route_geometries(routes_zip: Path) -> dict[str, list[tuple[float, float]]]:
    """Load WGS84 polyline coordinates keyed by ``route_id``."""
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


def build_route_graph(
    metrics: pd.DataFrame,
    *,
    routes_zip: Path | None = DEFAULT_ROUTES_ZIP,
) -> nx.DiGraph:
    """Build a directed graph of Bluetooth corridors with optional geo paths."""
    graph = nx.DiGraph()
    geometries: dict[str, list[tuple[float, float]]] = {}
    if routes_zip is not None and Path(routes_zip).exists():
        geometries = load_route_geometries(Path(routes_zip))

    node_xy: dict[str, list[tuple[float, float]]] = {}
    for row in metrics.itertuples(index=False):
        route_id = str(row.route_id)
        source = str(row.source)
        target = str(row.target)
        value = float(row.value) if pd.notna(row.value) else float("nan")
        points = geometries.get(route_id)
        graph.add_edge(
            source,
            target,
            route_id=route_id,
            value=value,
            points=points,
        )
        if points:
            node_xy.setdefault(source, []).append(points[0])
            node_xy.setdefault(target, []).append(points[-1])

    for node, samples in node_xy.items():
        xs = [p[0] for p in samples]
        ys = [p[1] for p in samples]
        graph.nodes[node]["pos"] = (float(np.mean(xs)), float(np.mean(ys)))

    return graph


def add_toronto_basemap(
    ax: Axes,
    *,
    style: str = "positron",
) -> bool:
    """Overlay Toronto street tiles under lon/lat axes (EPSG:4326).

    Returns:
        ``True`` if tiles were added, else ``False``.
    """
    try:
        import contextily as cx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Basemap requires contextily. Install with: pip install contextily pyproj"
        ) from exc

    providers = {
        "positron": cx.providers.CartoDB.Positron,
        "voyager": cx.providers.CartoDB.Voyager,
        "streets": cx.providers.OpenStreetMap.Mapnik,
        "dark": cx.providers.CartoDB.DarkMatter,
    }
    if style not in providers:
        raise ValueError(f"Unknown basemap style {style!r}; choose from {BASEMAP_STYLES}")

    try:
        cx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=providers[style],
            attribution_size=6,
            zorder=0,
        )
    except Exception as exc:
        ax.text(
            0.01,
            0.01,
            f"Basemap unavailable ({exc.__class__.__name__})",
            transform=ax.transAxes,
            fontsize=8,
            color="0.4",
            zorder=5,
        )
        return False
    return True


def _resolve_positions(graph: nx.DiGraph) -> tuple[dict[str, tuple[float, float]], bool]:
    geo_nodes = {
        node: data["pos"]
        for node, data in graph.nodes(data=True)
        if "pos" in data
    }
    if len(geo_nodes) == graph.number_of_nodes() and geo_nodes:
        return geo_nodes, True
    pos = nx.spring_layout(graph, seed=42, k=1.2 / max(graph.number_of_nodes(), 1) ** 0.5)
    return pos, False


def _edge_segments(
    graph: nx.DiGraph,
    pos: dict[str, tuple[float, float]],
) -> tuple[list[np.ndarray], np.ndarray]:
    segments: list[np.ndarray] = []
    values: list[float] = []
    for source, target, data in graph.edges(data=True):
        points = data.get("points")
        if points and len(points) >= 2:
            coords = np.asarray(points, dtype=float)
        else:
            coords = np.asarray([pos[source], pos[target]], dtype=float)
        segments.append(coords)
        values.append(float(data.get("value", np.nan)))
    return segments, np.asarray(values, dtype=float)


def _line_widths(values: np.ndarray, norm: Normalize) -> list[float]:
    widths: list[float] = []
    for value in values:
        if np.isfinite(value):
            widths.append(1.2 + 4.0 * float(norm(value)))
        else:
            widths.append(0.8)
    return widths


def plot_route_network(
    graph: nx.DiGraph,
    *,
    metric_label: str = "delay_s",
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
    node_labels: bool = True,
    cmap: str = "YlOrRd",
    basemap: bool = True,
    basemap_style: str = "positron",
    show: bool = False,
    save_path: Path | str | None = None,
) -> Axes:
    """Draw the route network colored by edge metric, optionally on a Toronto map."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    pos, is_geo = _resolve_positions(graph)
    if is_geo:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.set_xlabel("Spring-layout X")
        ax.set_ylabel("Spring-layout Y")

    segments, values = _edge_segments(graph, pos)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite edge metric values to plot")
    norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))

    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=_line_widths(values, norm),
        array=values,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    ax.add_collection(collection)

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    ax.scatter(xs, ys, s=28, c="0.15", zorder=3)
    if node_labels:
        for node, (x, y) in pos.items():
            ax.annotate(
                node,
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color="0.15",
                zorder=4,
            )

    pad_x = (max(xs) - min(xs)) * 0.05 or 0.01
    pad_y = (max(ys) - min(ys)) * 0.05 or 0.01
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    if is_geo and basemap:
        add_toronto_basemap(ax, style=basemap_style)
        ax.grid(False)
    else:
        ax.grid(True, alpha=0.25)

    cbar = ax.figure.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label)
    ax.set_title(title or f"Toronto Bluetooth network ({metric_label})")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def visualize_processed_network(
    panel: pd.DataFrame | Path = DEFAULT_PANEL,
    *,
    routes_zip: Path | None = DEFAULT_ROUTES_ZIP,
    metric: str = "delay_s",
    agg: str = "mean",
    title: str | None = None,
    basemap: bool = True,
    basemap_style: str = "positron",
    save_path: Path | str | None = None,
    show: bool = True,
    **plot_kwargs: Any,
) -> tuple[nx.DiGraph, Axes]:
    """Aggregate panel metrics and plot the Bluetooth network on a Toronto basemap.

    Args:
        panel: Processed ``route_time_panel`` path or DataFrame.
        routes_zip: Routes shapefile ZIP for geographic layout.
        metric: Panel column to color edges by.
        agg: Aggregation over timestamps per route (``mean``, ``median``, ``p95``).
        title: Optional plot title.
        basemap: Overlay Toronto street tiles when geometry is available.
        basemap_style: ``positron``, ``voyager``, ``streets``, or ``dark``.
        save_path: Optional image output path.
        show: Open an interactive window.
        **plot_kwargs: Extra args forwarded to :func:`plot_route_network`.

    Returns:
        ``(graph, axes)``.
    """
    metrics = aggregate_route_metrics(panel, metric=metric, agg=agg)
    graph = build_route_graph(metrics, routes_zip=routes_zip)
    label = f"{agg} {metric}"
    axes = plot_route_network(
        graph,
        metric_label=label,
        title=title or f"Toronto Bluetooth network · {label}",
        basemap=basemap,
        basemap_style=basemap_style,
        save_path=save_path,
        show=show,
        **plot_kwargs,
    )
    return graph, axes


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Toronto network map."""
    parser = argparse.ArgumentParser(
        description="Plot Bluetooth routes on a Toronto street basemap.",
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--routes-zip", type=Path, default=DEFAULT_ROUTES_ZIP)
    parser.add_argument("--metric", default="delay_s")
    parser.add_argument("--agg", default="mean")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-geo", action="store_true")
    parser.add_argument("--no-basemap", action="store_true")
    parser.add_argument(
        "--basemap-style",
        choices=BASEMAP_STYLES,
        default="positron",
    )
    args = parser.parse_args(argv)

    visualize_processed_network(
        args.panel,
        routes_zip=None if args.no_geo else args.routes_zip,
        metric=args.metric,
        agg=args.agg,
        basemap=not args.no_basemap,
        basemap_style=args.basemap_style,
        save_path=args.save,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
