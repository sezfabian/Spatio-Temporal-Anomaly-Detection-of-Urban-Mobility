"""Visualization helpers for processed mobility data."""

from src.viz.frames import build_congestion_frames
from src.viz.network import (
    aggregate_route_metrics,
    build_route_graph,
    plot_route_network,
    visualize_processed_network,
)

__all__ = [
    "aggregate_route_metrics",
    "build_animated_congestion_figure",
    "build_congestion_frames",
    "build_route_graph",
    "plot_route_network",
    "save_interactive_map",
    "visualize_processed_network",
]


def __getattr__(name: str):
    # Lazy imports avoid runpy warnings for ``python -m src.viz.interactive``.
    if name in {"build_animated_congestion_figure", "save_interactive_map"}:
        from src.viz import interactive as _interactive

        return getattr(_interactive, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
