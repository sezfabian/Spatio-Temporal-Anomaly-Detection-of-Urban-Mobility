"""Tests for Bluetooth route network visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import pandas as pd
import pytest

from src.viz.network import (
    aggregate_route_metrics,
    build_route_graph,
    plot_route_network,
    split_route_id,
    visualize_processed_network,
)


def test_split_route_id() -> None:
    assert split_route_id("AC4_AC3") == ("AC4", "AC3")
    assert split_route_id("B1_C") == ("B1", "C")
    with pytest.raises(ValueError):
        split_route_id("INVALID")


def test_aggregate_build_and_plot(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "route_id": ["A_B", "A_B", "B_C", "B_C"],
            "delay_s": [10.0, 30.0, 100.0, 200.0],
        }
    )
    metrics = aggregate_route_metrics(panel, metric="delay_s", agg="mean")
    assert metrics.loc[metrics["route_id"] == "A_B", "value"].iloc[0] == 20.0

    graph = build_route_graph(metrics, routes_zip=None)
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_edges() == 2
    assert graph["A"]["B"]["value"] == 20.0

    out = tmp_path / "network.png"
    ax = plot_route_network(graph, show=False, save_path=out, basemap=False)
    assert out.exists()
    assert ax is not None


def test_add_toronto_basemap_handles_failure(monkeypatch) -> None:
    import builtins
    import sys
    from types import SimpleNamespace

    import matplotlib.pyplot as plt

    from src.viz.network import add_toronto_basemap

    fig, ax = plt.subplots()
    ax.set_xlim(-79.5, -79.3)
    ax.set_ylim(43.6, 43.7)

    boom = SimpleNamespace(
        providers=SimpleNamespace(
            CartoDB=SimpleNamespace(Positron="x", Voyager="x", DarkMatter="x"),
            OpenStreetMap=SimpleNamespace(Mapnik="x"),
        ),
        add_basemap=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setitem(sys.modules, "contextily", boom)
    real_import = builtins.__import__

    def import_stub(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "contextily":
            return boom
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_stub)
    assert add_toronto_basemap(ax, style="positron") is False
    plt.close(fig)


def test_visualize_processed_network(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "route_id": ["J_I", "J_I", "I_H"],
            "delay_s": [5.0, 15.0, 40.0],
        }
    )
    out = tmp_path / "viz.png"
    graph, ax = visualize_processed_network(
        panel,
        routes_zip=None,
        metric="delay_s",
        agg="mean",
        save_path=out,
        show=False,
        basemap=False,
    )
    assert graph.has_edge("J", "I")
    assert out.exists()
    assert ax.get_title()
