"""Unit tests for src.ingestion.download (HTTP transfers are mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ingestion.ckan import ResolvedResource
from src.ingestion.download import (
    download_resource,
    download_resources,
    extension_for_resource,
    local_path_for_resource,
    main,
)


def _resource(**overrides: object) -> ResolvedResource:
    base = dict(
        dataset_key="travel_times_bluetooth",
        package_id="travel-times-bluetooth",
        name="travel-time-2014",
        kind="travel_times",
        year=2014,
        resource_id="rid",
        format="ZIP",
        url="https://example.ckan/travel-time-2014.zip",
    )
    base.update(overrides)
    return ResolvedResource(**base)  # type: ignore[arg-type]


def test_extension_for_resource_uses_format_map() -> None:
    assert extension_for_resource(_resource(format="xlsx")) == ".xlsx"
    assert extension_for_resource(_resource(format="ZIP")) == ".zip"
    assert extension_for_resource(_resource(format="XML")) == ".xml"


def test_extension_for_resource_falls_back_to_url_suffix() -> None:
    resource = _resource(
        format="UNKNOWN",
        url="https://example.ckan/files/routes.geojson?download=1",
    )
    assert extension_for_resource(resource) == ".geojson"


def test_local_path_for_resource_nests_by_dataset(tmp_path: Path) -> None:
    path = local_path_for_resource(_resource(), tmp_path)
    assert path == tmp_path / "travel_times_bluetooth" / "travel-time-2014.zip"


def test_download_resource_streams_to_destination(tmp_path: Path) -> None:
    destination = tmp_path / "travel_times_bluetooth" / "travel-time-2014.zip"
    resource = _resource()

    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    response.iter_content.return_value = [b"abc", b"def"]

    session = MagicMock(spec=requests.Session)
    session.get.return_value = response

    result = download_resource(resource, destination, session)

    assert result.status == "downloaded"
    assert result.bytes_written == 6
    assert destination.read_bytes() == b"abcdef"
    assert not destination.with_suffix(".zip.partial").exists()
    session.get.assert_called_once_with(resource.url, stream=True, timeout=120)


def test_download_resource_skips_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "file.zip"
    destination.write_bytes(b"already-here")
    session = MagicMock(spec=requests.Session)

    result = download_resource(_resource(), destination, session, force=False)

    assert result.status == "skipped"
    session.get.assert_not_called()
    assert destination.read_bytes() == b"already-here"


def test_download_resource_force_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "file.zip"
    destination.write_bytes(b"old")

    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    response.iter_content.return_value = [b"new"]

    session = MagicMock(spec=requests.Session)
    session.get.return_value = response

    result = download_resource(_resource(), destination, session, force=True)

    assert result.status == "downloaded"
    assert destination.read_bytes() == b"new"


def test_download_resource_failed_cleans_partial(tmp_path: Path) -> None:
    destination = tmp_path / "file.zip"
    resource = _resource()

    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    response.raise_for_status.side_effect = requests.HTTPError("500")

    session = MagicMock(spec=requests.Session)
    session.get.return_value = response

    result = download_resource(resource, destination, session)

    assert result.status == "failed"
    assert "500" in (result.error or "")
    assert not destination.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_download_resources_filters_by_kind(tmp_path: Path) -> None:
    resources = [
        _resource(name="travel-time-2014", kind="travel_times"),
        _resource(name="readme", kind="docs", year=None, format="XLSX", url="https://x/r.xlsx"),
        _resource(name="routes", kind="geo", year=None, url="https://x/routes.zip"),
    ]

    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    response.iter_content.return_value = [b"x"]

    session = MagicMock(spec=requests.Session)
    session.get.return_value = response

    results = download_resources(
        resources,
        tmp_path,
        session,
        kinds={"travel_times", "geo"},
    )

    assert [r.resource.name for r in results] == ["travel-time-2014", "routes"]
    assert all(r.status == "downloaded" for r in results)


def test_main_returns_nonzero_on_failure(tmp_path: Path) -> None:
    failed = MagicMock()
    failed.status = "failed"
    failed.resource = _resource()
    failed.path = tmp_path / "x.zip"
    failed.error = "boom"

    with (
        patch("src.ingestion.download.load_dataset_config", return_value={"ckan_base_url": "https://x/"}),
        patch("src.ingestion.download.CkanClient") as client_cls,
        patch(
            "src.ingestion.download.download_resources",
            return_value=[failed],
        ),
    ):
        client_cls.return_value.resolve_configured_resources.return_value = []
        client_cls.return_value.session = MagicMock()
        code = main(["--raw-dir", str(tmp_path)])

    assert code == 1
