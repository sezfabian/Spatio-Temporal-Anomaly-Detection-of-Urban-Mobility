"""Unit tests for src.ingestion.ckan (CKAN API calls are mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml

from src.ingestion.ckan import (
    CkanClient,
    ResolvedResource,
    _build_parser,
    load_dataset_config,
    main,
)


SAMPLE_CONFIG = {
    "ckan_base_url": "https://example.ckan/api/3/action",
    "datasets": {
        "travel_times_bluetooth": {
            "package_id": "travel-times-bluetooth",
            "description": "test package",
            "resources": [
                {"name": "bluetooth-routes-wgs84", "kind": "geo"},
                {"name": "travel-time-2014", "kind": "travel_times", "year": 2014},
            ],
        },
        "king_st_bluetooth_segments": {
            "package_id": "king-st-segments",
            "resources": [
                {"name": "segments-geojson", "kind": "geo"},
            ],
        },
    },
}


def _package_payload(package_id: str, resources: list[dict]) -> dict:
    return {
        "success": True,
        "result": {
            "name": package_id,
            "resources": resources,
        },
    }


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "toronto_datasets.yaml"
    path.write_text(yaml.safe_dump(SAMPLE_CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def mock_session() -> MagicMock:
    return MagicMock(spec=requests.Session)


def test_load_dataset_config_returns_mapping(config_path: Path) -> None:
    config = load_dataset_config(config_path)

    assert config["ckan_base_url"] == SAMPLE_CONFIG["ckan_base_url"]
    assert "travel_times_bluetooth" in config["datasets"]


def test_load_dataset_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected mapping"):
        load_dataset_config(path)


def test_ckan_client_normalizes_base_url_slash() -> None:
    client = CkanClient("https://example.ckan/api/3/action")
    assert client.base_url == "https://example.ckan/api/3/action/"

    client_with_slash = CkanClient("https://example.ckan/api/3/action/")
    assert client_with_slash.base_url == "https://example.ckan/api/3/action/"


def test_ckan_client_reuses_provided_session(mock_session: MagicMock) -> None:
    client = CkanClient("https://example.ckan/api/3/action/", session=mock_session)
    assert client.session is mock_session


def test_package_show_returns_result(mock_session: MagicMock) -> None:
    response = MagicMock()
    response.json.return_value = _package_payload(
        "travel-times-bluetooth",
        [{"name": "travel-time-2014", "id": "r1", "format": "zip", "url": "https://x/a.zip"}],
    )
    mock_session.get.return_value = response

    client = CkanClient("https://example.ckan/api/3/action/", session=mock_session)
    result = client.package_show("travel-times-bluetooth")

    assert result["name"] == "travel-times-bluetooth"
    mock_session.get.assert_called_once_with(
        "https://example.ckan/api/3/action/package_show",
        params={"id": "travel-times-bluetooth"},
        timeout=60,
    )
    response.raise_for_status.assert_called_once()


def test_package_show_raises_on_ckan_failure(mock_session: MagicMock) -> None:
    response = MagicMock()
    response.json.return_value = {"success": False, "error": {"message": "not found"}}
    mock_session.get.return_value = response

    client = CkanClient("https://example.ckan/api/3/action/", session=mock_session)

    with pytest.raises(RuntimeError, match="package_show failed"):
        client.package_show("missing-package")


def test_package_show_propagates_http_error(mock_session: MagicMock) -> None:
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("503")
    mock_session.get.return_value = response

    client = CkanClient("https://example.ckan/api/3/action/", session=mock_session)

    with pytest.raises(requests.HTTPError):
        client.package_show("travel-times-bluetooth")


def test_resolve_configured_resources_maps_local_to_remote(
    mock_session: MagicMock,
) -> None:
    response = MagicMock()
    response.json.return_value = _package_payload(
        "travel-times-bluetooth",
        [
            {
                "name": "bluetooth-routes-wgs84",
                "id": "geo-id",
                "format": "zip",
                "url": "https://x/routes.zip",
            },
            {
                "name": "travel-time-2014",
                "id": "tt-2014",
                "format": "ZIP",
                "url": "https://x/2014.zip",
            },
            {
                "name": "unconfigured-extra",
                "id": "extra",
                "format": "CSV",
                "url": "https://x/extra.csv",
            },
        ],
    )
    mock_session.get.return_value = response

    client = CkanClient("https://example.ckan/api/3/action/", session=mock_session)
    resolved = client.resolve_configured_resources(
        SAMPLE_CONFIG, dataset_keys=["travel_times_bluetooth"]
    )

    assert resolved == [
        ResolvedResource(
            dataset_key="travel_times_bluetooth",
            package_id="travel-times-bluetooth",
            name="bluetooth-routes-wgs84",
            kind="geo",
            year=None,
            resource_id="geo-id",
            format="ZIP",
            url="https://x/routes.zip",
        ),
        ResolvedResource(
            dataset_key="travel_times_bluetooth",
            package_id="travel-times-bluetooth",
            name="travel-time-2014",
            kind="travel_times",
            year=2014,
            resource_id="tt-2014",
            format="ZIP",
            url="https://x/2014.zip",
        ),
    ]
    assert mock_session.get.call_count == 1


def test_resolve_configured_resources_unknown_dataset(mock_session: MagicMock) -> None:
    client = CkanClient("https://example.ckan/api/3/action/", session=mock_session)

    with pytest.raises(KeyError, match="Unknown dataset key"):
        client.resolve_configured_resources(SAMPLE_CONFIG, dataset_keys=["nope"])


def test_resolve_configured_resources_missing_remote_resource(
    mock_session: MagicMock,
) -> None:
    response = MagicMock()
    response.json.return_value = _package_payload(
        "travel-times-bluetooth",
        [{"name": "only-other", "id": "1", "format": "ZIP", "url": "https://x/o.zip"}],
    )
    mock_session.get.return_value = response

    client = CkanClient("https://example.ckan/api/3/action/", session=mock_session)

    with pytest.raises(KeyError, match="bluetooth-routes-wgs84"):
        client.resolve_configured_resources(
            SAMPLE_CONFIG, dataset_keys=["travel_times_bluetooth"]
        )


def test_resolve_configured_resources_calls_package_show_per_dataset(
    mock_session: MagicMock,
) -> None:
    def _get(_url: str, params: dict, timeout: int) -> MagicMock:
        package_id = params["id"]
        response = MagicMock()
        if package_id == "travel-times-bluetooth":
            response.json.return_value = _package_payload(
                package_id,
                [
                    {
                        "name": "bluetooth-routes-wgs84",
                        "id": "g",
                        "format": "zip",
                        "url": "https://x/g.zip",
                    },
                    {
                        "name": "travel-time-2014",
                        "id": "t",
                        "format": "zip",
                        "url": "https://x/t.zip",
                    },
                ],
            )
        else:
            response.json.return_value = _package_payload(
                package_id,
                [
                    {
                        "name": "segments-geojson",
                        "id": "s",
                        "format": "geojson",
                        "url": "https://x/s.geojson",
                    }
                ],
            )
        return response

    mock_session.get.side_effect = _get
    client = CkanClient("https://example.ckan/api/3/action/", session=mock_session)

    resolved = client.resolve_configured_resources(SAMPLE_CONFIG)

    assert len(resolved) == 3
    assert {r.dataset_key for r in resolved} == {
        "travel_times_bluetooth",
        "king_st_bluetooth_segments",
    }
    assert mock_session.get.call_count == 2


def test_build_parser_accepts_repeatable_dataset_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["--dataset", "travel_times_bluetooth", "--dataset", "king_st_bluetooth_segments"]
    )
    assert args.datasets == ["travel_times_bluetooth", "king_st_bluetooth_segments"]


def test_main_prints_tsv_and_returns_zero(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_resources = [
        ResolvedResource(
            dataset_key="travel_times_bluetooth",
            package_id="travel-times-bluetooth",
            name="bluetooth-routes-wgs84",
            kind="geo",
            year=None,
            resource_id="geo-id",
            format="ZIP",
            url="https://x/routes.zip",
        ),
        ResolvedResource(
            dataset_key="travel_times_bluetooth",
            package_id="travel-times-bluetooth",
            name="travel-time-2014",
            kind="travel_times",
            year=2014,
            resource_id="tt-2014",
            format="ZIP",
            url="https://x/2014.zip",
        ),
    ]

    with (
        patch("src.ingestion.ckan.load_dataset_config", return_value=SAMPLE_CONFIG),
        patch("src.ingestion.ckan.CkanClient") as client_cls,
    ):
        client_cls.return_value.resolve_configured_resources.return_value = fake_resources
        exit_code = main(["--config", str(config_path), "--dataset", "travel_times_bluetooth"])

    assert exit_code == 0
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured[0] == (
        "travel_times_bluetooth\tbluetooth-routes-wgs84\tgeo\t-\tZIP\thttps://x/routes.zip"
    )
    assert captured[1] == (
        "travel_times_bluetooth\ttravel-time-2014\ttravel_times\t2014\tZIP\thttps://x/2014.zip"
    )
