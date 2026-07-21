"""Resolve Toronto Open Data (CKAN) resources from the local dataset config."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "toronto_datasets.yaml"
)


@dataclass(frozen=True)
class ResolvedResource:
    """Immutable record of one CKAN resource matched to a local config entry.

    Attributes:
        dataset_key: Key under ``config["datasets"]`` (e.g. ``travel_times_bluetooth``).
        package_id: CKAN package / dataset id used in the API call.
        name: Resource name as published on the portal and listed in config.
        kind: Local semantic label from config (``geo``, ``travel_times``, ``docs``).
        year: Optional year from config for time-partitioned resources.
        resource_id: CKAN resource UUID.
        format: Uppercased file format from CKAN (e.g. ``ZIP``, ``XLSX``).
        url: Direct download URL for the resource.
    """

    dataset_key: str
    package_id: str
    name: str
    kind: str
    year: int | None
    resource_id: str
    format: str
    url: str


def load_dataset_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the Toronto datasets YAML config.

    Args:
        path: Filesystem path to ``toronto_datasets.yaml``. Defaults to the
            repo ``configs/toronto_datasets.yaml`` resolved from this file.

    Returns:
        Parsed config mapping with at least ``ckan_base_url`` and ``datasets``.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in config: {config_path}")
    return config


class CkanClient:
    """Thin client for Toronto Open Data CKAN Action API lookups."""

    def __init__(self, base_url: str, session: requests.Session | None = None) -> None:
        """Create a client bound to a CKAN Action API base URL.

        Args:
            base_url: CKAN action endpoint root, e.g.
                ``https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/``.
                Trailing slash is normalized.
            session: Optional ``requests.Session`` to reuse. If omitted, a new
                session is created for this client instance.
        """
        self.base_url = base_url.rstrip("/") + "/"
        self.session = session or requests.Session()

    def package_show(self, package_id: str) -> dict[str, Any]:
        """Fetch full package metadata (including all resources) from CKAN.

        Args:
            package_id: CKAN package id or name
                (e.g. ``travel-times-bluetooth``).

        Returns:
            The CKAN ``result`` object for the package (title, resources, etc.).
        """
        response = self.session.get(
            self.base_url + "package_show",
            params={"id": package_id},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"CKAN package_show failed for {package_id}: {payload}")
        return payload["result"]

    def resolve_configured_resources(
        self,
        config: dict[str, Any],
        dataset_keys: list[str] | None = None,
    ) -> list[ResolvedResource]:
        """Map local config resource names to live CKAN download URLs.

        For each selected dataset, calls ``package_show`` once, then matches
        configured resource names against the remote resource list.

        Args:
            config: Parsed YAML config from ``load_dataset_config``.
            dataset_keys: Optional subset of keys under ``config["datasets"]``.
                If ``None``, all configured datasets are resolved.

        Returns:
            Flat list of ``ResolvedResource`` rows in config order
            (dataset order, then resource order within each dataset).
        """
        datasets = config.get("datasets") or {}
        selected = dataset_keys or list(datasets)
        resolved: list[ResolvedResource] = []

        for dataset_key in selected:
            if dataset_key not in datasets:
                raise KeyError(f"Unknown dataset key: {dataset_key}")

            dataset = datasets[dataset_key]
            package_id = dataset["package_id"]
            # One network round-trip per package; index resources for O(1) lookup.
            package = self.package_show(package_id)
            resources_by_name = {
                resource["name"]: resource for resource in package.get("resources", [])
            }

            for resource_cfg in dataset.get("resources", []):
                name = resource_cfg["name"]
                remote = resources_by_name.get(name)
                if remote is None:
                    available = ", ".join(sorted(resources_by_name)) or "(none)"
                    raise KeyError(
                        f"Resource '{name}' not found in package '{package_id}'. "
                        f"Available: {available}"
                    )

                resolved.append(
                    ResolvedResource(
                        dataset_key=dataset_key,
                        package_id=package_id,
                        name=name,
                        kind=resource_cfg.get("kind", "unknown"),
                        year=resource_cfg.get("year"),
                        resource_id=remote["id"],
                        format=(remote.get("format") or "").upper(),
                        url=remote["url"],
                    )
                )

        return resolved


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for listing resolved resources.

    Args:
        None.

    Returns:
        Configured ``argparse.ArgumentParser`` with ``--config`` and
        repeatable ``--dataset`` flags.
    """
    parser = argparse.ArgumentParser(
        description="List Toronto Open Data resources resolved from config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to toronto_datasets.yaml",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset key to resolve (repeatable). Defaults to all.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: resolve configured resources and print them as TSV.

    Args:
        argv: Optional argument list (without the program name). If ``None``,
            ``sys.argv[1:]`` is used by argparse.

    Returns:
        Process exit code: ``0`` on success.
    """
    args = _build_parser().parse_args(argv)
    config = load_dataset_config(args.config)
    client = CkanClient(config["ckan_base_url"])
    resources = client.resolve_configured_resources(config, args.datasets)

    for resource in resources:
        year = resource.year if resource.year is not None else "-"
        print(
            f"{resource.dataset_key}\t{resource.name}\t{resource.kind}\t"
            f"{year}\t{resource.format}\t{resource.url}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
