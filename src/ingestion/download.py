"""Download Toronto Open Data resources resolved by the CKAN client."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from src.ingestion.ckan import (
    DEFAULT_CONFIG_PATH,
    CkanClient,
    ResolvedResource,
    load_dataset_config,
)

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHUNK_SIZE = 1024 * 1024  # 1 MiB

FORMAT_EXTENSIONS = {
    "ZIP": ".zip",
    "XLSX": ".xlsx",
    "XLS": ".xls",
    "CSV": ".csv",
    "GEOJSON": ".geojson",
    "JSON": ".json",
    "GPKG": ".gpkg",
    "SHP": ".shp",
}


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of attempting to download one resolved resource.

    Attributes:
        resource: The CKAN resource that was targeted.
        path: Local destination path for the file.
        status: One of ``downloaded``, ``skipped``, or ``failed``.
        bytes_written: Number of bytes written on a successful download.
        error: Error message when ``status`` is ``failed``; otherwise ``None``.
    """

    resource: ResolvedResource
    path: Path
    status: str
    bytes_written: int = 0
    error: str | None = None


def extension_for_resource(resource: ResolvedResource) -> str:
    """Infer a file extension for a resolved resource.

    Args:
        resource: Resolved CKAN resource with ``format`` and ``url``.

    Returns:
        Extension including the leading dot (e.g. ``.zip``). Falls back to the
        URL path suffix when the CKAN format is unknown, else ``.bin``.
    """
    fmt = (resource.format or "").upper()
    if fmt in FORMAT_EXTENSIONS:
        return FORMAT_EXTENSIONS[fmt]

    url_path = unquote(urlparse(resource.url).path)
    suffix = Path(url_path).suffix
    if suffix:
        return suffix.lower()
    return ".bin"


def local_path_for_resource(resource: ResolvedResource, raw_dir: Path) -> Path:
    """Build the local path where a resource should be stored.

    Args:
        resource: Resolved CKAN resource.
        raw_dir: Root directory for raw downloads (typically ``data/raw``).

    Returns:
        Path of the form ``{raw_dir}/{dataset_key}/{resource_name}{ext}``.
    """
    filename = f"{resource.name}{extension_for_resource(resource)}"
    return raw_dir / resource.dataset_key / filename


def download_resource(
    resource: ResolvedResource,
    destination: Path,
    session: requests.Session,
    *,
    force: bool = False,
    chunk_size: int = CHUNK_SIZE,
    timeout: int = 120,
) -> DownloadResult:
    """Stream one remote resource to disk.

    Args:
        resource: Resolved CKAN resource with a download URL.
        destination: Local file path to write.
        session: Shared ``requests.Session`` for connection reuse.
        force: If ``True``, re-download even when ``destination`` already exists.
        chunk_size: Streaming chunk size in bytes.
        timeout: Per-request timeout in seconds.

    Returns:
        ``DownloadResult`` describing whether the file was downloaded, skipped,
        or failed.

    Notes:
        Writes to a temporary ``*.partial`` file beside the destination, then
        atomically renames on success so interrupted runs do not leave a
        corrupt final file.
    """
    if destination.exists() and not force:
        return DownloadResult(
            resource=resource,
            path=destination,
            status="skipped",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    bytes_written = 0

    try:
        with session.get(resource.url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_written += len(chunk)
        partial.replace(destination)
    except Exception as exc:  # noqa: BLE001 - surface any transfer failure as result
        if partial.exists():
            partial.unlink(missing_ok=True)
        return DownloadResult(
            resource=resource,
            path=destination,
            status="failed",
            bytes_written=bytes_written,
            error=str(exc),
        )

    return DownloadResult(
        resource=resource,
        path=destination,
        status="downloaded",
        bytes_written=bytes_written,
    )


def download_resources(
    resources: list[ResolvedResource],
    raw_dir: Path,
    session: requests.Session,
    *,
    force: bool = False,
    kinds: set[str] | None = None,
) -> list[DownloadResult]:
    """Download many resolved resources into the raw data directory.

    Args:
        resources: Resources previously resolved by ``CkanClient``.
        raw_dir: Root directory for raw downloads.
        session: Shared HTTP session.
        force: Re-download existing files when ``True``.
        kinds: Optional set of resource kinds to keep
            (e.g. ``{"travel_times", "geo"}``). If ``None``, download all.

    Returns:
        One ``DownloadResult`` per selected resource, in input order.
    """
    results: list[DownloadResult] = []
    for resource in resources:
        if kinds is not None and resource.kind not in kinds:
            continue
        destination = local_path_for_resource(resource, raw_dir)
        results.append(
            download_resource(
                resource,
                destination,
                session,
                force=force,
            )
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for downloading configured resources.

    Args:
        None.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Download Toronto Open Data resources into data/raw."
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
        help="Dataset key to download (repeatable). Defaults to all.",
    )
    parser.add_argument(
        "--kind",
        action="append",
        dest="kinds",
        help="Resource kind filter (repeatable), e.g. travel_times, geo, docs.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Destination root for raw files (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: resolve configured resources and download them.

    Args:
        argv: Optional argument list (without the program name). If ``None``,
            ``sys.argv[1:]`` is used by argparse.

    Returns:
        ``0`` if every attempted download succeeded or was skipped; ``1`` if
        any download failed.
    """
    args = _build_parser().parse_args(argv)
    config = load_dataset_config(args.config)
    client = CkanClient(config["ckan_base_url"])
    resources = client.resolve_configured_resources(config, args.datasets)
    kinds = set(args.kinds) if args.kinds else None

    results = download_resources(
        resources,
        args.raw_dir,
        client.session,
        force=args.force,
        kinds=kinds,
    )

    failures = 0
    for result in results:
        year = result.resource.year if result.resource.year is not None else "-"
        print(
            f"{result.status}\t{result.resource.dataset_key}\t"
            f"{result.resource.name}\t{year}\t{result.path}"
            + (f"\t{result.error}" if result.error else "")
        )
        if result.status == "failed":
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
