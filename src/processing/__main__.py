"""CLI for raw → interim → processed normalization steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.processing.civic import normalize_civic_days
from src.processing.events import normalize_events
from src.processing.panel import build_route_time_panel
from src.processing.paths import DEFAULT_INTERIM_DIR, DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR
from src.processing.routes import normalize_routes
from src.processing.travel_times import normalize_travel_times
from src.processing.weather import normalize_weather

STEPS = (
    "travel_times",
    "weather",
    "civic",
    "events",
    "routes",
    "panel",
    "all",
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the processing CLI parser."""
    parser = argparse.ArgumentParser(
        description="Normalize Toronto mobility datasets into interim/processed Parquet tables."
    )
    parser.add_argument(
        "--step",
        choices=STEPS,
        default="all",
        help="Which normalization step to run.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--year",
        action="append",
        type=int,
        dest="years",
        help="Optional travel-time year filter (repeatable). Also applied to panel.",
    )
    return parser


def run_step(step: str, args: argparse.Namespace) -> int:
    """Execute one named processing step and print the output path."""
    if step == "travel_times":
        path = normalize_travel_times(args.raw_dir, args.interim_dir, years=args.years)
        print(f"wrote\t{path}")
        return 0
    if step == "weather":
        path = normalize_weather(args.raw_dir, args.interim_dir)
        print(f"wrote\t{path}")
        return 0
    if step == "civic":
        path = normalize_civic_days(interim_dir=args.interim_dir)
        print(f"wrote\t{path}")
        return 0
    if step == "events":
        path = normalize_events(args.raw_dir, args.interim_dir)
        print(f"wrote\t{path}")
        return 0
    if step == "routes":
        path = normalize_routes(args.raw_dir, args.interim_dir)
        print(f"wrote\t{path}")
        return 0
    if step == "panel":
        path, qa = build_route_time_panel(
            args.interim_dir,
            args.processed_dir,
            years=args.years,
        )
        print(f"wrote\t{path}")
        print(json.dumps(qa, indent=2))
        return 0
    if step == "all":
        for name in ("travel_times", "weather", "civic", "events", "routes", "panel"):
            code = run_step(name, args)
            if code != 0:
                return code
        return 0
    raise ValueError(f"Unknown step: {step}")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the normalization pipeline."""
    args = _build_parser().parse_args(argv)
    return run_step(args.step, args)


if __name__ == "__main__":
    sys.exit(main())
