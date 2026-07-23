"""Paths and shared constants for interim/processed normalization."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_INTERIM_DIR = REPO_ROOT / "data" / "interim"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
LOCAL_TZ = "America/Toronto"
