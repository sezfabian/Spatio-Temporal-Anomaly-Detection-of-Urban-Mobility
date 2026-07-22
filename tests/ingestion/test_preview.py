"""Unit tests for src.ingestion.preview."""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.ingestion.preview import (
    format_preview,
    list_raw_files,
    main,
    preview_csv,
    preview_file,
    preview_zip,
)


def test_list_raw_files_skips_gitkeep_and_partial(tmp_path: Path) -> None:
    dataset = tmp_path / "travel_times_bluetooth"
    dataset.mkdir()
    (dataset / ".gitkeep").write_text("", encoding="utf-8")
    (dataset / "keep.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (dataset / "keep.csv.partial").write_text("tmp", encoding="utf-8")

    files = list_raw_files(tmp_path, "travel_times_bluetooth")
    assert files == [dataset / "keep.csv"]


def test_preview_csv_samples_rows(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text(
        "resultId,timeInSeconds,count\nJ_I,57,24\nJ_I,60,39\n",
        encoding="utf-8",
    )

    preview = preview_csv(path, n_rows=1)
    assert preview.columns == ["resultId", "timeInSeconds", "count"]
    assert preview.rows == [["J_I", "57", "24"]]


def test_preview_zip_samples_csv_member(tmp_path: Path) -> None:
    path = tmp_path / "travel-time-2014.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "travel-time-2014.csv",
            "resultId,timeInSeconds,count,updated\nJ_I,57,24,2014-01-02T14:35:00-05\n",
        )
        archive.writestr("__MACOSX/._travel-time-2014.csv", "junk")

    preview = preview_zip(path, n_rows=1)
    assert len(preview.members) == 1
    assert preview.columns == ["resultId", "timeInSeconds", "count", "updated"]
    assert preview.rows[0][0] == "J_I"
    assert preview.extras["sampled_member"] == "travel-time-2014.csv"


def test_preview_file_dispatches_csv(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    preview = preview_file(path)
    assert preview.kind == "csv"


def test_format_preview_includes_table(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    text = format_preview(preview_csv(path))
    assert "## " in text
    assert "columns: a, b" in text
    assert "1 | 2" in text


def test_main_returns_1_when_empty(tmp_path: Path, capsys) -> None:
    code = main(["--raw-dir", str(tmp_path), "--dataset", "missing"])
    assert code == 1
    assert "No files found" in capsys.readouterr().err


def test_main_prints_preview(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "a.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    code = main(["--raw-dir", str(tmp_path), "--dataset", "ds", "--rows", "1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "a.csv" in out
    assert "columns: a, b" in out
