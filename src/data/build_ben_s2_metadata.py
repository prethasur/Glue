"""Join BigEarthNet-S2 parquet metadata to local image folders."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from os import scandir
from pathlib import Path
from typing import Any

import pandas as pd


PATCH_RE = re.compile(
    r"^(?P<sat>S2[AB])_MSIL2A_(?P<date>\d{8})T\d+_N\d+_R\d+_(?P<tile>T\d{2}[A-Z]{3})"
)
RGB_BANDS = ("B02", "B03", "B04")
CLEAN_COLUMNS = [
    "patch_id",
    "patch_dir",
    "B02_path",
    "B03_path",
    "B04_path",
    "labels",
    "tile_id",
    "date",
    "month",
    "season",
]


def season_from_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    if month in {9, 10, 11}:
        return "autumn"
    raise ValueError(f"Invalid month: {month}")


def find_column(columns: list[str], candidates: tuple[str, ...], contains: tuple[str, ...] = ()) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        column_lower = column.lower()
        if any(token in column_lower for token in contains):
            return column
    raise ValueError(f"Could not find required column. Candidates={candidates}, contains={contains}")


def normalize_labels(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value if str(item))
    if hasattr(value, "tolist"):
        return normalize_labels(value.tolist())
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            return normalize_labels(json.loads(text))
        except json.JSONDecodeError:
            return text
    return text


def scan_patch_dirs(data_root: Path) -> dict[str, Path]:
    patch_dirs: dict[str, Path] = {}
    with scandir(data_root) as scene_entries:
        scenes = sorted((entry for entry in scene_entries if entry.is_dir()), key=lambda entry: entry.name)
    for scene in scenes:
        with scandir(scene.path) as patch_entries:
            patches = sorted((entry for entry in patch_entries if entry.is_dir()), key=lambda entry: entry.name)
        for patch in patches:
            if PATCH_RE.match(patch.name):
                patch_dirs[patch.name] = Path(patch.path)
    return patch_dirs


def band_path(patch_dir: Path, patch_id: str, band: str) -> str:
    path = patch_dir / f"{patch_id}_{band}.tif"
    return str(path)


def extract_patch_fields(patch_id: str) -> tuple[str, str, int, str]:
    match = PATCH_RE.match(patch_id)
    if not match:
        raise ValueError(f"Could not parse tile/date from patch_id: {patch_id}")
    date = match.group("date")
    month = int(date[4:6])
    return match.group("tile"), date, month, season_from_month(month)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_clean_metadata(
    metadata: pd.DataFrame,
    patch_id_column: str,
    labels_column: str,
    patch_dirs: dict[str, Path],
) -> tuple[list[dict[str, Any]], set[str], int]:
    rows: list[dict[str, Any]] = []
    parquet_patch_ids: set[str] = set()
    empty_labels = 0

    for record in metadata[[patch_id_column, labels_column]].itertuples(index=False, name=None):
        patch_id = str(record[0])
        labels = normalize_labels(record[1])
        parquet_patch_ids.add(patch_id)
        patch_dir = patch_dirs.get(patch_id)
        if patch_dir is None:
            continue
        tile_id, date, month, season = extract_patch_fields(patch_id)
        if not labels:
            empty_labels += 1
        rows.append(
            {
                "patch_id": patch_id,
                "patch_dir": str(patch_dir),
                "B02_path": band_path(patch_dir, patch_id, "B02"),
                "B03_path": band_path(patch_dir, patch_id, "B03"),
                "B04_path": band_path(patch_dir, patch_id, "B04"),
                "labels": labels,
                "tile_id": tile_id,
                "date": date,
                "month": month,
                "season": season,
            }
        )
    return rows, parquet_patch_ids, empty_labels


def build_tile_season_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row["tile_id"], row["season"]) for row in rows)
    return [
        {"tile_id": tile_id, "season": season, "count": count}
        for (tile_id, season), count in sorted(counts.items())
    ]


def build_label_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        labels = [label for label in str(row["labels"]).split("|") if label]
        for label in labels:
            counts[(label, row["season"])] += 1
    return [
        {"label": label, "season": season, "count": count}
        for (label, season), count in sorted(counts.items())
    ]


def print_summary(
    columns: list[str],
    patch_id_column: str,
    labels_column: str,
    patch_folders_found: int,
    parquet_rows: int,
    joined_rows: int,
    missing_image_folders: int,
    folders_missing_metadata: int,
    empty_labels: int,
    tile_counts: list[dict[str, Any]],
) -> None:
    print("metadata.parquet columns")
    for column in columns:
        print(f"  - {column}")
    print(f"\npatch_id column: {patch_id_column}")
    print(f"labels column: {labels_column}")

    print("\nsummary")
    print(f"  patch folders found: {patch_folders_found}")
    print(f"  rows in metadata.parquet: {parquet_rows}")
    print(f"  successfully joined rows: {joined_rows}")
    print(f"  missing image folders: {missing_image_folders}")
    print(f"  folders missing metadata: {folders_missing_metadata}")
    print(f"  rows with empty labels: {empty_labels}")

    print("\ntop 30 tile_id x season counts")
    top_counts = sorted(tile_counts, key=lambda row: row["count"], reverse=True)[:30]
    for row in top_counts:
        print(f"  {row['tile_id']:6s} {row['season']:6s}: {row['count']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path, help="Path to the BigEarthNet-S2 folder.")
    parser.add_argument(
        "--metadata_parquet",
        type=Path,
        default=None,
        help="Path to metadata.parquet. Defaults to <data_root>/metadata.parquet.",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("outputs"),
        type=Path,
        help="Directory for verification CSV outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    metadata_parquet = (
        args.metadata_parquet.expanduser().resolve()
        if args.metadata_parquet
        else data_root / "metadata.parquet"
    )
    output_dir = args.output_dir.expanduser().resolve()

    if not data_root.is_dir():
        print(f"ERROR: --data_root is not a directory: {data_root}", file=sys.stderr)
        return 2
    if not metadata_parquet.is_file():
        print(f"ERROR: metadata parquet not found: {metadata_parquet}", file=sys.stderr)
        return 2

    metadata = pd.read_parquet(metadata_parquet)
    columns = [str(column) for column in metadata.columns]
    patch_id_column = find_column(columns, ("patch_id", "patchid", "patch_name"), ("patch",))
    labels_column = find_column(columns, ("labels", "label", "class_labels", "land_cover_labels"), ("label",))

    patch_dirs = scan_patch_dirs(data_root)
    clean_rows, parquet_patch_ids, empty_labels = build_clean_metadata(
        metadata,
        patch_id_column,
        labels_column,
        patch_dirs,
    )

    missing_image_folders = len(parquet_patch_ids - set(patch_dirs))
    folders_missing_metadata = len(set(patch_dirs) - parquet_patch_ids)
    tile_counts = build_tile_season_counts(clean_rows)
    label_distribution = build_label_distribution(clean_rows)

    write_csv(output_dir / "ben_s2_clean_metadata.csv", clean_rows, CLEAN_COLUMNS)
    write_csv(output_dir / "tile_season_counts.csv", tile_counts, ["tile_id", "season", "count"])
    write_csv(
        output_dir / "label_distribution_by_season.csv",
        label_distribution,
        ["label", "season", "count"],
    )

    print_summary(
        columns=columns,
        patch_id_column=patch_id_column,
        labels_column=labels_column,
        patch_folders_found=len(patch_dirs),
        parquet_rows=len(metadata),
        joined_rows=len(clean_rows),
        missing_image_folders=missing_image_folders,
        folders_missing_metadata=folders_missing_metadata,
        empty_labels=empty_labels,
        tile_counts=tile_counts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
