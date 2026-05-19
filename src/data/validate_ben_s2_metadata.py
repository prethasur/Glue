"""Validate the clean BigEarthNet-S2 metadata CSV before split generation."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
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
PATH_COLUMNS = ["B02_path", "B03_path", "B04_path"]
TILE_RE = re.compile(r"^T\d{2}[A-Z]{3}$")
SEASON_BY_MONTH = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}


def split_labels(labels: str) -> Iterable[str]:
    for label in str(labels).split("|"):
        label = label.strip()
        if label:
            yield label


def count_missing_paths(metadata_csv: Path, required_columns: list[str]) -> dict[str, int]:
    missing = {column: 0 for column in PATH_COLUMNS}
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return missing
        for row in reader:
            for column in PATH_COLUMNS:
                path_value = row.get(column, "")
                if column in required_columns and (not path_value or not Path(path_value).is_file()):
                    missing[column] += 1
    return missing


def format_section(title: str, lines: list[str]) -> list[str]:
    return [title, "-" * len(title), *lines, ""]


def validate(metadata_csv: Path, output_dir: Path, random_seed: int) -> str:
    df = pd.read_csv(metadata_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = len(df)
    existing_required = [column for column in REQUIRED_COLUMNS if column in df.columns]
    missing_required = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    missing_values: dict[str, int | str] = {}
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            missing_values[column] = "COLUMN_MISSING"
            continue
        missing_values[column] = int(df[column].isna().sum() + (df[column].astype(str).str.strip() == "").sum())

    missing_paths = count_missing_paths(metadata_csv, existing_required)

    tile_valid = df["tile_id"].astype(str).str.match(TILE_RE) if "tile_id" in df.columns else pd.Series([], dtype=bool)
    invalid_tile_count = int((~tile_valid).sum()) if len(tile_valid) else total_rows

    month_values = pd.to_numeric(df["month"], errors="coerce") if "month" in df.columns else pd.Series([], dtype=float)
    expected_seasons = month_values.map(SEASON_BY_MONTH)
    season_values = df["season"].astype(str) if "season" in df.columns else pd.Series([], dtype=str)
    invalid_season_mapping = int((expected_seasons != season_values).sum()) if len(expected_seasons) else total_rows
    non_null_months = month_values.dropna().astype(int)
    invalid_month_count = int(month_values.isna().sum() + (~non_null_months.between(1, 12)).sum())

    month_counts = (
        month_values.dropna().astype(int).value_counts().sort_index()
        if len(month_values)
        else pd.Series(dtype="int64")
    )
    season_counts = df["season"].value_counts().sort_index() if "season" in df.columns else pd.Series(dtype="int64")

    top_tile_season = (
        df.groupby(["tile_id", "season"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top_tile_season.head(50).to_csv(output_dir / "top_tile_season_cells.csv", index=False)

    label_counts: Counter[str] = Counter()
    if "labels" in df.columns:
        for labels in df["labels"].fillna(""):
            label_counts.update(split_labels(labels))
    top_labels = pd.DataFrame(label_counts.most_common(30), columns=["label", "count"])
    top_labels.to_csv(output_dir / "top_label_counts.csv", index=False)

    sample_columns = ["patch_id", "tile_id", "date", "month", "season", "labels", "B02_path"]
    sample_df = df[sample_columns].sample(n=min(10, total_rows), random_state=random_seed)

    report: list[str] = []
    report.extend(format_section("Input", [f"metadata_csv: {metadata_csv}", f"output_dir: {output_dir}"]))
    report.extend(format_section("Total Rows", [str(total_rows)]))
    report.extend(
        format_section(
            "Required Columns",
            [
                f"present: {', '.join(existing_required)}",
                f"missing: {', '.join(missing_required) if missing_required else 'none'}",
            ],
        )
    )
    report.extend(
        format_section(
            "Missing Or Null Values",
            [f"{column}: {missing_values[column]}" for column in REQUIRED_COLUMNS],
        )
    )
    report.extend(
        format_section(
            "Missing Image Paths",
            [f"{column}: {missing_paths[column]}" for column in PATH_COLUMNS],
        )
    )
    report.extend(
        format_section(
            "Tile Pattern Check",
            [
                "expected pattern: ^T\\d{2}[A-Z]{3}$",
                f"invalid tile_id rows: {invalid_tile_count}",
            ],
        )
    )
    report.extend(
        format_section(
            "Season Mapping Check",
            [
                "winter: Dec/Jan/Feb",
                "spring: Mar/Apr/May",
                "summer: Jun/Jul/Aug",
                "autumn: Sep/Oct/Nov",
                f"invalid month rows: {invalid_month_count}",
                f"invalid season mapping rows: {invalid_season_mapping}",
            ],
        )
    )
    report.extend(format_section("Month Counts", [f"{month}: {count}" for month, count in month_counts.items()]))
    report.extend(format_section("Season Counts", [f"{season}: {count}" for season, count in season_counts.items()]))
    report.extend(
        format_section(
            "Top 50 Tile x Season Cells",
            [
                f"{row.tile_id} {row.season}: {row.count}"
                for row in top_tile_season.head(50).itertuples(index=False)
            ],
        )
    )
    report.extend(
        format_section(
            "Label Counts",
            [
                f"unique labels: {len(label_counts)}",
                *[f"{label}: {count}" for label, count in label_counts.most_common(30)],
            ],
        )
    )
    report.extend(
        format_section(
            "Random Sample Rows",
            sample_df.to_string(index=False).splitlines(),
        )
    )

    report_text = "\n".join(report)
    report_path = output_dir / "ben_s2_metadata_validation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata_csv",
        default=Path("outputs/ben_s2_clean_metadata.csv"),
        type=Path,
        help="Clean metadata CSV to validate.",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("outputs"),
        type=Path,
        help="Directory for validation report artifacts.",
    )
    parser.add_argument("--random_seed", default=42, type=int, help="Seed for reproducible random row samples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_csv = args.metadata_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not metadata_csv.is_file():
        print(f"ERROR: metadata CSV not found: {metadata_csv}")
        return 2
    print(validate(metadata_csv, output_dir, args.random_seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
