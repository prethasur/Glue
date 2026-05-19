"""Create the first fixed BigEarthNet-S2 compositional split."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd


HELDOUT_TILE = "T29SND"
HELDOUT_SEASON = "summer"
TRAIN_CELLS = [
    ("T29SND", "spring"),
    ("T29SNC", "summer"),
    ("T29SND", "autumn"),
]
SEED = 0
SPLIT_COLUMNS = [
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


def split_labels(labels: str) -> list[str]:
    return [label.strip() for label in str(labels).split("|") if label.strip()]


def label_counts(df: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for labels in df["labels"].fillna(""):
        counts.update(split_labels(labels))
    return counts


def label_set(df: pd.DataFrame) -> set[str]:
    return set(label_counts(df))


def distribution(counts: Counter[str], labels: list[str]) -> list[float]:
    total = sum(counts.values())
    if total == 0:
        return [0.0 for _ in labels]
    return [counts[label] / total for label in labels]


def kl_divergence(p: list[float], q: list[float]) -> float:
    return sum(p_i * math.log2(p_i / q_i) for p_i, q_i in zip(p, q) if p_i > 0 and q_i > 0)


def js_divergence(first: Counter[str], second: Counter[str], labels: list[str]) -> float:
    p = distribution(first, labels)
    q = distribution(second, labels)
    midpoint = [(p_i + q_i) / 2 for p_i, q_i in zip(p, q)]
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def cell_mask(df: pd.DataFrame, tile_id: str, season: str) -> pd.Series:
    return (df["tile_id"] == tile_id) & (df["season"] == season)


def shuffled_split(df: pd.DataFrame, frac_first: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed)
    first_size = int(round(len(shuffled) * frac_first))
    first = shuffled.iloc[:first_size].sort_index()
    second = shuffled.iloc[first_size:].sort_index()
    return first, second


def write_split(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[SPLIT_COLUMNS].to_csv(path, index=False)


def build_label_distribution(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, split_df in splits.items():
        counts = label_counts(split_df)
        for label, count in sorted(counts.items()):
            rows.append({"split": split_name, "label": label, "count": count})
    return pd.DataFrame(rows)


def top_labels_text(splits: dict[str, pd.DataFrame], top_n: int = 10) -> list[str]:
    lines: list[str] = []
    for split_name, split_df in splits.items():
        lines.append(f"{split_name}:")
        for label, count in label_counts(split_df).most_common(top_n):
            lines.append(f"  {label}: {count}")
    return lines


def create_split(metadata_csv: Path, output_dir: Path, seed: int) -> str:
    df = pd.read_csv(metadata_csv, dtype={"date": str})
    missing_columns = [column for column in SPLIT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    heldout = df[cell_mask(df, HELDOUT_TILE, HELDOUT_SEASON)].copy()
    train_source = pd.concat(
        [df[cell_mask(df, tile_id, season)].copy() for tile_id, season in TRAIN_CELLS],
        ignore_index=False,
    )

    train_seen, val_seen = shuffled_split(train_source, 0.9, seed)
    oracle_train_unseen, oracle_test_unseen = shuffled_split(heldout, 0.5, seed)
    test_unseen = heldout.sort_index()

    splits = {
        "train_seen": train_seen,
        "val_seen": val_seen,
        "test_unseen": test_unseen,
        "oracle_train_unseen": oracle_train_unseen,
        "oracle_test_unseen": oracle_test_unseen,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        write_split(output_dir / f"{split_name}.csv", split_df)

    all_labels = sorted(label_set(df))
    train_labels = label_set(train_seen)
    test_labels = label_set(test_unseen)
    oracle_test_labels = label_set(oracle_test_unseen)
    missing_test_labels = sorted(test_labels - train_labels)
    missing_oracle_test_labels = sorted(oracle_test_labels - train_labels)
    missing_all_from_train = sorted(set(all_labels) - train_labels)
    jsd = js_divergence(label_counts(train_seen), label_counts(test_unseen), all_labels)

    summary_rows = [
        {
            "split": split_name,
            "sample_count": len(split_df),
            "unique_labels": len(label_set(split_df)),
            "tile_season_cells": ";".join(
                f"{tile}_{season}"
                for tile, season in sorted(set(zip(split_df["tile_id"], split_df["season"])))
            ),
        }
        for split_name, split_df in splits.items()
    ]
    summary_rows.append(
        {
            "split": "metadata",
            "sample_count": len(df),
            "unique_labels": len(all_labels),
            "tile_season_cells": "",
        }
    )
    pd.DataFrame(summary_rows).to_csv(output_dir / "split_summary.csv", index=False)
    build_label_distribution(splits).to_csv(output_dir / "label_distribution_by_split.csv", index=False)

    report_lines = [
        "BigEarthNet-S2 First Compositional Split Report",
        "================================================",
        "",
        f"held-out cell: {HELDOUT_TILE}_{HELDOUT_SEASON}",
        "train cells: " + ", ".join(f"{tile}_{season}" for tile, season in TRAIN_CELLS),
        f"seed: {seed}",
        "",
        "sample counts per split:",
        *[f"  {name}: {len(split_df)}" for name, split_df in splits.items()],
        "",
        "labels present per split:",
        *[f"  {name}: {len(label_set(split_df))}" for name, split_df in splits.items()],
        "",
        "missing-label warnings:",
        f"  labels in test_unseen missing from train_seen: {', '.join(missing_test_labels) if missing_test_labels else 'none'}",
        (
            "  labels in oracle_test_unseen missing from train_seen: "
            f"{', '.join(missing_oracle_test_labels) if missing_oracle_test_labels else 'none'}"
        ),
        f"  labels from full metadata missing from train_seen: {', '.join(missing_all_from_train) if missing_all_from_train else 'none'}",
        "",
        f"JS divergence train_seen vs test_unseen: {jsd:.6f}",
        "",
        "top 10 labels by split:",
        *top_labels_text(splits),
        "",
        "experimental meaning:",
        (
            "  The model sees tile T29SND in spring/autumn and sees summer in tile T29SNC, "
            "but it does not see T29SND_summer during standard adapter training."
        ),
        "",
    ]
    report_text = "\n".join(report_lines)
    (output_dir / "split_report.txt").write_text(report_text, encoding="utf-8")
    return report_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata_csv",
        default=Path("outputs/ben_s2_clean_metadata.csv"),
        type=Path,
        help="Clean BigEarthNet-S2 metadata CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("outputs/ben_s2_splits_first"),
        type=Path,
        help="Output directory for the first compositional split.",
    )
    parser.add_argument("--seed", default=SEED, type=int, help="Fixed random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_csv = args.metadata_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not metadata_csv.is_file():
        print(f"ERROR: metadata CSV not found: {metadata_csv}")
        return 2
    print(create_split(metadata_csv, output_dir, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
