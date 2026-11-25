"""Recommend label-aware BigEarthNet-S2 held-out compositional split candidates."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SEED = 0
MIN_HELDOUT_LABELS = 10
MIN_ORACLE_TEST_LABELS = 10
PREFERRED_MIN_COUNT = 5000
PREFERRED_MIN_LABELS = 12
PREFERRED_JSD_LOW = 0.03
PREFERRED_JSD_HIGH = 0.12
TOP_PRINT = 20


@dataclass(frozen=True)
class CellStats:
    tile_id: str
    season: str
    indices: list[int]
    sample_count: int
    label_counts: Counter[str]

    @property
    def key(self) -> tuple[str, str]:
        return (self.tile_id, self.season)

    @property
    def name(self) -> str:
        return f"{self.tile_id}_{self.season}"

    @property
    def labels(self) -> set[str]:
        return set(self.label_counts)


def split_labels(labels: str) -> list[str]:
    return [label.strip() for label in str(labels).split("|") if label.strip()]


def label_counts_for_rows(labels: pd.Series) -> Counter[str]:
    counts: Counter[str] = Counter()
    for label_string in labels.fillna(""):
        counts.update(split_labels(label_string))
    return counts


def distribution(label_counts: Counter[str], label_vocab: list[str]) -> list[float]:
    total = sum(label_counts.values())
    if total == 0:
        return [0.0 for _ in label_vocab]
    return [label_counts[label] / total for label in label_vocab]


def kl_divergence(p: list[float], q: list[float]) -> float:
    return sum(p_i * math.log2(p_i / q_i) for p_i, q_i in zip(p, q) if p_i > 0 and q_i > 0)


def js_divergence(first: Counter[str], second: Counter[str], label_vocab: list[str]) -> float:
    p = distribution(first, label_vocab)
    q = distribution(second, label_vocab)
    midpoint = [(p_i + q_i) / 2 for p_i, q_i in zip(p, q)]
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def combined_label_counts(cells: list[CellStats]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for cell in cells:
        counts.update(cell.label_counts)
    return counts


def load_cells(df: pd.DataFrame) -> tuple[dict[tuple[str, str], CellStats], list[str]]:
    all_labels = sorted({label for labels in df["labels"].fillna("") for label in split_labels(labels)})
    cells: dict[tuple[str, str], CellStats] = {}
    for (tile_id, season), group in df.groupby(["tile_id", "season"], sort=False):
        label_counts = label_counts_for_rows(group["labels"])
        cells[(str(tile_id), str(season))] = CellStats(
            tile_id=str(tile_id),
            season=str(season),
            indices=list(group.index),
            sample_count=len(group),
            label_counts=label_counts,
        )
    return cells, all_labels


def oracle_test_label_counts(df: pd.DataFrame, cell: CellStats, seed: int) -> Counter[str]:
    heldout = df.loc[cell.indices]
    shuffled = heldout.sample(frac=1.0, random_state=seed)
    oracle_train_size = int(round(len(shuffled) * 0.8))
    oracle_test = shuffled.iloc[oracle_train_size:]
    return label_counts_for_rows(oracle_test["labels"])


def choose_train_cells(
    heldout: CellStats,
    cells: dict[tuple[str, str], CellStats],
    label_vocab: list[str],
) -> list[CellStats]:
    selected: list[CellStats] = []

    same_tile = [
        cell for cell in cells.values()
        if cell.tile_id == heldout.tile_id and cell.season != heldout.season
    ]
    if same_tile:
        selected.append(max(same_tile, key=lambda cell: (len(cell.labels & heldout.labels), cell.sample_count)))

    same_season = [
        cell for cell in cells.values()
        if cell.season == heldout.season and cell.tile_id != heldout.tile_id
    ]
    if same_season:
        selected_keys = {cell.key for cell in selected}
        selected.append(
            max(
                [cell for cell in same_season if cell.key not in selected_keys],
                key=lambda cell: (len(cell.labels & heldout.labels), cell.sample_count),
            )
        )

    while len(selected) < 3:
        selected_keys = {cell.key for cell in selected}
        pool = [cell for cell in cells.values() if cell.key != heldout.key and cell.key not in selected_keys]
        if not pool:
            break
        best_cell = min(
            pool,
            key=lambda cell: (
                len(heldout.labels - set(combined_label_counts(selected + [cell]))) * 100,
                len(set(label_vocab) - set(combined_label_counts(selected + [cell]))) * 10,
                js_divergence(combined_label_counts(selected + [cell]), heldout.label_counts, label_vocab),
                -cell.sample_count,
            ),
        )
        selected.append(best_cell)

    return selected


def preference_rank(row: dict[str, object]) -> tuple[int, int, int, float, int]:
    jsd = float(row["label_js_divergence"])
    preferred_jsd = PREFERRED_JSD_LOW <= jsd <= PREFERRED_JSD_HIGH
    preferred = (
        int(row["heldout_sample_count"]) >= PREFERRED_MIN_COUNT
        and int(row["heldout_labels_present"]) >= PREFERRED_MIN_LABELS
        and str(row["missing_labels_from_train"]) == ""
        and preferred_jsd
    )
    distance_to_band = 0.0 if preferred_jsd else min(abs(jsd - PREFERRED_JSD_LOW), abs(jsd - PREFERRED_JSD_HIGH))
    return (
        0 if preferred else 1,
        len(str(row["missing_labels_from_train"]).split("|")) if str(row["missing_labels_from_train"]) else 0,
        -int(row["heldout_labels_present"]),
        distance_to_band,
        -int(row["heldout_sample_count"]),
    )


def recommend(metadata_csv: Path, output_csv: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv, usecols=["tile_id", "season", "labels"])
    cells, label_vocab = load_cells(df)
    rows: list[dict[str, object]] = []

    for heldout in cells.values():
        heldout_label_count = len(heldout.labels)
        if heldout_label_count < MIN_HELDOUT_LABELS:
            continue
        oracle_counts = oracle_test_label_counts(df, heldout, seed)
        oracle_test_labels = set(oracle_counts)
        if len(oracle_test_labels) < MIN_ORACLE_TEST_LABELS:
            continue

        train_cells = choose_train_cells(heldout, cells, label_vocab)
        train_counts = combined_label_counts(train_cells)
        train_labels = set(train_counts)
        missing = sorted(heldout.labels - train_labels)
        jsd = js_divergence(train_counts, heldout.label_counts, label_vocab)
        per_label_counts = ";".join(
            f"{label}:{heldout.label_counts[label]}" for label in sorted(heldout.label_counts)
        )
        rows.append(
            {
                "heldout_cell": heldout.name,
                "heldout_tile_id": heldout.tile_id,
                "heldout_season": heldout.season,
                "heldout_sample_count": heldout.sample_count,
                "heldout_labels_present": heldout_label_count,
                "oracle_test_labels_present": len(oracle_test_labels),
                "oracle_80_20_keeps_at_least_10_labels": len(oracle_test_labels) >= 10,
                "heldout_per_label_counts": per_label_counts,
                "train_cells": ";".join(cell.name for cell in train_cells),
                "train_sample_count": sum(cell.sample_count for cell in train_cells),
                "train_labels_present": len(train_labels),
                "missing_labels_from_train": "|".join(missing),
                "label_js_divergence": round(jsd, 6),
            }
        )

    rows.sort(key=preference_rank)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    result = pd.DataFrame(rows)
    columns = ["rank"] + [column for column in result.columns if column != "rank"]
    result = result[columns]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata_csv",
        default=Path("outputs/ben_s2_clean_metadata.csv"),
        type=Path,
        help="Clean BigEarthNet-S2 metadata CSV.",
    )
    parser.add_argument(
        "--output_csv",
        default=Path("outputs/ben_s2_label_aware_split_candidates.csv"),
        type=Path,
        help="CSV path for label-aware candidate recommendations.",
    )
    parser.add_argument("--seed", default=SEED, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_csv = args.metadata_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    if not metadata_csv.is_file():
        print(f"ERROR: metadata CSV not found: {metadata_csv}")
        return 2
    result = recommend(metadata_csv, output_csv, args.seed)
    display_columns = [
        "heldout_cell",
        "heldout_sample_count",
        "heldout_labels_present",
        "oracle_test_labels_present",
        "train_cells",
        "train_sample_count",
        "missing_labels_from_train",
        "label_js_divergence",
    ]
    print(result[display_columns].head(TOP_PRINT).to_string(index=False))
    print(f"\nSaved: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

